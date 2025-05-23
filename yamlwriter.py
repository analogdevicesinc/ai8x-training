###################################################################################################
#
# Copyright (C) 2022-2024 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
Generate YAML template from PyTorch model

ALPHA version

TODO: Implement smart processor and data memory allocator (out_offset, in_offset)
TODO: Implement dilation
TODO: Testing

NOTE: This code partially depends on ai8x.py:
      - Quantization information is expected in '.weight_bits'.
"""
import argparse
import os
import warnings
from typing import Any, Callable, Dict, List, Optional, OrderedDict, Tuple, Union

import torch

import distiller

import ai8x
import devices


def allocate_offset(
        _layer_name: str,
        _processor_map: int,
        prev_offset: int = 0,
) -> int:
    """
    Find a data memory offset for the given layer. Currently just uses "ping-pong" from 0 to
    half of the data memory.
    TODO: Implement an algorithm that interacts with weight memories and processor selection,
    and understands element-wise data.
    """
    assert ai8x.dev is not None
    HALF_DATA = 0x4000 if ai8x.dev.device == 85 else 0xa000

    return HALF_DATA if prev_offset == 0 else 0


# pylint: disable=too-many-branches, too-many-statements, too-many-locals
def create(
        model: Any,
        dataset: str,
        arch: str,
        hwc: bool = False,
        use_fifos: bool = False,
        move_l0: bool = False,
        filename: str = 'template.yaml',
        qat_policy: Optional[str] = None,
        verbose=True,
) -> None:
    """
    Create YAML template
    """
    assert ai8x.dev is not None
    if ai8x.dev.device == 85:
        MAX_PIXELS = 8192
    elif ai8x.dev.device == 87:
        MAX_PIXELS = 20480
    else:
        print(f'Unknown device {ai8x.dev.device}')
        return

    # Filter warnings
    s0 = 'The shape inference of prim::Constant type is missing, so it may result in wrong '\
         'shape inference for the exported graph. Please consider adding it in symbolic function.'
    warnings.filterwarnings(action='ignore', message=s0)
    s1 = 'Constant folding - Only steps=1 can be constant folded for opset >= 10 ' \
         'onnx::Slice op. Constant folding not applied.'
    warnings.filterwarnings(action='ignore', message=s1)

    MAX_PROC = 64

    model = distiller.make_non_parallel_copy(model)  # type: ignore[attr-defined]

    # Replace unsupported BitwiseOr and BitwiseXor with an Add() and record the locations.
    # After ONNX conversion, put them back. This is necessary until PyTorch ONNX export supports
    # BitwiseOr/BitwiseXor.
    bitwise_replacements: List[Tuple[str, str]] = []

    def replace_bitwise(m):
        for attr_str in dir(m):
            target_attr = getattr(m, attr_str)
            if isinstance(target_attr, (ai8x.BitwiseXor, ai8x.BitwiseOr)):
                print(attr_str)
                source = 'bitwise_or' if isinstance(target_attr, ai8x.BitwiseOr) else 'bitwise_xor'
                bitwise_replacements.append((attr_str, source))
                setattr(m, attr_str, ai8x.Add())

    model.apply(replace_bitwise)

    # Apply the QAT policy early to set weight_bits
    ai8x.fuse_bn_layers(model)
    if qat_policy is not None:
        ai8x.initiate_qat(model, qat_policy, export=True)
    ai8x.onnx_export_prep(model, simplify=False, remove_clamp=False)

    dummy_input = distiller.get_dummy_input(  # type: ignore[attr-defined]
        dataset=None,
        device=distiller.model_device(model),  # type: ignore[attr-defined]
        input_shape=None,
    )
    g = distiller.SummaryGraph(model, dummy_input, True, '#')  # type: ignore[attr-defined]

    # Get the input/output dimensions
    shapes: Dict[str, Tuple] = {}
    for i, param in g.params.items():
        shape = param['shape']
        if len(shape) > 1:
            shapes[i] = shape[1:]  # Remove batch dimension

    # all_ops is the main dictionary for the layers
    all_ops: OrderedDict[str, Dict[str, Any]] = g.ops

    # Put the bitwise operations back
    for (attr_str, bitwise_op) in bitwise_replacements:
        assert all_ops[attr_str]['type'] == 'Add'
        all_ops[attr_str]['type'] = 'BitwiseOr' if bitwise_op == 'bitwise_or' else 'BitwiseXor'

    # ONNX operators for convolution:
    convolution_ops: Tuple[str, ...] = ('Conv', 'ConvTranspose', 'Gemm', 'MatMul')

    def canonical_name(
            s: str,
    ) -> str:
        """
        Return the canonical name for this layer
        """
        separator = s.rfind('_MatMul_1')
        if separator > 0:
            s = s[:separator]
        separator = s.rfind('.')
        if separator > 0:
            if s[separator + 1:] == 'op':
                s = s[:separator]
        return f'layer_{s}' if s.isnumeric() else s

    def ignore_layer(
            _name: str,
            layer: Dict[str, Any],
    ) -> bool:
        """
        Remove unnecessary layers from the graph
        """
        if not layer['inputs'] or not layer['outputs']:
            return True  # Not consuming or producing anything, useless layer
        return layer['type'] not in convolution_ops and layer['type'] not in (
            'Add', 'Sub', 'BitwiseOr', 'BitwiseXor',  # element-wise
            'MaxPool', 'AveragePool',
            'Abs', 'Relu',
        )

    def follow_input(
            trace_in: List[str],
            condition: Callable[[str], bool],
            replace: Tuple[str, str],
    ) -> List[str]:
        """
        Trace a list of inputs back upstream and stop when condition is met
        """
        results: List[str] = []
        while len(trace_in) > 0:
            next_trace: List[str] = []
            for t in trace_in:
                if condition(t):
                    results.append(t.replace(replace[0], replace[1]))
                else:
                    for n in all_ops:
                        if t in all_ops[n]['outputs']:
                            next_trace += all_ops[n]['inputs']
            trace_in = next_trace

        return results

    # 1 - Set output_width to 32 where needed
    # Trace all "-32768" constants to the next 'Min' operation (there may be more than one)
    results: List[str] = []

    for n in all_ops:
        if all_ops[n]['type'] in convolution_ops:
            # Ignore weights/biases in the input list
            # These operations have 2 or three arguments: data/weights/biases(optional)
            all_ops[n]['weights'] = follow_input([all_ops[n]['inputs'][1]],
                                                 lambda s: s in shapes, ('#', '.'))
            if len(all_ops[n]['inputs']) == 3:
                all_ops[n]['biases'] = follow_input([all_ops[n]['inputs'][2]],
                                                    lambda s: s in shapes, ('#', '.'))
            # Remove weights and biases from the inputs for this operation
            all_ops[n]['inputs'] = [all_ops[n]['inputs'][0]]

        if all_ops[n]['type'] != 'Constant' or 'value' not in all_ops[n]['attrs'] \
           or all_ops[n]['attrs']['value'].dim() != 0 or all_ops[n]['attrs']['value'] != -32768:
            continue
        trace_out: List[str] = all_ops[n]['outputs']
        while len(trace_out) > 0:
            next_trace: List[str] = []
            for t in trace_out:
                for p in all_ops:
                    if t in all_ops[p]['inputs']:
                        if all_ops[p]['type'] in ('Min', 'Clip'):
                            results.append(p)
                        else:
                            next_trace += all_ops[p]['outputs']
            trace_out = next_trace
    # Trace the inputs of these layers back to the previous convolution
    for n in results:
        trace_in: List[str] = all_ops[n]['inputs']
        while len(trace_in) > 0:
            next_trace = []
            for t in trace_in:
                for p in all_ops:
                    if t in all_ops[p]['outputs']:
                        if all_ops[p]['type'] in convolution_ops:
                            all_ops[p]['wide'] = True
                        else:
                            next_trace += all_ops[p]['inputs']
            trace_in = next_trace

    # 2 - Remove inputs with zero dimensions
    remove_list: List[str] = []

    for n in all_ops:
        s: List[str] = all_ops[n]['inputs']
        all_ops[n]['original_inputs'] = s
        remove: bool = False
        new: List[str] = []
        for i in s:
            if i in shapes:
                new.append(i)
            else:
                remove = True
                remove_list.append(i)
        if remove:
            all_ops[n]['inputs'] = new

        s = all_ops[n]['outputs']
        remove = False
        new = []
        for i in s:
            if i in shapes:
                new.append(i)
            else:
                remove = True
                remove_list.append(i)
        if remove:
            all_ops[n]['outputs'] = new

    # 3 - Remove layers
    ignore_layers: List[str] = []

    for n in all_ops:
        if ignore_layer(n, all_ops[n]):
            ignore_layers.append(n)

    def follow_chain(
            layer: str,
            remove: str,
            replacements: List[str],
            which: str,
    ) -> None:
        for n in all_ops:
            if layer == n:
                continue
            for an_output in remove:
                new_inputs: List[str] = []
                for an_input in all_ops[n][which]:
                    if an_output == an_input:
                        new_inputs += replacements
                    else:
                        new_inputs.append(an_input)

                filtered: List[str] = []
                for ie in new_inputs:
                    if ie not in filtered:
                        filtered.append(ie)
                if filtered != all_ops[n][which]:
                    all_ops[n][which] = filtered

    # Fix up inputs after removing a layer
    for layer in ignore_layers:
        # Look for layer's outputs in the inputs of all other layers and replace
        follow_chain(
            layer,
            remove=all_ops[layer]['outputs'],
            replacements=all_ops[layer]['inputs'],
            which='inputs',
        )

    # 4 - Associate inputs and outputs
    input_layer: Dict[str, str] = {}
    output_layer: Dict[str, str] = {}
    final_layer: str = ''

    for name in all_ops:
        for ie in all_ops[name]['inputs']:
            if ie != '':
                input_layer[ie] = name

        for ie in all_ops[name]['outputs']:
            if ie != '':
                output_layer[ie] = name

        final_layer = name

    # 5 - Collect inputs and outputs
    inputs: Dict[str, List[str]] = {}
    outputs: Dict[str, List[str]] = {}

    for name in all_ops:
        if ignore_layer(name, all_ops[name]):
            continue

        # Get input names from first of the layer group
        if name not in inputs:
            inputs[name] = []
        for ie in all_ops[name]['inputs']:
            if ie != '' and (ie not in output_layer or output_layer[ie] != name):
                inputs[name].append(ie)

        if name not in outputs:
            outputs[name] = []
        for ie in all_ops[name]['outputs']:
            if ie != '' and (ie not in input_layer or input_layer[ie] != name):
                outputs[name].append(ie)

    # 6 - Collect ops
    prev_op_name: str = ''
    layers: Dict[str, Dict[str, Any]] = {}
    input_hwc: bool = False
    input_processors: int = 0

    for name in all_ops:
        if ignore_layer(name, all_ops[name]):
            continue

        this_layer: Dict[str, Any] = {}

        this_layer['name'] = name
        ins = inputs[name]

        # Mark output layers (the final layer is always an output layer)
        if name != final_layer and any(x not in input_layer for x in outputs[name]):
            this_layer['output'] = 'true'

        this_layer['op'] = main_op = 'Passthrough'
        operands: int = 1

        op = all_ops[name]['type']

        if op in ('Add', 'Sub', 'BitwiseOr', 'BitwiseXor'):
            operands = len(ins)
            if op.startswith('Bitwise'):
                op = op[7:]
            this_layer['eltwise'] = op
            this_layer['operands'] = operands
        elif op in ('Gemm', 'MatMul'):
            this_layer['op'] = main_op = 'Linear'
            this_layer['activate'] = 'None'
        elif op in ('Conv', 'ConvTranspose'):
            kernel_size = all_ops[name]['attrs']['kernel_shape']
            this_layer['op'] = f'{op}{len(kernel_size)}d'
            this_layer['activate'] = 'None'
            main_op = op
        elif op in ('MaxPool', 'AveragePool'):
            shape = all_ops[name]['attrs']['kernel_shape']
            if len(shape) == 1 or shape[0] == shape[1]:
                shape = shape[0]
            if all_ops[name]['type'] == 'MaxPool':
                this_layer['max_pool'] = shape
            else:
                this_layer['avg_pool'] = shape
            this_layer['pool_stride'] = all_ops[name]['attrs']['strides'][0]
            if 'dilations' in all_ops[name]['attrs']:
                this_layer['pool_dilation'] = all_ops[name]['attrs']['dilations']
        elif op in ('Abs', 'Relu'):
            this_layer['activate'] = op
        else:
            this_layer['op'] = f'Unknown ({op})'

        this_layer['main_op'] = main_op

        if op in ('Conv', 'ConvTranspose'):
            if len(kernel_size) == 1:
                this_layer['kernel_size'] = str(kernel_size[0])
            else:
                this_layer['kernel_size'] = f'{kernel_size[0]}x{kernel_size[1]}'
            pad = all_ops[name]['attrs']['pads']
            this_layer['pad'] = pad[0]
            groups = all_ops[name]['attrs']['group']
            if groups != 1:
                this_layer['groups'] = groups

            # Quantization uses hard-coded name from ai8x.py
            quantization = 8
            try:
                quantization = int(model.get_parameter(name + '.weight_bits'))
            except AttributeError:
                try:
                    quantization = int(model.get_parameter(canonical_name(name) + '.weight_bits'))
                except AttributeError:
                    pass
            if quantization not in (0, 8):
                assert quantization in (1, 2, 4), f'ERROR: {name}: quantization={quantization}'
                this_layer['quantization'] = quantization

        # Check for biases and 32-bit output using the data recorded earlier
        if op in convolution_ops:
            if 'wide' in all_ops[name] and all_ops[name]['wide']:
                this_layer['output_width'] = 32
            this_layer['have_bias'] = 'biases' in all_ops[name]
            this_layer['weight_count'] = int(all_ops[name]['attrs']['weights_vol'])

        # Check whether inputs need to be flattened (when they are not in 1x1 dimensions)
        flatten: bool = False
        if op in ('Gemm', 'MatMul'):
            for ie in inputs[name]:
                if ie in shapes:
                    mult: int = 1
                    for x in shapes[ie]:
                        mult *= x
                    if shapes[ie][0] != mult:
                        flatten = True
            if flatten:
                this_layer['flatten'] = 'true'

        in_dim: Optional[Union[Tuple[int, ...], List[int]]] = None
        prev_dim: Optional[Union[Tuple[int, ...], List[int]]] = None

        # Check whether dimensions need to change
        if not flatten:
            mult = 1
            for n in inputs[name]:
                if n in shapes:
                    for x in shapes[n]:
                        mult *= x
                    if len(shapes[n]) > 1 and mult != shapes[n][0]:
                        in_dim = shapes[n][1:]
                    else:
                        in_dim = shapes[n]
                    break
            mult = 1
            for n in all_ops[name]['original_inputs']:
                if n in shapes:
                    for x in shapes[n]:
                        mult *= x
                    if len(shapes[n]) > 1 and mult != shapes[n][0]:
                        prev_dim = shapes[n][1:]
                    else:
                        prev_dim = shapes[n]
                    break
            if in_dim is not None and prev_dim is not None:
                if in_dim != prev_dim \
                   and (in_dim[0] != prev_dim[0] or len(prev_dim) > 1 or mult != prev_dim[0]):
                    if len(prev_dim) > 0:
                        prev_dim = list(prev_dim)
                    this_layer['in_dim'] = prev_dim

        # Find number of processors needed
        processors: int = 0
        for ie in inputs[name]:
            if ie in shapes:
                processors += shapes[ie][0]
        if operands > 1:  # Element-wise: data is always interleaved
            processors //= operands
        this_layer['proc_count'] = max(1, processors)

        # Inner layers and more than 16 channels are always HWC
        hwc = hwc or (processors > 16) or (prev_op_name != '')

        if prev_op_name == '':
            # Show input dimensions and data format for input layers
            this_layer['data_format'] = hwc
            input_hwc = hwc
            input_processors = this_layer['proc_count']
        else:
            # Don't set in_sequences when using strictly sequential single inputs
            ins = [output_layer[x] for x in ins]
            if len(ins) > 1 or (len(ins) > 0 and ins[0] != prev_op_name):
                if operands > 1:
                    ins.reverse()  # Reverse the list since PyTorch does it backwards
                this_layer['in_sequences'] = ins

        prev_op_name = name
        layers[name] = this_layer

    del input_layer
    del output_layer

    prev_name: str = ''
    pop_list: List[Tuple[str, str]] = []
    veto: bool = False
    prev: Dict[str, Any] = {}

    # 7a - Merge (fuse) conv and activation layers
    for count, (name, ll) in enumerate(layers.items()):
        if ll['main_op'] == 'Passthrough' and 'activate' in ll and prev_name != '':
            prev = layers[prev_name]
            if prev['main_op'] not in ('Conv', 'ConvTranspose', 'Linear'):
                print(f'ERROR: Activation layer {name} does not follow '
                      f'Conv/Linear layer {prev_name}!')
                prev_name = name
                continue
            # Check that no layer other than the activation layer uses the intermediate output of
            # the conv layer as an input
            veto = False
            for (other_name, ol) in layers.items():
                if other_name != name and other_name != prev_name and 'in_sequences' in ol:
                    if prev_name in ol['in_sequences']:
                        veto = True
                        break
            if veto:
                print(f'ERROR: Activation layer {name} has inputs that are not from a directly '
                      f'preceding Conv/Linear layer {prev_name}!')
                prev_name = name
                continue
            # Combine both layers
            if 'comment' not in prev:
                prev['comment'] = f'{prev_name} fused with {name}'
            else:
                prev['comment'] += f' and {name}'
            pop_list.append((prev_name, name))  # Mark second layer for deletion

            # Copy over convolution operation and keep the element-wise operation in place
            outputs[prev_name] = outputs[name]

            prev['activate'] = ll['activate']
            if 'output' in ll:
                prev['output'] = ll['output']
            if 'quantization' in ll:
                prev['quantization'] = ll['quantization']
            if 'output_width' in ll:
                prev['output_width'] = ll['output_width']

        prev_name = name

    # Delete the conv layers that were fused into the eltwise layer
    for (prev_name, name) in pop_list:
        # Change any dangling input sequences to the fused layer
        for (other_name, ol) in layers.items():
            if other_name != name and other_name != prev_name and 'in_sequences' in ol:
                for i, e in enumerate(ol['in_sequences']):
                    if e == name:
                        ol['in_sequences'][i] = prev_name

        # Delete the conv portion
        layers.pop(name)

    # 7b - Merge (fuse) pooling and conv layers
    prev_name = ''
    pop_list = []
    veto = False
    prev = {}

    for count, (name, ll) in enumerate(layers.items()):
        if ll['main_op'] in ('Conv', 'ConvTranspose', 'Linear') and prev_name != '':
            prev = layers[prev_name]
            if prev['main_op'] != 'Passthrough' or (
                'max_pool' not in prev and 'avg_pool' not in prev
            ):
                prev_name = name
                continue
            # Check that no layer other than the activation layer uses the intermediate output of
            # the conv layer as an input
            veto = False
            for (other_name, ol) in layers.items():
                if other_name != name and other_name != prev_name and 'in_sequences' in ol:
                    if prev_name in ol['in_sequences']:
                        veto = True
                        break
            if veto:
                prev_name = name
                continue
            # Combine both layers
            if 'comment' not in ll:
                ll['comment'] = f'{prev_name} fused with {name}'
            else:
                ll['comment'] = f'{prev_name} and ' + ll['comment']
            pop_list.append((prev_name, name))  # Mark first layer for deletion

            # Copy the pooling information into the convolution layer
            inputs[name] = inputs[prev_name]

            if 'in_sequences' in prev:
                ll['in_sequences'] = prev['in_sequences']
            if 'avg_pool' in prev:
                ll['avg_pool'] = prev['avg_pool']
            if 'max_pool' in prev:
                ll['max_pool'] = prev['max_pool']
            if 'pool_stride' in prev:
                ll['pool_stride'] = prev['pool_stride']
            if 'pool_dilation' in prev:
                ll['pool_dilation'] = prev['pool_dilation']
            if 'data_format' in prev:
                ll['data_format'] = prev['data_format']

        prev_name = name

    # Delete the pooling layers that were fused into the conv layer
    for (prev_name, _) in pop_list:
        # Delete the pooling portion
        layers.pop(prev_name)

    # 7c - Merge (fuse) element-wise and convolution layers
    prev_name = ''
    pop_list = []
    for count, (name, ll) in enumerate(layers.items()):
        if ll['main_op'] in ('Conv', 'ConvTranspose') and prev_name != '':
            prev = layers[prev_name]
            # Only one pooling operation possible
            pool_count: int = 0
            if 'max_pool' in prev:
                pool_count += 1
            if 'avg_pool' in prev:
                pool_count += 1
            if 'max_pool' in ll:
                pool_count += 1
            if 'avg_pool' in ll:
                pool_count += 1
            # Check that no layer other than the conv layer uses the intermediate output of the
            # element-wise layer as an input
            veto = False
            for (other_name, ol) in layers.items():
                if other_name != name and other_name != prev_name and 'in_sequences' in ol:
                    if prev_name in ol['in_sequences']:
                        veto = True
                        break
            if veto or 'in_sequences' in ll or 'in_dim' in ll or 'flatten' in ll:
                prev_name = name
                continue
            if prev['main_op'] != 'Passthrough' or 'operands' not in prev \
               or prev['operands'] == 1 or pool_count > 1:
                prev_name = name
                continue
            # MAX78002 - avoid element-wise with bias and convolution when using multi-pass
            if ai8x.dev.device == 87 and ll['have_bias'] and prev['proc_count'] > MAX_PROC:
                prev_name = name
                continue
            # Combine both layers
            if 'comment' not in ll:
                ll['comment'] = f'{prev_name} fused with {name}'
            else:
                ll['comment'] = f'{prev_name} and ' + ll['comment']
            pop_list.append((prev_name, name))  # Mark first layer for deletion

            # Copy over convolution operation and keep the element-wise operation in place
            inputs[name] = inputs[prev_name]

            if 'in_sequences' in prev:
                ll['in_sequences'] = prev['in_sequences']
            if 'max_pool' in prev:
                ll['max_pool'] = prev['max_pool']
                ll['pool_first'] = 'true'
            if 'avg_pool' in prev:
                ll['avg_pool'] = prev['avg_pool']
                ll['pool_first'] = 'true'
            if 'max_pool' in ll or 'avg_pool' in ll:
                ll['pool_first'] = 'false'
            if 'pool_stride' in prev:
                ll['pool_stride'] = prev['pool_stride']
            if 'pool_dilation' in prev:
                ll['pool_dilation'] = prev['pool_dilation']
            if 'data_format' in prev:
                ll['data_format'] = prev['data_format']
            ll['eltwise'] = prev['eltwise']
            ll['operands'] = prev['operands']

        prev_name = name

    # Delete the element-wise layers that were fused into the convolution layer
    for (prev_name, _) in pop_list:
        # Delete the element-wise portion
        layers.pop(prev_name)

    # 8 - Insert passthrough layers for write_gap
    write_gap_list: List[Tuple[str, int]] = []
    insert_list: List[Tuple[str, int]] = []
    source_list: List[Tuple[str, str]] = []

    prev_name = ''
    for (name, ll) in layers.items():
        if 'in_sequences' not in ll:
            if prev_name == '':
                prev_name = name
                continue
            sources: List[str] = [prev_name]
        else:
            sources = ll['in_sequences']
        operands = ll['operands'] if 'operands' in ll else len(sources)
        if operands < 2:
            prev_name = name
            continue

        # For each input, check whether anybody else is using the input. If yes, insert a dummy
        # layer that creates a write_gap version of the data. If no, add the write_gap to the
        # producer.
        for source in sources:
            must_insert: bool = False
            prev_name_inner = ''
            for (other_name, ol) in layers.items():
                if other_name not in (name, source):
                    if 'in_sequences' in ol:
                        for e in ol['in_sequences']:
                            if e == source:
                                must_insert = True
                    elif prev_name_inner == source:
                        must_insert = True
                        # Break the sequence
                        ol['in_sequences'] = [source]
                prev_name_inner = other_name
            if not must_insert:
                # The source is used only by the element-wise layer, so we can insert the write gap
                # directly
                write_gap_list.append((source, operands))
            else:
                insert_list.append((source, operands))
                # Replace source with source_gap in layers[name]['in_sequences']
                source_list.append((name, source))
                new_name = 'gap_' + source
                # ...and insert shaope information (input and output are both the same as the
                # original layer's output)
                inputs[new_name] = [source + '_data']
                outputs[new_name] = [name + '_data']
                for ie in outputs[source]:
                    if ie in shapes:
                        shapes[name + '_data'] = shapes[ie]
                        shapes[source + '_data'] = shapes[ie]

        prev_name = name

    # Insert simple write gaps
    for (name, operands) in write_gap_list:
        layers[name]['write_gap'] = operands - 1
    # Break sequence
    for (name, source) in source_list:
        seq = layers[name]['in_sequences']
        for i, s in enumerate(seq):
            if s == source:
                seq[i] = 'gap_' + source
    # Insert additional layers
    for (name, operands) in insert_list:
        new_layer: Dict[str, Any] = {}
        new_name = 'gap_' + name
        new_layer['name'] = new_name
        processors = 0
        for ie in inputs[new_name]:
            if ie in shapes:
                processors += shapes[ie][0]
        new_layer['proc_count'] = max(1, processors)
        new_layer['op'] = new_layer['main_op'] = 'Passthrough'
        new_layer['write_gap'] = operands - 1

        # Insert into dict via list
        insert_pos: int = list(layers.keys()).index(name) + 1
        layers_list: List[Any] = list(layers.items())
        layers_list.insert(insert_pos, (new_name, new_layer))
        layers = dict(layers_list)

    # 9 - Record actual used processors for all layers and weight cost for all conv layers
    # Add all_inputs (either in_sequence or the previous layer's name if not defined)
    all_inputs: Dict[str, List[str]] = {}  # Input LAYERS to the key layer (vs. shapes in inputs)
    all_outputs: Dict[str, List[str]] = {}  # Output LAUYERS to th key layer (vs. shapes)

    prev_name = ''
    for (name, ll) in layers.items():
        val: Optional[List[str]] = None
        if 'in_sequences' not in ll:
            val = [prev_name] if prev_name != '' else None
        else:
            val = ll['in_sequences']
        if val is not None:
            all_inputs[name] = val
            for ie in val:
                if ie in all_outputs:
                    all_outputs[ie].append(name)
                else:
                    all_outputs[ie] = [name]
        prev_name = name

        # There are two cases: element-wise operations ('operands' defined and > 1), and conv
        # operations with multi-pass (> 64 input channels). In either case, in_sequences has
        # more than one member.
        if 'in_sequences' not in ll or len(ll['in_sequences']) < 2:
            continue
        operands = ll['operands'] if 'operands' in ll else 1
        if operands == 1:
            operands = len(ll['in_sequences'])
            # Concat instead of interleave for small channel counts
            if ll['proc_count'] <= MAX_PROC:
                ll['concat'] = True
                for i, ie in enumerate(ll['in_sequences']):
                    lin = layers[ie]
                    lin['concat_source'] = max(i, lin['concat_source']) \
                        if 'concat_source' in lin else i
            else:
                # Interleave the inputs in the channels
                ll['proc_count'] //= operands

    # Mark the size of all concat outputs
    for (name, ll) in layers.items():
        if 'concat_source' not in ll:
            continue

        if name not in outputs or len(outputs[name]) != 1 or outputs[name][0] not in shapes:
            print('ERROR: Cannot derive output shape(s) for layer', name)
        oshape = shapes[outputs[name][0]]
        ll['concat_shape'] = oshape[0]

    def calculate_processors(processors: int) -> Tuple[int, int]:
        """
        Given a channel count, return the multi-pass adjusted processor count
        """
        multipass: int = 1
        if processors > MAX_PROC:  # Multi-pass processor count
            multipass = (processors + MAX_PROC - 1) // MAX_PROC
            processors = (processors + multipass - 1) // multipass  # Rounded up
            remainder: int = processors % 4
            if remainder != 0:
                remainder = 4 - remainder
            processors += remainder  # To next multiple of 4
        return processors, multipass

    # cost_list: processors/weights per layer for Conv layers with 60 or fewer processors
    cost_list: List[Tuple[int, int, str]] = []

    for (name, ll) in layers.items():
        hwc = ll['data_format'] if 'data_format' in ll else True
        processors = ll['proc_count']
        multipass: int = 1
        # Calculate the final number of used processors
        if hwc:
            processors, multipass = calculate_processors(processors)
            assert processors <= MAX_PROC
        else:
            assert processors <= MAX_PROC // 4
        ll['proc_used'] = processors
        ll['multipass'] = multipass

        weights_per_processor: int = 0
        if ll['main_op'] in ('Conv', 'ConvTranspose', 'Linear'):
            # Account for kernel size for convolution operations
            weights_per_processor = ll['weight_count'] // processors
            if 'quantization' in ll:
                weights_per_processor //= 8 // ll['quantization']
        ll['weight_cost'] = weights_per_processor

        if processors <= 60:
            # Only add to this list if we're using less than the full processor count
            # (for 64 processors, there are no options and hence no optimizations)
            cost_list.append((weights_per_processor, processors, name))
        else:
            # Append even those layers that use 64 processors, because they may be 'concat' layer
            # or it may be a layer 0 and therefore, the source may need to be arranged properly.
            # But use zero cost to get them to move to the end (or ignored in the cost
            # calculation).
            cost_list.append((0, processors, name))

    # print('\nFINAL cost_list', cost_list, '\n\n')

    # Layer 0 is not the output of anything, so add it in from the beginning
    bucket_groups: List[List[Tuple[int, int, str]]] = [[cost_list[0]]]
    bucket: Dict[str, int] = {}
    bucket[cost_list[0][2]] = 0

    # Find layers where the output is used as input of more than one layer. Those layers then
    # must use the same processors. This is achieved by "combining the rectangles".
    for (name, ll) in layers.items():
        if name not in all_outputs:
            continue  # This shouldn't ever happen except for the final layer

        # print(name, 'grouping everything in', all_outputs[name])
        # FIXME What is the output shape of this layer? May need the processor_count
        merge_buckets: List[int] = []
        for i, (_, _, n) in enumerate(cost_list):
            if n in all_outputs[name]:
                if n in bucket:  # We already have a bucket that contains n.
                    # Merge the bucket contents
                    merge_buckets.append(bucket[n])
                else:
                    # Create a new empty bucket and add the layer to it. Also add the new bucket
                    # to the merge list.
                    bucket_number = len(bucket_groups)
                    bucket[n] = bucket_number
                    merge_buckets.append(bucket_number)
                    bucket_groups.append([cost_list[i]])
        if len(merge_buckets) > 1:
            target: int = merge_buckets[0]
            for i in merge_buckets[1:]:
                bucket_groups[target] += bucket_groups[i]
                bucket_groups[i] = []  # Rather than deleting, we just set invalid values

    # For the remaining buckets, combine the weights, check the processors, and make a layer list
    # print('FINAL bucket_groups', bucket_groups)

    del cost_list  # No longer needed

    # Same as cost_list, but now with multiple entries
    group_list: List[Tuple[int, int, int, int, List[Tuple[int, int, str]]]] = []

    for b in bucket_groups:
        if not b:
            continue
        # print('============ bucket group', b)
        group_weights: int = 0
        group_procs: int = -1
        group_item: List[Tuple[int, int, str]] = []
        min_shift = 0
        last_processor = MAX_PROC - 1
        for (weights_per_processor, processors, name) in b:
            # print('examining - weights', weights_per_processor, 'procs', processors, 'name',
            #       name, 'concat', 'YES' if 'concat' in layers[name] else 'NO')
            group_weights += weights_per_processor  # Add weights
            # print('processors', processors, 'vs', group_procs)
            if 'concat' in layers[name]:  # For concatenation, we need partials
                assert group_procs < 0 or group_procs == processors \
                    or group_procs == processors // len(all_inputs[name])
                # FIXME: a straight divide may not be correct
                # FIXME: If there's a concat target, make sure to leave the extra space
                # print('Found a concat, should set min_shift')
                # min_shift = max(min_shift, processors // len(all_inputs[name]))
            else:
                # print(layers[name])
                assert group_procs < 0 or group_procs == processors
            group_procs = max(processors, group_procs)
            group_item.append((weights_per_processor, processors, name))
        if group_procs >= 0:
            group_list.append((group_weights, group_procs,
                               min_shift, last_processor, group_item))

    # print('group_list', group_list, '\n')

    # 10 - Assign processors
    # TODO: Real algorithm for output_offset
    weights_used: List[int] = [0] * MAX_PROC

    def allocate_processors(
            weight_cost: int,
            count: int,
            min_shift: int,
            data: List[Tuple[int, int, str]],
            hwc: bool = True,
    ) -> Tuple[List[int], int]:
        """
        Allocate a given count of processors and return the bit map.
        TODO: This function should be expanded to evenly distribute resources, not only based on
        weights but also data memory utilization.
        Additionally, the weight allocator could instead use a box packing algorithm, but the
        expected improvements are modest.
        """
        min_cost: Tuple[int, int] = (2**63-1, -1)
        shifted_map: List[int] = [0] * len(data)
        processor_map: List[int] = shifted_map

        # Move processor map left if needed to achieve the lowest combined 'weights_used'
        for shift in range(min_shift, MAX_PROC - count + 1, 4):
            for d, (item_cost, item_procs, _) in enumerate(data):
                if hwc:
                    # Pick "count" processors starting from 0
                    shifted_map[d] = ((1 << item_procs) - 1) << shift
                else:
                    # Pick the first for every quadrant (FIFO compatible) or one every 4
                    mult = 16 if count <= 4 else 4
                    shifted_map[d] = 0
                    for i in range(item_procs):
                        shifted_map[d] |= 1 << mult * i
                    shifted_map[d] <<= shift

            if weight_cost == 0:
                min_cost = (0, min_shift)
                break

            cost: List[int] = weights_used.copy()

            for p in range(MAX_PROC):
                for d, (item_cost, item_procs, _) in enumerate(data):
                    if shifted_map[d] & (1 << p):
                        cost[p] += item_cost

            max_cost = max(cost)  # High water mark

            if max_cost < min_cost[0]:  # Better option?
                min_cost = (max_cost, shift)
                processor_map = shifted_map.copy()

        if weight_cost > 0:
            # Accounting based on the result
            for p in range(MAX_PROC):
                proc_used: bool = False
                for d, _ in enumerate(data):
                    if processor_map[d] & (1 << p):
                        proc_used = True
                if proc_used:
                    weights_used[p] = min_cost[0]

        return processor_map, min_cost[1]

    # When using CHW inputs with more than 1 processor, or we use fifos, we leave layer 0 alone
    # and don't move processors. Technically, there would be a few more options for these layers
    # but they are not considered since the potential gain is minimal.
    l0: Optional[Tuple[int, int, int, int, List[Tuple[int, int, str]]]] = None
    if not move_l0 or not input_hwc or input_processors != 1 or use_fifos:
        l0 = group_list[0]
        del group_list[0]

    # Sort by height and then width for the row allocator
    group_list.sort(reverse=True)  # Sort by weight depth, then processor count
    if l0 is not None:
        group_list.insert(0, l0)  # Put layer 0 back at the start

    # print('SORTED FINAL group_list', group_list, '\n\n')

    # Allocate and record processors for the conv layers, and follow back to the producing layers
    # and record the 'output_processors'.
    for (weights_per_processor, processors, min_shift, _stop, data) in group_list:
        # print('\n', weights_per_processor, processors, min_shift, _stop, data)

        # Grab the first item. This should be the input layer for the very first item in the list.
        ll = layers[data[0][2]]
        assert processors > 0
        hwc = ll['data_format'] if 'data_format' in ll else True

        processor_map, _ = \
            allocate_processors(
                weights_per_processor,
                processors,
                min_shift,
                data,
                hwc=hwc,
            )

        # print(data)

        # When an output is used as an input to both a regular layer and a concatenation
        intersected_map: int = 0xffffffffffffffff
        for i, (_, _, name) in enumerate(data):
            # print(name, f'0x{processor_map[i]:016x}')
            layers[name]['processors'] = processor_map[i]
            intersected_map &= processor_map[i]
            # print(f'processor_map[{i}] 0x{processor_map[i]:016x}')

        # print(f'intersected_map 0x{intersected_map:016x}')
        # Number of 1-bits in the processor map, used for shifting concatenated inputs
        shift_count: int = 0
        while intersected_map & 1 == 0:
            shift_count += 1
            intersected_map >>= 1

        for i, (_, _, name) in enumerate(data):
            ll = layers[name]

            if name not in all_inputs:
                continue

            if 'concat' not in ll:
                # Interleaved (write_gap) outputs must share the same output processors.
                for seq in all_inputs[name]:
                    layers[seq]['output_processors'] = processor_map[i]
            else:
                # Concatenation: Create partial maps
                shift: int = shift_count
                for seq in all_inputs[name]:
                    # print('layer', seq, 'concat_shape', layers[seq]['concat_shape'],
                    #       'concat_source', layers[seq]['concat_source'], end='')
                    item_procs: int = layers[seq]['concat_shape']
                    layers[seq]['output_processors'] = ((1 << item_procs) - 1) << shift
                    # print(f' map 0x{layers[seq]["output_processors"]:016x}, item_procs',
                    #       item_procs)
                    if item_procs % 4 != 0:
                        item_procs += 4 - item_procs % 4
                    shift += item_procs

    # 11 - Set output_offset. TODO: Use real algorithm
    out_offset: int = 0  # Start at 0 (default input offset)

    for (name, ll) in layers.items():
        out_offset = allocate_offset(name, ll['processors'], out_offset)
        ll['out_offset'] = out_offset

    # 12 - Sanity check
    for (name, ll) in layers.items():
        if name not in all_inputs:
            continue
        if 'concat' in ll:
            processor_union: int = 0
            for ie in all_inputs[name]:
                processor_union |= layers[ie]['output_processors']
            if ll['processors'] != processor_union:
                print(f'ERROR: Layer {name} processors 0x{ll["processors"]:016x} does not match '
                      f'the contributing output_processors union 0x{processor_union:016x} of '
                      f'the inputs {all_inputs[name]}')
        else:
            for ie in all_inputs[name]:
                if ll['processors'] != layers[ie]['output_processors']:
                    print(f'ERROR: Layer {name} processors 0x{ll["processors"]:016x} does not '
                          f'match the output_processors 0x{layers[ie]["output_processors"]} of '
                          f'the input {ie}')

    # 13 - Clean up unnecessary output_processors
    prev_name = ''
    for (name, ll) in layers.items():
        if prev_name != '':
            prev = layers[prev_name]
            if 'output_processors' in prev and prev['output_processors'] == ll['processors']:
                del prev['output_processors']
        prev_name = name

    # 14 - Print
    target_dir: str = os.path.dirname(filename)
    if target_dir != '':
        os.makedirs(target_dir, exist_ok=True)
    with open(filename, mode='w', encoding='utf-8') as f:
        f.write(
            '---\n'
            '# YAML template -- requires manual editing, particularly with regard to out_offset '
            'and processors\n'
            f'# Generated for {devices.partnum(ai8x.dev.device)} with input format '
            f'{"HWC" if input_hwc else "CHW"}\n\n'
            f'arch: {arch}\n'
            f'dataset: {dataset}\n'
            '\n'
            'layers:'
        )

        prev_name = ''
        for count, (name, ll) in enumerate(layers.items()):
            f.write('\n'
                    f'  # Layer {count}\n'
                    f"  - name: {canonical_name(ll['name'])}")
            if 'comment' in ll:
                f.write(f"  # {ll['comment']}")
            f.write('\n')

            hwc = ll['data_format'] if 'data_format' in ll else True

            # Check input sequences and dimensions
            warn_dim = False
            print_dim = verbose or prev_name == ''
            if print_dim:
                f.write('    # input shape: ')
            i = 0
            max_pixels = MAX_PIXELS if hwc else 4 * MAX_PIXELS
            for ie in inputs[name]:
                if ie in shapes:
                    if i > 0 and print_dim:
                        f.write(', ')
                    if print_dim:
                        f.write(str(shapes[ie]))
                    pixels = 1
                    for x in range(1, len(shapes[ie])):
                        pixels *= shapes[ie][x]
                    if pixels > max_pixels:
                        warn_dim = True
                    i += 1
            if print_dim:
                f.write('\n')
            if warn_dim:
                f.write(f'    # dimensions ({pixels} pixels) may require streaming or folding\n')

            if 'data_format' in ll:
                f.write(f'    data_format: {"HWC" if hwc else "CHW"}\n')
            if 'in_sequences' in ll:
                f.write('    in_sequences: [')
                ins = ll['in_sequences']
                for i, ie in enumerate(ins):
                    if i > 0:
                        f.write(', ')
                    f.write(canonical_name(ie))
                f.write(']\n')
            if 'in_dim' in ll:
                f.write(f"    in_dim: {ll['in_dim']}\n")
            show_output = verbose
            if 'output' in ll:
                f.write(f"    output: {ll['output']}\n")
                show_output = True
            processors = ll['processors']
            if processors == 0:
                f.write('    processors: unknown\n')
            else:
                f.write(f'    processors: 0x{processors:016x}\n')
            f.write(f"    out_offset: 0x{ll['out_offset']:04x}\n")
            if 'quantization' in ll:
                f.write(f"    quantization: {ll['quantization']}\n")
            if 'output_width' in ll:
                f.write(f"    output_width: {ll['output_width']}\n")
            f.write(f"    op: {ll['op']}\n")
            if 'eltwise' in ll:
                f.write(f"    eltwise: {ll['eltwise']}\n")
            if 'operands' in ll:
                f.write(f"    operands: {ll['operands']}\n")
            if 'kernel_size' in ll:
                f.write(f"    kernel_size: {ll['kernel_size']}\n")
            if 'pad' in ll:
                f.write(f"    pad: {ll['pad']}\n")
            if 'groups' in ll:
                f.write(f"    groups: {ll['groups']}\n")
            if 'flatten' in ll:
                f.write(f"    flatten: {ll['flatten']}\n")
            if 'activate' in ll:
                f.write(f"    activate: {ll['activate']}\n")
            if 'pool_first' in ll:
                f.write(f"    pool_first: {ll['pool_first']}\n")
            if 'max_pool' in ll:
                f.write(f"    max_pool: {ll['max_pool']}\n")
            if 'avg_pool' in ll:
                f.write(f"    avg_pool: {ll['avg_pool']}\n")
            if 'pool_stride' in ll:
                f.write(f"    pool_stride: {ll['pool_stride']}\n")
            if 'pool_dilation' in ll:
                f.write(f"    pool_dilation: {ll['pool_dilation']}\n")
            if 'write_gap' in ll:
                f.write(f"    write_gap: {ll['write_gap']}\n")
            if 'output_processors' in ll:
                processors = ll['output_processors']
                if processors == 0:
                    f.write('    output_processors: unknown\n')
                else:
                    f.write(f'    output_processors: 0x{processors:016x}\n')

            # Show output dimensions for all output layers
            if name == final_layer or show_output:
                f.write('    # output shape: ')
                i = 0
                for ie in outputs[name]:
                    if ie in shapes:
                        if i > 0:
                            f.write(', ')
                        f.write(str(shapes[ie]))
                        i += 1
                f.write('\n')

            prev_name = name


if __name__ == "__main__":
    parser = argparse.ArgumentParser("YAML configuration generator")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the model")
    parser.add_argument("--arch", type=str, default="custom", help="Model architecture")
    parser.add_argument("--device-id", type=int, default=None, help="Device id")
    parser.add_argument(
        "--input-shape", type=str, required=True, help="Model input shape, i.e. 3,28,28"
    )
    parser.add_argument("--output-path", type=str, required=True, help="Output YAML path")

    args = parser.parse_args()

    input_shape = list(map(int, args.input_shape.split(',')))

    # see: https://github.com/analogdevicesinc/ai8x-training/blob/main/train.py#L691-L695
    # pylint: disable=protected-access
    distiller.utils._validate_input_shape = lambda _a, _b: input_shape

    ai8x.set_device(args.device_id, False, False)

    torch_model = torch.load(args.model_path, weights_only=False)

    create(
        model=torch_model,
        dataset="unknown",
        arch=args.arch,
        filename=args.output_path,
    )
