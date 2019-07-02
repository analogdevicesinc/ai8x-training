# AI8X Model Training and Quantization
# AI8X Network Loader and RTL Simulation Generator

_7/2/2019_

_Open this file in a markdown enabled viewer, for example Visual Studio Code
(https://code.visualstudio.com). See https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet
for a description of Markdown._

This software consists of two related projects:
1. AI84 Model Training and Quantization
2. AI84 Network Loader and RTL Simulation Generator

## Contents

- [Contents](#Contents)
- [Overview](#Overview)
- [Recommended Software](#Recommended-Software)
- [Installation](#Installation)
  - [File System Layout](#File-System-Layout)
  - [Upstream Code](#Upstream-Code)
  - [Prerequisites](#Prerequisites)
  - [Project](#Project)
    - [Creating the Virtual Environment](#Creating-the-Virtual-Environment)
    - [Building TensorFlow (for old CPUs)](#Building-TensorFlow-for-old-CPUs)
    - [Nervana Distiller](#Nervana-Distiller)
    - [Synthesis Project](#Synthesis-Project)
- [Usage: AI84 Model Training and Quantization](#Usage-AI84-Model-Training-and-Quantization)
  - [Quantization](#Quantization)
  - [Alternative Quantization Approaches](#Alternative-Quantization-Approaches)
- [AI84 Hardware and Resources](#AI84-Hardware-and-Resources)
  - [Overview](#Overview-1)
  - [Number Format](#Number-Format)
  - [Channel Data Formats](#Channel-Data-Formats)
    - [HWC](#HWC)
    - [CHW](#CHW)
  - [Accelerator Limits](#Accelerator-Limits)
- [Data, Weight, and Processors](#Data-Weight-and-Processors)
  - [Weight Memory](#Weight-Memory)
  - [Data Memory](#Data-Memory)
  - [CHW Data Format and Consequences for Weight Memory Layout](#CHW-Data-Format-and-Consequences-for-Weight-Memory-Layout)
- [Active Processors and Layers](#Active-Processors-and-Layers)
  - [Layers and Weight Memory](#Layers-and-Weight-Memory)
  - [Weight Storage Example](#Weight-Storage-Example)
  - [Limitations of AI84 Networks](#Limitations-of-AI84-Networks)
- [Adding Datasets and New Networks](#Adding-Datasets-and-New-Networks)
- [Usage: AI84 Network Loader](#Usage-AI84-Network-Loader)
  - [Network Loader Configuration Language](#Network-Loader-Configuration-Language)
    - [Global Configuration](#Global-Configuration)
      - [`arch` (Mandatory)](#arch-Mandatory)
      - [`dataset` (Mandatory)](#dataset-Mandatory)
      - [`output_map` (Optional)](#outputmap-Optional)
      - [`layers` (Mandatory)](#layers-Mandatory)
    - [Per-Layer Configuration](#Per-Layer-Configuration)
      - [`sequence` (Optional)](#sequence-Optional)
      - [`processors` (Mandatory)](#processors-Mandatory)
      - [`output_processors` (Optional)](#outputprocessors-Optional)
      - [`out_offset` (Optional)](#outoffset-Optional)
      - [`in_offset` (Optional)](#inoffset-Optional)
      - [`output_width` (Optional)](#outputwidth-Optional)
      - [`data_format` (Optional)](#dataformat-Optional)
      - [`convolution` (Optional)](#convolution-Optional)
      - [`activate` (Optional)](#activate-Optional)
      - [`quantization` (Optional)](#quantization-Optional)
      - [`kernel_size` (Optional)](#kernelsize-Optional)
      - [`stride` (Optional)](#stride-Optional)
      - [`pad` (Optional)](#pad-Optional)
      - [`max_pool` (Optional)](#maxpool-Optional)
      - [`avg_pool` (Optional)](#avgpool-Optional)
      - [`pool_stride` (Optional)](#poolstride-Optional)
  - [Adding Datasets](#Adding-Datasets)
- [AI84 SDK](#AI84-SDK)
- [AI85/AI86](#AI85AI86)
- [CMSIS5 NN Emulation](#CMSIS5-NN-Emulation)
- [Contributing Code](#Contributing-Code)
  - [Linting](#Linting)
  - [Submitting Changes](#Submitting-Changes)

## Overview

The following graphic shows an overview of the development flow:

![Development Flow](docs/DevelopmentFlow.png)

## Recommended Software

The following software is optional, and can be replaced with other similar software of the 
user's choosing.

1. Visual Studio Code (Editor, Free), https://code.visualstudio.com
2. CoolTerm (Serial Terminal, Free), http://freeware.the-meiers.org
3. Git Fork (Graphical Git Client, Free), https://git-fork.com
4. Beyond Compare (Diff and Merge Tool, $60), https://scootersoftware.com

## Installation

### File System Layout

Including the SDK from SVN, the expected file system layout will be:

    ..../ai8x-training/
    ..../ai8x-synthesis/
    ..../AI84SDK/

### Upstream Code

Change to the project root (denoted as `....` above).

    git clone https://first.last@gerrit.maxim-ic.com:8443/ai8x-training
    git clone https://first.last@gerrit.maxim-ic.com:8443/ai8x-synthesis

### Prerequisites

When going beyond simple tests, model training requires hardware acceleration (the network loader
does not require CUDA).
Install CUDA10 and CUDNN:
https://developer.nvidia.com/cuda-downloads
https://developer.nvidia.com/cudnn

### Project

*The software in this project requires Python 3.6.5 or a later 3.6.x version.*

It is not necessary to install Python 3.6.5 system-wide, or to rely on the system-provided
Python. To manage Python versions, use `pyenv` (https://github.com/pyenv/pyenv).

On macOS (no CUDA support available):

    $ brew install pyenv pyenv-virtualenv libomp

On Ubuntu 18.04 LTS:

    $ sudo apt-get install -y make build-essential libssl-dev zlib1g-dev libbz2-dev \
        libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
        xz-utils tk-dev libffi-dev liblzma-dev libbz2-dev libreadline-dev libssl-dev
    $ curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash

Then, add to ~/.bash_profile or ~/.profile (as shown by the previous step):

    eval "$(pyenv init -)"
    eval "$(pyenv virtualenv-init -)"

Next, close the Terminal and install Python 3.6.5:

    $ pyenv install 3.6.5

#### Creating the Virtual Environment

To create the virtual environment and install basic wheels:

    $ cd ai8x-training
    $ pyenv local 3.6.5
    $ python3 -m venv .
    $ source bin/activate
    (ai8x-training) $ pip3 install -U pip setuptools
    (ai8x-training) $ pip3 install numpy
    (ai8x-training) $ pip3 install pyyaml tabulate future six typing
    (ai8x-training) $ pip3 install scipy

For macOS and systems without CUDA:
    
    (ai8x-training) $ pip3 install torch

For CUDA 10 on Linux (see https://pytorch.org/get-started/locally/):

    (ai8x-training) $ pip3 install https://download.pytorch.org/whl/cu100/torch-1.1.0-cp36-cp36m-linux_x86_64.whl

On all systems, install TorchVision:

    (ai8x-training) $ pip3 install torchvision

Next, install TensorFlow. TensorFlow is needed for TensorBoard support.

    (ai8x-training) $ pip3 install tensorflow

*NOTE*: On x86 systems, the pre-built TensorFlow wheels require AVX (`sysctl -n machdep.cpu.features`
to find out). Running a TensorFlow wheel that requires AVX instructions on unsupported CPUs results
in `Illegal instruction: 4` on startup.

#### Building TensorFlow (for old CPUs)

If the CPU does not support AVX, or to enable support for AVX2, or CUDA, or AMD64, build
TensorFlow locally. Otherwise, skip ahead to [TorchVision and Distiller](#TorchVision-and-Distiller).
_To repeat: Building TensorFlow is not needed when the binary wheels are functioning properly._

The TensorFlow build requires Java 8 and Bazel, and takes over two hours. 
Building TensorFlow requires Bazel 0.21.0 (newer or much older versions do not work). See
https://docs.bazel.build/versions/master/install-compile-source.html#build-bazel-using-bazel for
build instructions.

Once Bazel is installed, compile and install TensorFlow. On Jetson TX1, disable S3. See 
https://github.com/peterlee0127/tensorflow-nvJetson/blob/master/patch/tensorflow1.12.patch
The removal of `-mfpu=neon` does not seem to be necessary.

    (ai8x-training) $ pip3 install keras_preprocessing
    (ai8x-training) $ git clone https://github.com/tensorflow/tensorflow 
    (ai8x-training) $ cd tensorflow
    (ai8x-training) $ git checkout tags/v1.13.1
    (ai8x-training) $ ./configure
    (ai8x-training) $ bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package
    (ai8x-training) $ bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
    (ai8x-training) $ pip3 install /tmp/tensorflow_pkg/tensorflow-1.13.1-cp36-cp36m-macosx_10_13_x86_64.whl 

#### Nervana Distiller

To install Nervana's distiller:

    (ai8x-training) $ git clone https://github.com/NervanaSystems/distiller.git
    (ai8x-training) $ cp requirements.txt.distiller distiller/requirements.txt
    (ai8x-training) $ pip3 install -e distiller

On macOS, comment out the following line in `distiller/distiller/apputils/execution_env.py`:

    # logger.debug("Number of CPUs: %d", len(os.sched_getaffinity(0)))

(macOS) Add to `~/.matplotlib/matplotrc`:

    backend: TkAgg

#### Synthesis Project

For `ai8x-synthesis`, some of the installation steps can be simplified. Specifically, TorchVision,
TensorFlow, and SciPy are not needed, CUDA is unnecessary, and Distiller can be installed from the
`ai8x-training` project.

Start by creating a second virtual environment:

    $ cd ai8x-synthesis
    $ pyenv local 3.6.5
    $ python3 -m venv .
    $ source bin/activate
    (ai8x-synthesis) $ pip3 install -U pip setuptools
    (ai8x-synthesis) $ pip3 install numpy
    (ai8x-synthesis) $ pip3 install pyyaml tabulate future six typing
    (ai8x-synthesis) $ pip3 install torch
    (ai8x-synethsis) $ pip3 install -e ../ai8x-training/distiller

----

## Usage: AI84 Model Training and Quantization

The `ai84net.py` file contains models that fit into AI84's weight memory. These models rely on
the AI84 hardware operators that are defined in `ai84.py`.

To train the FP32 model for FashionMIST, run `go_fashionmnist.sh` in the `ai8x-training` project.
This script will place checkpoint files into the log directory. Training makes use of the Distiller
framework, but the `train.py` software has been modified slightly to improve it and add some AI84
specifics.

### Quantization

There are two main approaches to quantization -- quantization-aware training and post-training
quantization.

Since performance for 8-bit weights is decent enough, _naive post-training quantization_ is used
in the `ai84ize.py` software. While several approaches are implemented, a simple fixed scale
factor is used based on experimental results. The approach requires the clamping operators
implemented in `ai84.py`.

The software quantizes an existing PyTorch checkpoint file and writes
out a new PyTorch checkpoint file that can then be used to evaluate the quality of the
quantized network, using the same PyTorch framework used for training.
The same new checkpoint file will also be used to feed the [AI84 Network Loader](#Usage-AI84-Network-Loader).

Copy the weight files into the `trained/` folder of the `ai8x-synthesis` project.

Example:

    (ai8x-synthesis) $ ./ai84ize.py logs/path-to-checkpoint/checkpoint.pth.tar ../ai8x-synthesis/trained/ai84-fashionmnist.pth.tar -v

To evaluate the quantized network:

    (ai8x-synthesis) $ ./evaluate_fashionmnist.sh

### Alternative Quantization Approaches

Post-training quantization can be improved using more sophisticated methods. For example, see
https://github.com/pytorch/glow/blob/master/docs/Quantization.md,
https://github.com/ARM-software/ML-examples/tree/master/cmsisnn-cifar10,
https://github.com/ARM-software/ML-KWS-for-MCU/blob/master/Deployment/Quant_guide.md,
or Distiller's approach (installed with this software).

Further, a quantized network can be refined using post-quantization training (see Distiller).

The software also includes an `AI84RangeLinear.py` training quantizer that plugs into the Distiller
framework for quantization-aware training. However, it needs work as its performance is not
good enough yet and the distiller source needs to be patched to enable it (add 
`from range_linear_ai84 import QuantAwareTrainRangeLinearQuantizerAI84` to `distiller/config.py`
and remove `False and` from `if False and args.ai84` in `train.py`).

Note that AI84 does not have a configurable per-layer output shift. The addition of this
shift value will allow easier quantization on AI85, since fractional bits can be used if weights
do not span the full 8-bit range (most read-made quantization approaches require a weight scale
or output shift).

In all cases, ensure that the quantizer writes out a checkpoint file that the AI84 Network Loader
can read.


## AI84 Hardware and Resources

AI84 is an embedded accelerator. Unlike GPUs, it does not have gigabytes of weight memory, and cannot
support arbitrary data (image) sizes.

### Overview

TBD

### Number Format

All weights and data are stored and computed in Q7 format (signed two's complement 8-bit integers,
[-128...+127]).

### Channel Data Formats

#### HWC

All internal data are stored in HWC format, 4 channels per 32-bit word. Assuming 3-color (or
3-channel) input, one byte will be unused. Example:

![0BGR 0BGR 0 BGR 0BGR...](docs/HWC.png)

#### CHW

The input layer can also use the CHW format (sequence of channels), for example:
  
![RRRRRR...GGGGGG...BBBBBB...](docs/CHW.png)

### Accelerator Limits

* The weight memory supports up to 128 * 64 3x3 Q7 kernels. However, weights must be arranged
  according to specific rules detailed below.
* There are 16 instances of 16 KB data memory. Any data channel (input, intermediate, or output)
  must completely fit into one memory instance. This limits the first-layer input to 128x128 pixels
  per channel in the CHW format. However, when using more than one input channel, the HWC format may
  be preferred, and all layer output are in HWC format as well. In those cases, it is required that
  four channels fit into a single memory instance -- or 64x64 pixels per channel. Note that the
  first layer commonly creates a wide expansion (i.e., large number of output channels) that needs
  to fit into data memory, so the input size limit is mostly theoretical.
* The maximum number of layers is 32.
* The maximum number of input or output channels in any layer is 64 each.

## Data, Weight, and Processors

Data memory, weight memory, and processors are interrelated.

In the AI84 accelerator, processors are organized as follows:

* Each processor is connected to its own dedicated weight memory instance.
* Four processors share one data memory instance.
* A group of sixteen processors shares certain common controls and can be operated as a slave
  to another group, or independently/separately.

Any given processor has visibility of:

* Its dedicated weight memory, and
* The data memory instance it shares with three other processors.

### Weight Memory

For each of the four 16-processor groups, weight memory and processors can be visualized as
follows. Assuming one input channel processed by processor 0, and 8 output channels, the shaded kernels
will be used:

![Weight Memory Map](docs/KernelMemory.png)

### Data Memory

Data memory connections can be visualized as follows:

![Data Memory Map](docs/DataMemory.png)

All input data must be located in the data memory instance the processor can access. Conversely,
output data can be written to any data memory instance inside the accelerator (but not to general
purpose SRAM on the Arm microcontroller bus).

The data memory instances inside the accelerator are single-port memories. This means that only
one read operation can happen per clock cycle. When using the HWC data format, this means that
each of the four processors sharing the data memory instance will receive one byte of data per
clock cycle (since each 32-bit data word consists of four packed channels).

### CHW Data Format and Consequences for Weight Memory Layout

On the other hand, when using the CHW data format, only one of the four processors sharing the data
memory instance can be used. The next channel needs to use a processor connected to a different
data memory instance, so that the machine can deliver one byte per clock cycle to each enabled
processor.

Because of the fact that a processor has its own dedicated weight memory, this will introduce
"gaps" in the weight memory map, as shown in the following illustration:

![Kernel Memory Gaps](docs/KernelMemoryGaps.png)


## Active Processors and Layers

For each layer, a set of active processors must be specified. The number of active processors
must be the same as the number of input channels for the layer, and the input data for that
layer must be located in data memory instances accessible to the selected processors.

It is possible to specify a relative offset into the data memory instance that applies to all
processors. _Example:_ Assuming HWC data format, specifying the offset as 8192 bytes will cause
processors 0-3 to read their input from the second half of data memory 0, processors 4-7 will
read from the second half of data memory instance 1, etc.

For most simple networks with limited data sizes, it is easiest to ping-pong between the first
and second halves of the data memories - specify the data offset as 0 for the first layer, 0x2000
for the second layer, 0 for the third layer, etc. This strategy avoids overlapping inputs and
outputs when a given processor is used in two consecutive layers.

Even though this is supported by the accelerator, the AI84 Network Generator will not
be able to check for inadvertent overwriting of unprocessed input data by newly generated output
data. Use the `--overlap-data` command line switch to disable these checks, and to allow overlapped
data.

### Layers and Weight Memory

For each layer, the weight memory start column must be specified. The start column must be a
multiple of 4, and the value applies to all processors.

The following example shows the weight memory layout for two layers. The first layer (L0) has 7
inputs and 9 outputs, and the second layer (L1) has 9 inputs and 2 outputs.

![Layers and Weight Memory](docs/KernelMemoryLayers.png)


### Weight Storage Example

The file `ai84net.xlsx` contains an example for a single-channel CHW input using the `AI85Net5`
network (this example also supports or up to four channels in HWC).

*NOTE*: As described above, multiple CHW channels must be loaded into separate
memory instances. When using a large number of channels, this can cause 'holes' in the processor
map, which in turn can cause subsequent layers' kernels to require padding.

The AI84 Network Loader prints a kernel map that shows the kernel arrangement based on the provided
network description. It will also flag cases where kernel or bias memories are exceeded.

### Limitations of AI84 Networks

The AI84 hardware does not support arbitrary network parameters. Specifically,
* Dilation, element-wise addition, and 1D convolutions are not supported.
* `Conv2D` kernel sizes must be 3x3.
* `Conv2D` padding can be 0, 1, or 2.
* The `Conv2D` stride is fixed to 1 when pooling is used.
* `Conv1D` kernel sizes must be 9.
* `Conv1D` padding can be 0, 3, or 6.
* The `Conv1D` stride is fixed to 3 in all cases.
* The only supported activation function is `ReLU`.
* Pooling is only supported for `Conv2D`.
* Pooling is always combined with a convolution. Both max pooling and average pooling are
  available.
* Pooling does not support padding.
* Pooling strides can be 1, 2, or 4.
* On AI84, three pooling sizes are supported:
  - 2x2 with stride 1
  - 2x2 with stride 2
  - and, in the last layer, using a custom unload function, 4x4 stride 4.
* Pooling must cleanly divide the input data width and height. For example, a 2x2 pool on
  17x17 data is not supported.
* Average pooling does not support more than 2048 bits in the accumulator. This translates to
  a 4x4 pooling window if activation was used on the prior layer, 3x3 otherwise. Additionally,
  average pooling is currently implemented as a `floor()` operation. Since there is also a
  quantization step at the output of the average pooling, it may not perform as intended
  (for example, a 2x2 `AvgPool2D` of `[[0, 0], [0, 3]]` will return `0`).
* Pooling window sizes must be even numbers, and have equal H and W dimensions.
* The number of input or output channels must not exceed 64.
* The number of layers must not exceed 32.
* The maximum dimension for input or output data is 256.
* Input data must be square (i.e., rows == columns).
* Overall weight storage is limited to 64*128 3x3 kernels. However, weights must be arranged
  in a certain order, see above.
* The hardware supports only 2D convolution layers. For convenience, a single final fully
  connected layer with 8-bit inputs/weights/bias, and 16-bit output is supported in software,
  as well as a software `SoftMax` operator.
* Since the internal network format is HWC in groups of four channels, output concatenation only
  works properly when all components of the concatenation other than the last have multiples
  of four channels. Output concatenation is not yet supported in the `ai84.py` primitives.
* It is recommended to not use bias on the AI84 convolutions when using post-training
  quantization as described in this document. The reason is that the bias is not shifted by the
  same amount as the weights. This could be mitigated using a more involved training procedure,
  but since bias values do not significantly improve performance in the example networks, and
  since this will be corrected for AI85, the recommendation is to not use bias values for now.

With the exception of weight storage, and bias use, most of these limitations are flagged when
using the network primitives from `ai84.py`.


## Adding Datasets and New Networks

The following steps are needed to add new data formats and networks:
1. Develop a data loader in PyTorch.
2. Implement a new network model (see `ai84net.py` for an example).
3. Add data loader and network model to `train.py`.
4. Train the new network with the new data. See `go_cifar.sh` for a command line example.


## Usage: AI84 Network Loader

_The network loader currently depends on PyTorch and Nervana's distiller. This requirement will be
removed in the long term._

The network loader creates C code that programs the AI84 (for embedded execution, or RTL
simulation). Additionally, the generated code contains a sample image and the expected output
for the sample image as well as code that verifies the expected output.

The `cnn-gen.py` program needs two inputs:
1. A quantized checkpoint file, generated by the AI84 model quantization program `ai84ize.py`.
2. A YAML description of the network.

An example network description for the ai84net5 architecture and FashionMNIST is shown below:

    # CHW (big data) configuration for FashionMNIST

    arch: ai84net5
    dataset: FashionMNIST

    # Define layer parameters in order of the layer sequence
    layers:
    - pad: 1
      activate: ReLU
      out_offset: 0x2000
      processors: 0x0000000000000001
      data_format: CHW
    - max_pool: 2
      pool_stride: 2
      pad: 2
      activate: ReLU
      out_offset: 0
      processors: 0xfffffffffffffff0
    - max_pool: 2
      pool_stride: 2
      pad: 1
      activate: ReLU
      out_offset: 0x2000
      processors: 0xfffffffffffffff0
    - avg_pool: 2
      pool_stride: 2
      pad: 1
      activate: ReLU
      out_offset: 0
      processors: 0x0ffffffffffffff0

To generate an embedded AI84 demo in the `demos/FashionMNIST/` folder, use the following command
line:

    ./cnn-gen.py --verbose -L --top-level cnn --test-dir demos --prefix FashionMNIST --checkpoint-file trained/ai84-mnist.pth.tar --config-file networks/fashionmnist-chw.yaml --fc-layer --embedded-code

Running this command will combine the network described above with a fully connected software
classification layer. The generated code will include all loading, unloading, and configuration
steps.

To generate an RTL simulation for the same network and sample data in the directory
`tests/fmnist-....` (where .... is an autogenerated string based on the network topology), use:

    ./cnn-gen.py --verbose --autogen rtlsim --top-level cnn -L --test-dir rtlsim --prefix fmnist --checkpoint-file trained/ai84-fashionmnist.pth.tar --config-file networks/fashionmnist-chw.yaml

### Network Loader Configuration Language

Network descriptions are written in YAML (see https://en.wikipedia.org/wiki/YAML). There are two
sections in each file - global statements and a sequence of layer descriptions.

#### Global Configuration

##### `arch` (Mandatory)

`arch` specifies the network architecture, for example `ai84net5`. This key is matched against
the architecture embedded in the checkpoint file.

##### `dataset` (Mandatory)

`dataset` configures the data set for the network. This determines the input data size and
dimensions as well as the number of input channels.

Supported data sets at this time are `mnist`, `fashionmnist`, and `cifar-10`.

##### `output_map` (Optional)

The global `output_map`, if specified, overrides the memory instances where the last layer outputs
its results. If not specified, this will be either the `output_processors` specified for the last
layer, or, if that key does not exist, default to the number of processors needed for the output
channels, starting at 0.

Example: `output_map: 0x0000000000000ff0`

##### `layers` (Mandatory)

`layers` is a list that defines the per-layer description.

#### Per-Layer Configuration

Each layer in the `layers` list describes the layer's processors, convolution type, activation,
pooling, weight and output sizes, data input format, data memory offsets, and its processing
sequence. Several examples are located in the `networks/` and `tests/` folders.

##### `sequence` (Optional)

This key allows overriding of the processing sequence. The default is `0` for the first layer, or
the previous layer's sequence + 1 for other layers.

`sequence` numbers may have gaps. The software will sort layers by their numeric value, with the
lowest value first.

##### `processors` (Mandatory)

`processors` specifies which processors will handle the input data. The processor map must match
the number of input channels, and the input data format. For example, in CHW format, processors
must be attached to different data memory instances.

Example: `processors: 0x0000000000000111`

##### `output_processors` (Optional)

`output_processors` specifies which data memory instances and 32-bit word offsets to use for the
layer's output data. When not specified, this key defaults to the next layer's `processors`, or,
for the last layer, to the global `output_map`.

##### `out_offset` (Optional)

`out_offset` specifies the relative offset inside the data memory instance where the output data
should be written to. When not specified, `out_offset` defaults to `0`.

Example: `out_offset: 0x2000`.

##### `in_offset` (Optional)

`in_offset` specifies the offset into the data memory instances where the input data should be
loaded from. When not specified, this key defaults to the previous layer's `out_offset`, or `0` for
the first layer.

Example: `in_offset: 0x2000`.

##### `output_width` (Optional)

On AI84, this value (if specified) has to be `8`.

On AI85, when __not__ using an `activation`, the last layer can output `32` bits of unclipped data
in Q25.7 format. The default is `8` bits.

##### `data_format` (Optional)

When specified for the first layer only, `data_format` can be either `chw`/`big` or `hwc`/`little`.
The default is `hwc`. Note that the data format interacts with `processors`. 

##### `convolution` (Optional)

This selects between a 2D convolution (`conv2d`) and a 1D convolution (`conv1d`).

##### `activate` (Optional)

This key describes whether to activate the layer output (the default is to not activate). When
specified, this key must be `ReLU`.

##### `quantization` (Optional)

On AI84, this key must always be `8` (the default value).

On AI85, this key describes the width of the weight memory in bits and can be `1`, `2`, `4`, or
`8` (`8` is the default).

##### `kernel_size` (Optional)

This key must always be `3x3` (the default).

##### `stride` (Optional)

2D convolutions:

When pooling is used, this key must be `1`. When pooling is not used, this key can be set from `1`
to TBD.

1D convolutions:

This key must be set to `3`.

##### `pad` (Optional)

`pad` sets the padding for the convolution. For `Conv2D`, this value can be `0`, `1` (the default),
or `2`. For `Conv1D`, the value can be `0`, `3` (the default), or `6`.

##### `max_pool` (Optional)

When specified, performs a `MaxPool` before the convolution.

Example: `max_pool: 2`

##### `avg_pool` (Optional)

When specified, performs an `AvgPool` before the convolution.

Example: `avg_pool: 2`

##### `pool_stride` (Optional)

When performing a pooling operation, this key describes the pool stride. The default is `1`.

Example: `pool_stride: 2`


### Adding Datasets

Adding new datasets to the AI84 Network Loader is implemented by
1. Providing a network model, its YAML description and quantized weights.
2. Providing a sample input, and the expected output (modify `sampledata.py` as well as the
   `SUPPORTED_DATASETS` in `yamlcfg.py`).

## AI84 SDK

Check out the AI84 SDK from SVN, https://svn.maxim-ic.com/svn/mcbusw/Hardware/Micro/AI84/.

    svn co https://svn.maxim-ic.com/svn/mcbusw/Hardware/Micro/AI84/ AI84SDK

Additionally, the Arm embedded compiler is required, it is available from https://developer.arm.com/tools-and-software/open-source-software/developer-tools/gnu-toolchain/gnu-rm/downloads.

In order for the debugger to work, the OpenOCD `max32xxx` branch from
https://github.com/MaximIntegratedMicros/openocd.git must be installed, and an ai84.cfg file must
be installed in `/usr/local/share/openocd/scripts/target/` (or 
`C:\Maxim\Toolchain\share\openocd\scripts\target\` on Windows). A copy of the file is contained in
the `hardware` folder of the `ai8x-synthesis` project.

Windows binaries for OpenOCD can be installed via https://www.maximintegrated.com/en/design/software-description.html/swpart=SFW0001500A. This download also includes a version of the Arm
compilers.

To compile and install this OpenOCD version on macOS, use:

    git clone https://github.com/MaximIntegratedMicros/openocd.git
    cd openocd
    git checkout max32xxx
    MAKEINFO=true LIBTOOL=/usr/local/bin/glibtool ./configure --disable-dependency-tracking --enable-dummy --enable-buspirate --enable-jtag_vpi --enable-remote-bitbang --enable-legacy-ft2232_libftdi
    make
    make install

Additional SDK instructions can be found in a separate document,
https://svn.maxim-ic.com/svn/mcbusw/Hardware/Micro/AI84/docs/trunk/AI84%20Test%20Board%20Setup%20Instructions.docx.

## AI85/AI86

The `ai84ize.py` quantization tool and the `cnn-gen.py` network generator both have an 
`--ai85` command line argument that enables code generation for the AI85 and AI86.

The `--ai85` option enables:
* Bias shift << 7 (done).
* Per-layer support for 1, 2 and 4-bit weight sizes in addition to 8-bit weights (this is
  supported using the `quantization` keyword in the configuration file, and the configuration
  file can also be read by the quantization tool) (done).
* A scale factor on the output of the convolution that allows for better use of the entire range of
  weight bits (done).
* Support for many more pooling sizes, and strides, and larger limits for average pooling
  (in progress).
* 1D convolutions (in progress).
* 1x1 kernels (in progress).
* Support for more weight memory, and more input and output channels (in progress).
* Support for non-square data (in progress).
* Support for 32-bit Q25.7 data output for last layer when not using ReLU (in progress).


## CMSIS5 NN Emulation

The AI84 Network Loader tool can _also_ create code that executes the same exact network using
Arm's CMSISv5 Neural Networking library, optimized for Arm's Microcontroller DSP (reportedly
4.6x faster than executing on the CPU), see
https://developer.arm.com/solutions/machine-learning-on-arm/developer-material/how-to-guides/converting-a-neural-network-for-arm-cortex-m-with-cmsis-nn
and https://github.com/ARM-software/CMSIS_5 for the source code.

The results of the generated code have been verified to match AI84 exactly and may be used to
demonstrate the efficacy of the custom CNN accelerator.

The `Device/` folder contains a sample Makefile, and a custom fully connected layer in the
file `arm_fully_connected_q7_q8p7_opt.c` that returns Q8.7 fixed-point outputs, and a custom
`arm_softmax_q8p7_q15.c` which is aware of the fixed-point input (both of these files are also used
for the software classification layer on AI84/AI85).
Additionally, a `tornadocnn.h` header file is included which helps both embedded examples as
well as CMSIS NN code.

For example, the following command would generate code that performs a CIFAR-10 inference from the
`trained/ai84-cifar10.pth.tar` checkpoint file and the `cifar10-hwc.yaml` network description
file:

    (ai8x-synthesis) $ ./cnn-gen.py --top-level cnn --test-dir demos --prefix CIFAR-10-Arm --checkpoint-file trained/ai84-cifar10.pth.tar --config-file networks/cifar10-hwc.yaml --fc-layer --embedded-code --cmsis-software-nn

## Contributing Code

### Linting

Both projects are set up for `flake8`, `pylint`, and `mypy`. The line width is related to 100 (instead
of the default of 80), and the number of lines per module was increased.
Code should not generate any warnings in any of the tools (some of the components in the
`ai8x-training` project will create warnings as they are based on third-party code).

`flake8`, `pylint` and `mypy` need to be installed into the virtual environment:

    (ai8x-synthesis) $ pip3 install flake8 pylint mypy

### Submitting Changes

Do not try to push any changes into the master branch. Instead, create a local "feature" branch.
The easiest way to do this is using a graphical client such as [Fork](#Recommended-Software).
The command line equivalent is:

    $ git checkout ​-​b my-new-feature
    $ git status
    On branch my-new-feature
    ...

Commit changes and create a description of the changes:

    $ git commit

Check what the Maxim server is called:

    $ git remote -v
    origin	https://first.last@gerrit.maxim-ic.com:8443/ai8x-synthesis (fetch)
    origin	https://first.last@gerrit.maxim-ic.com:8443/ai8x-synthesis (push)

And push the changes to Maxim's Gerrit server using the name ('origin' in this example;
do not change 'master' to anything else even though the local branch is called 'my-new-feature'):

    $ git push origin HEAD​:​refs​/f​or​/m​aster
    ...
    remote​:
    To​ URL
    ​*​ ​[​new​ branch​]​ my-new-feature ​->​ refs​/​for​/​master

Open the URL in a web browser and request a review for the change list.

-----------------------
