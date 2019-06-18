###################################################################################################
# Copyright (C) 2018-2019 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Confidential
#
# Written by RM
###################################################################################################
"""
Routines to read and write the APB peripherals.
"""
import sys

import toplevel
import tornadocnn
import unload


class APB(object):
    """
    APB read and write functionality.
    """

    def __init__(self,
                 memfile,
                 apb_base,
                 verify_writes=False,
                 no_error_stop=False,
                 weight_header=None,
                 sampledata_header=None,
                 embedded_code=False):
        """
        Create an APB class object that writes to memfile.
        """
        self.memfile = memfile
        self.apb_base = apb_base
        self.verify_writes = verify_writes
        self.no_error_stop = no_error_stop
        self.weight_header = weight_header
        self.sampledata_header = sampledata_header
        self.embedded_code = embedded_code

        self.data = 0
        self.num = 0
        self.data_offs = 0
        self.mem = [None] * tornadocnn.C_GROUP_OFFS * tornadocnn.P_NUMGROUPS

    def write(self,
              addr,
              val,
              comment='',
              no_verify=False):  # pylint: disable=unused-argument
        """
        Write address `addr` and data `val` to the output file.
        if `no_verify` is `True`, do not check the result of the write operation, even if
        `verify_writes` is globally enabled.
        An optional `comment` can be added to the output.
        """
        raise NotImplementedError

    def verify(self,
               addr,
               val,
               comment='',
               rv=False):  # pylint: disable=unused-argument
        """
        Verify that memory at address `addr` contains data `val`.
        """
        raise NotImplementedError

    def set_memfile(self,
                    memfile):
        """
        Change the file handle to `memfile` and reset the .mem output location to 0.
        """
        self.memfile = memfile

    def write_ctl(self, group, reg, val, debug=False, comment=''):
        """
        Set global control register `reg` in group `group` to value `val`.
        """
        if comment is None:
            comment = f' // global ctl {reg}'
        addr = tornadocnn.C_GROUP_OFFS*group + tornadocnn.C_CNN_BASE + reg*4
        self.write(addr, val, comment)
        if debug:
            print(f'R{reg:02} ({addr:08x}): {val:08x}{comment}')

    def write_lreg(self, group, layer, reg, val, debug=False, comment=''):
        """
        Set layer `layer` register `reg` in group `group` to value `val`.
        """
        if comment is None:
            comment = f' // reg {reg}'
        addr = tornadocnn.C_GROUP_OFFS*group + tornadocnn.C_CNN_BASE + tornadocnn.C_CNN*4 \
            + reg*4 * tornadocnn.MAX_LAYERS + layer*4
        self.write(addr, val, comment)
        if debug:
            print(f'G{group} L{layer} R{reg:02} ({addr:08x}): {val:08x}{comment}')

    def write_bias(self, group, offs, bias):
        """
        Write bias value `bias` to offset `offs` in bias memory #`group`.
        """
        addr = tornadocnn.C_GROUP_OFFS*group + tornadocnn.C_BRAM_BASE + offs * 4
        self.write(addr, bias & 0xff, f' // Bias')

    def write_tram(self, group, proc, offs, d, comment=''):
        """
        Write value `d` to TRAM in group `group` and processor `proc` to offset `offs`.
        """
        addr = tornadocnn.C_GROUP_OFFS*group + tornadocnn.C_TRAM_BASE \
            + proc * tornadocnn.TRAM_SIZE * 4 + offs * 4
        self.write(addr, d, f' // {comment}TRAM G{group} P{proc} #{offs}')

    def write_kern(self, ll, ch, idx, k):
        """
        Write single 3x3 kernel `k` for layer `ll`, channel `ch` to index `idx` in weight
        memory.
        """
        assert ch < tornadocnn.MAX_CHANNELS
        assert idx < tornadocnn.MASK_WIDTH
        addr = tornadocnn.C_GROUP_OFFS * (ch // tornadocnn.P_NUMPRO) \
            + tornadocnn.C_MRAM_BASE \
            + (ch % tornadocnn.P_NUMPRO) * tornadocnn.MASK_OFFS * 16 + idx * 16

        self.write(addr, k[0] & 0xff, no_verify=True,
                   comment=f' // Layer {ll}: processor {ch} kernel #{idx}')
        self.write(addr+4, (k[1] & 0xff) << 24 | (k[2] & 0xff) << 16 |
                   (k[3] & 0xff) << 8 | k[4] & 0xff, no_verify=True)
        self.write(addr+8, (k[5] & 0xff) << 24 | (k[6] & 0xff) << 16 |
                   (k[7] & 0xff) << 8 | k[8] & 0xff, no_verify=True)
        self.write(addr+12, 0, no_verify=True)  # Execute write
        if self.verify_writes:
            self.verify(addr, k[0] & 0xff)
            self.verify(addr+4, (k[1] & 0xff) << 24 | (k[2] & 0xff) << 16 |
                        (k[3] & 0xff) << 8 | k[4] & 0xff)
            self.verify(addr+8, (k[5] & 0xff) << 24 | (k[6] & 0xff) << 16 |
                        (k[7] & 0xff) << 8 | k[8] & 0xff)
            self.verify(addr+12, 0)

    def check_overwrite(self, offs):
        """
        Check whether we're overwriting location `offs`.
        """
        if self.mem[offs >> 2]:
            print(f'Overwriting location {offs:08x}')
            if not self.no_error_stop:
                sys.exit(1)

    def write_byte_flush(self, offs, comment=''):
        """
        Flush the contents of the internal buffer at offset `offs`, adding an optional
        `comment` to the output.
        This function also keeps track of all addresses that have been written before and
        can detect whether previous information is being overwritten.
        """
        if self.num > 0:
            woffs = self.data_offs - self.num
            self.check_overwrite(woffs)
            self.write(woffs, self.data, comment)
            self.mem[woffs >> 2] = True
            self.num = 0
            self.data = 0
        self.data_offs = offs

    def write_byte(self, offs, val, comment=''):
        """
        Add byte `val` that should be written at offset `offs` to the internal buffer.
        When reaching 4 bytes, or when the offset is not contiguous, pad with zeros and
        flush the before adding the new value to the buffer.
        An optional `comment` can be added to the output.
        """
        if offs != self.data_offs:
            self.write_byte_flush(offs)

        # Collect and write if multiple of 4 (little endian byte order)
        self.data |= (val & 0xff) << (8*self.num)
        self.num += 1
        self.data_offs += 1
        if self.num == 4:
            self.write_byte_flush(offs+1, comment)

    def get_mem(self):
        """
        Return reference to the memory array.
        """
        return self.mem

    def output(self, comment):
        """
        Write the string `comment` to the output file without further interpretation.
        """
        if self.memfile is None:
            return

        self.memfile.write(comment)

    def copyright_header(self):
        """
        Write copyright headers.
        The base class does nothing.
        """
        return

    def header(self):
        """
        Write file headers.
        The base class does nothing.
        """
        return

    def verify_header(self):
        """
        Write the header for the CNN verification function.
        The base class does nothing.
        """
        return

    def verify_footer(self):
        """
        Write the footer for the CNN verification function.
        The base class does nothing.
        """
        return

    def load_header(self):
        """
        Write the header for the CNN configuration loader function.
        The base class does nothing.
        """
        return

    def load_footer(self):
        """
        Write the footer for the CNN configuration loader function.
        The base class does nothing.
        """
        return

    def main(self, classification_layer=False):  # pylint: disable=unused-argument
        """
        Write the main function, including an optional call to the fully connected layer if
        `classification_layer` is `True`.
        The base class does nothing.
        """
        return

    def fc_layer(self, weights, bias):  # pylint: disable=unused-argument
        """
        Write the call to the fully connected layer for the given `weights` and
        `bias`. The `bias` argument can be `None`.
        The base class does nothing.
        """
        return

    def fc_verify(self, data):  # pylint: disable=unused-argument
        """
        Write the code to verify the fully connected layer against `data`.
        The base class does nothing.
        """
        return

    def unload(self, processor_map, input_shape,  # pylint: disable=unused-argument
               output_offset=0):  # pylint: disable=unused-argument
        """
        Write the unload function. The layer to unload has the shape `input_shape`,
        and the optional `output_offset` argument can shift the output.
        The base class does nothing.
        """
        return

    def output_define(self, array, define_name,  # pylint: disable=unused-argument
                      fmt, columns, weights=True):  # pylint: disable=unused-argument
        """
        Write a #define for array `array` to `define_name`, using format `fmt` and creating
        a line break after `columns` items each.
        The base class does nothing.
        """
        return


class APBBlockLevel(APB):
    """
    APB read and write functionality for block level tests.
    """
    def __init__(self,
                 memfile,
                 apb_base,
                 verify_writes=False,
                 no_error_stop=False,
                 weight_header=None,
                 sampledata_header=None,
                 embedded_code=False):
        super(APBBlockLevel, self).__init__(memfile,
                                            apb_base,
                                            verify_writes=verify_writes,
                                            no_error_stop=no_error_stop,
                                            weight_header=weight_header,
                                            sampledata_header=sampledata_header,
                                            embedded_code=embedded_code)
        self.foffs = 0

    def write(self,
              addr,
              val,
              comment='',
              no_verify=False):  # pylint: disable=unused-argument
        """
        Write address `addr` and data `val` to the .mem file.
        """
        assert val >= 0
        assert addr >= 0
        addr += self.apb_base

        self.memfile.write(f'@{self.foffs:04x} {addr:08x}\n')
        self.memfile.write(f'@{self.foffs+1:04x} {val:08x}\n')
        self.foffs += 2

    def verify(self,
               addr,
               val,
               comment='',
               rv=False):  # pylint: disable=unused-argument
        """
        Verify that memory at address `addr` contains data `val`.
        For block level tests, this function does nothing useful other than ensuring the inputs
        address and data are not negative.
        """
        assert val >= 0
        assert addr >= 0

    def set_memfile(self, memfile):
        """
        Change the file handle to `memfile` and reset the .mem output location to 0.
        """
        super(APBBlockLevel, self).set_memfile(memfile)
        self.foffs = 0


class APBTopLevel(APB):
    """
    APB read and write functionality for top level tests.
    """
    def write(self,
              addr,
              val,
              comment='',
              no_verify=False):
        """
        Write address `addr` and data `val` to the .c file.
        if `no_verify` is `True`, do not check the result of the write operation, even if
        `verify_writes` is globally enabled.
        An optional `comment` can be added to the output.
        """
        assert val >= 0
        assert addr >= 0
        addr += self.apb_base

        if self.memfile is None:
            return

        self.memfile.write(f'  *((volatile uint32_t *) 0x{addr:08x}) = 0x{val:08x};{comment}\n')
        if self.verify_writes and not no_verify:
            self.memfile.write(f'  if (*((volatile uint32_t *) 0x{addr:08x}) != 0x{val:08x}) '
                               'return 0;\n')

    def verify(self,
               addr,
               val,
               comment='',
               rv=False):
        """
        Verify that memory at address `addr` contains data `val`.
        If `rv` is `True`, do not immediately return 0, but just set the status word.
        An optional `comment` can be added to the output.
        """
        assert val >= 0
        assert addr >= 0
        addr += self.apb_base

        if self.memfile is None:
            return

        if rv:
            self.memfile.write(f'  if (*((volatile uint32_t *) 0x{addr:08x}) != 0x{val:08x}) '
                               f'return 0;{comment}\n')  # FIXME
        else:
            self.memfile.write(f'  if (*((volatile uint32_t *) 0x{addr:08x}) != 0x{val:08x}) '
                               f'return 0;{comment}\n')

    def copyright_header(self):
        """
        Write copyright headers.
        """
        toplevel.copyright_header(self.memfile)

    def header(self):
        """
        Write include files and forward definitions to .c file.
        """
        toplevel.header(self.memfile, self.apb_base, embedded_code=self.embedded_code)

    def verify_header(self):
        """
        Write the header for the CNN verification function.
        """
        toplevel.verify_header(self.memfile)

    def verify_footer(self):
        """
        Write the footer for the CNN verification function.
        """
        toplevel.verify_footer(self.memfile)

    def load_header(self):
        """
        Write the header for the CNN configuration loader function.
        """
        toplevel.load_header(self.memfile)

    def load_footer(self):
        """
        Write the footer for the CNN configuration loader function.
        """
        toplevel.load_footer(self.memfile, embedded_code=self.embedded_code)

    def main(self, classification_layer=False):
        """
        Write the main function, including an optional call to the fully connected layer if
        `classification_layer` is `True`.
        """
        toplevel.main(self.memfile, classification_layer=classification_layer,
                      embedded_code=self.embedded_code)

    def fc_layer(self, weights, bias):
        """
        Write call to the fully connected layer for the given `weights` and
        `bias`. The `bias` argument can be `None`.
        """
        toplevel.fc_layer(self.memfile, self.weight_header, weights, bias)

    def fc_verify(self, data):
        """
        Write the code to verify the fully connected layer against `data`.
        """
        toplevel.fc_verify(self.memfile, self.sampledata_header, data)

    def unload(self, processor_map, input_shape, output_offset=0):
        """
        Write the unload function. The layer to unload has the shape `input_shape`,
        and the optional `output_offset` argument can shift the output.
        """
        unload.unload(self.memfile, self.apb_base, processor_map, input_shape, output_offset)

    def output_define(self, array, define_name, fmt, columns, weights=True):
        """
        Write a #define for array `array` to `define_name`, using format `fmt` and creating
        a line break after `columns` items each.
        If `weight`, write to the `weights.h` file, else to `sampledata.h`.
        """
        if weights:
            toplevel.c_define(self.weight_header, array, define_name, fmt, columns)
        else:
            toplevel.c_define(self.sampledata_header, array, define_name, fmt, columns)


def apbwriter(memfile,
              apb_base,
              block_level=False,
              verify_writes=False,
              no_error_stop=False,
              weight_header=None,
              sampledata_header=None,
              embedded_code=False):
    """
    Depending on `block_level`, return a block level .mem file writer or a top level .c file
    writer to the file `memfile` with APB base address `apb_base`.
    If `verify_writes` is set, read back everything that was written.
    If `no_error_stop` is set, continue in the case when the data is trying to overwrite
    previously written data.
    """
    APBClass = APBBlockLevel if block_level else APBTopLevel
    return APBClass(memfile,
                    apb_base,
                    verify_writes=verify_writes,
                    no_error_stop=no_error_stop,
                    weight_header=weight_header,
                    sampledata_header=sampledata_header,
                    embedded_code=embedded_code)
