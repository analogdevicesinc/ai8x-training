# AI8X Model Training and Quantization
# AI8X Network Loader and RTL Simulation Generator

_9/27/2019_

_Open this file in a markdown enabled viewer, for example Typora (http://typora.io) or Visual Studio Code 
(https://code.visualstudio.com). See https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet for a description of Markdown._

This software consists of two related projects:
1. AI8X Model Training and Quantization
2. AI8X Network Loader and RTL Simulation Generator

---

## Contents

- [Contents](#contents)
- [Overview](#overview)
- [Installation](#installation)
  - [File System Layout](#file-system-layout)
  - [Upstream Code](#upstream-code)
  - [Prerequisites](#prerequisites)
    - [Recommended Software](#recommended-software)
  - [Project Installation](#project-installation)
    - [Windows Systems](#windows-systems)
    - [Creating the Virtual Environment](#creating-the-virtual-environment)
    - [Building TensorFlow (for old CPUs)](#building-tensorflow-for-old-cpus)
    - [Nervana Distiller](#nervana-distiller)
    - [Synthesis Project](#synthesis-project)
- [AI8X Hardware and Resources](#ai8x-hardware-and-resources)
  - [Overview](#overview-1)
  - [Data, Weights, and Processors](#data-weights-and-processors)
    - [Weight Memory](#weight-memory)
    - [Data Memory](#data-memory)
  - [Streaming Mode](#streaming-mode)
  - [Accelerator Limits](#accelerator-limits)
  - [Number Format](#number-format)
    - [Rounding](#rounding)
    - [Addition](#addition)
    - [Saturation and Clipping](#saturation-and-clipping)
    - [Multiplication](#multiplication)
    - [Sign Bit](#sign-bit)
  - [Channel Data Formats](#channel-data-formats)
    - [HWC](#hwc)
    - [CHW](#chw)
  - [CHW Data Format and Consequences for Weight Memory Layout](#chw-data-format-and-consequences-for-weight-memory-layout)
  - [Active Processors and Layers](#active-processors-and-layers)
  - [Layers and Weight Memory](#layers-and-weight-memory)
  - [Weight Storage Example](#weight-storage-example)
  - [Example: `Conv2D`](#example-conv2d)
  - [Limitations of AI84 Networks](#limitations-of-ai84-networks)
  - [Limitations of AI85 Networks](#limitations-of-ai85-networks)
  - [Fully Connected (Linear) Layers](#fully-connected-linear-layers)
- [Model Training and Quantization](#model-training-and-quantization)
  - [Quantization](#quantization)
  - [Alternative Quantization Approaches](#alternative-quantization-approaches)
  - [Adding Datasets and New Networks to the Training Process](#adding-datasets-and-new-networks-to-the-training-process)
- [Network Loader](#network-loader)
  - [Network Loader Configuration Language](#network-loader-configuration-language)
    - [Global Configuration](#global-configuration)
      - [`arch` (Mandatory)](#arch-mandatory)
      - [`bias` (Optional, Test Only)](#bias-optional-test-only)
      - [`dataset` (Mandatory)](#dataset-mandatory)
      - [`output_map` (Optional)](#outputmap-optional)
      - [`layers` (Mandatory)](#layers-mandatory)
    - [Per-Layer Configuration](#per-layer-configuration)
      - [`sequence` (Optional)](#sequence-optional)
      - [`processors` (Mandatory)](#processors-mandatory)
      - [`output_processors` (Optional)](#outputprocessors-optional)
      - [`out_offset` (Optional)](#outoffset-optional)
      - [`in_offset` (Optional)](#inoffset-optional)
      - [`output_width` (Optional)](#outputwidth-optional)
      - [`data_format` (Optional)](#dataformat-optional)
      - [`operation` (Optional)](#operation-optional)
      - [`eltwise` (Optional)](#eltwise-optional)
      - [`pool_first` (Optional)](#poolfirst-optional)
      - [`operands` (Optional)](#operands-optional)
      - [`activate` (Optional)](#activate-optional)
      - [`quantization` (Optional)](#quantization-optional)
      - [`kernel_size` (Optional)](#kernelsize-optional)
      - [`stride` (Optional)](#stride-optional)
      - [`pad` (Optional)](#pad-optional)
      - [`max_pool` (Optional)](#maxpool-optional)
      - [`avg_pool` (Optional)](#avgpool-optional)
      - [`pool_stride` (Optional)](#poolstride-optional)
      - [`in_dim` (Optional)](#indim-optional)
      - [`streaming` (Optional)](#streaming-optional)
      - [`flatten` (Optional)](#flatten-optional)
  - [Adding Datasets to the Network Loader](#adding-datasets-to-the-network-loader)
  - [Starting an Inference, Waiting for Completion, Multiple Inferences in Sequence](#starting-an-inference-waiting-for-completion-multiple-inferences-in-sequence)
  - [CMSIS5 NN Emulation](#cmsis5-nn-emulation)
- [AI84 SDK](#ai84-sdk)
- [AI85/AI86 Changes](#ai85ai86-changes)
- [Updating the Project](#updating-the-project)
- [Contributing Code](#contributing-code)
  - [Linting](#linting)
  - [Submitting Changes](#submitting-changes)

## Overview

The following graphic shows an overview of the development flow:

![Development Flow](docs/DevelopmentFlow.png)

## Installation

### File System Layout

Including the SDK from SVN, the expected file system layout will be:

    ..../ai8x-training/
    ..../ai8x-synthesis/
    ..../AI84SDK/

### Upstream Code

Change to the project root (denoted as `....` above).

```shell
$ git clone https://first.last@gerrit.maxim-ic.com:8443/ai8x-training
$ git clone https://first.last@gerrit.maxim-ic.com:8443/ai8x-synthesis
```

### Prerequisites

When going beyond simple tests, model training requires CUDA hardware acceleration (the network loader does not require CUDA).

Install CUDA10 and CUDNN:
https://developer.nvidia.com/cuda-downloads
https://developer.nvidia.com/cudnn

#### Recommended Software

The following software is optional, and can be replaced with other similar software of the user’s choosing.

1. Visual Studio Code (Editor, Free), https://code.visualstudio.com
2. Typora (Markdown Editor, Free during beta), http://typora.io
3. CoolTerm (Serial Terminal, Free), http://freeware.the-meiers.org
   or Serial ($30), https://apps.apple.com/us/app/serial/id877615577?mt=12
4. Git Fork (Graphical Git Client, Free), https://git-fork.com
5. Beyond Compare (Diff and Merge Tool, $60), https://scootersoftware.com

### Project Installation

*The software in this project requires Python 3.6.9 or a later 3.6.x version.*

It is not necessary to install Python 3.6.9 system-wide, or to rely on the system-provided Python. To manage Python versions, use `pyenv` (https://github.com/pyenv/pyenv).

On macOS (no CUDA support available):

```shell
$ brew install pyenv pyenv-virtualenv libomp
```

On Ubuntu 18.04 LTS:

```
$ sudo apt-get install -y make build-essential libssl-dev zlib1g-dev \
  libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
  libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev
$ curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash
```

Then, add to ~/.bash_profile or ~/.profile (as shown by the previous step):

```shell
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
```

Next, close the Terminal and install Python 3.6.9:

```shell
$ pyenv install 3.6.9
```

#### Windows Systems

Windows is not supported. Please see `docs/Windows.md` for unofficial Windows installation notes.

#### Creating the Virtual Environment

To create the virtual environment and install basic wheels:

```shell
$ cd ai8x-training
$ git submodule update --init
$ pyenv local 3.6.9
$ python3 -m venv .
$ source bin/activate
(ai8x-training) $ pip3 install -U pip setuptools
(ai8x-training) $ pip3 install -r requirements.txt
```

For macOS and systems without CUDA:
```shell
(ai8x-training) $ pip3 install torch==1.1.0
(ai8x-training) $ pip3 install torchvision==0.3.0
```

For CUDA 10 on Linux (see https://pytorch.org/get-started/locally/):

```shell
(ai8x-training) $ pip3 install https://download.pytorch.org/whl/cu100/torch-1.1.0-cp36-cp36m-linux_x86_64.whl
(ai8x-training) $ pip3 install https://download.pytorch.org/whl/cu100/torchvision-0.3.0-cp36-cp36m-linux_x86_64.whl
```

Next, install TensorFlow. TensorFlow is needed for TensorBoard support.

```shell
(ai8x-training) $ pip3 install "tensorflow>=1.14"
```

*Note*: On x86 systems, the pre-built TensorFlow wheels require AVX (on macOS, run `sysctl -n machdep.cpu.features` to find out, on Linux use `cat /proc/cpuinfo | grep avx`).
Running a TensorFlow wheel that requires AVX instructions on unsupported CPUs results in `Illegal instruction: 4` on startup.

#### Building TensorFlow (for old CPUs)

If the CPU does not support AVX, or to enable support for AVX2, or CUDA, or AMD64, build TensorFlow locally. Otherwise, skip ahead to [Nervana Distiller](#Nervana-Distiller).
_To repeat: Building TensorFlow is not needed when the binary wheels are functioning properly._

The TensorFlow build requires Java 8 and Bazel, and takes over two hours. 
Building TensorFlow requires Bazel 0.25.2 (newer or much older versions do not work). See
https://docs.bazel.build/versions/master/install-compile-source.html#build-bazel-using-bazel for build instructions.

Once Bazel is installed, compile and install TensorFlow. On Jetson TX1, disable S3. See 
https://github.com/peterlee0127/tensorflow-nvJetson/blob/master/patch/tensorflow1.12.patch
The removal of `-mfpu=neon` does not seem to be necessary.

```shell
(ai8x-training) $ pip3 install keras_preprocessing
(ai8x-training) $ git clone https://github.com/tensorflow/tensorflow 
(ai8x-training) $ cd tensorflow
(ai8x-training) $ git checkout tags/v1.14.0
(ai8x-training) $ ./configure
(ai8x-training) $ bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package
(ai8x-training) $ bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
(ai8x-training) $ pip3 install /tmp/tensorflow_pkg/tensorflow-1.14.0-cp36-cp36m-macosx_10_13_x86_64.whl 
```

#### Nervana Distiller

To install Nervana’s distiller:

```shell
(ai8x-training) $ pip3 install -e distiller
```

On macOS, add to `~/.matplotlib/matplotrc`:

    backend: TkAgg

#### Synthesis Project

For `ai8x-synthesis`, some of the installation steps can be simplified. Specifically, CUDA is not necessary.

Start by creating a second virtual environment:

```shell
$ cd ai8x-synthesis
$ git submodule update --init
$ pyenv local 3.6.9
$ python3 -m venv .
$ source bin/activate
(ai8x-synthesis) $ pip3 install -U pip setuptools
(ai8x-synthesis) $ pip3 install -r requirements.txt
```

---

## AI8X Hardware and Resources

AI8X are embedded accelerators. Unlike GPUs, AI8X do not have gigabytes of memory, and cannot support arbitrary data (image) sizes.

### Overview

A typical CNN operation consists of pooling followed by a convolution. While these are traditionally expressed as separate layers, pooling can be done “in-flight” on AI8X for greater efficiency.

To minimize data movement, the accelerator is optimized for convolutions with in-flight pooling on a sequence of layers. AI85 and AI86 also support in-flight element-wise operations, pass-through layers and 1D convolutions (without element-wise operations):

![CNNInFlight](docs/CNNInFlight.png)

The AI8X accelerator consists of 64 parallel processors. Each processor includes a pooling unit and a convolutional engine with dedicated weight memory:

![Overview](docs/Overview.png)

Data is read from data memory associated with the processor, and written out to any data memory located within the accelerator. To run a deep convolutional neural network, multiple layers are chained together, where each layer’s operation is individually configurable. The output data from one layer is used as the input data for the next layer, for up to 32 layers (where in-flight pooling operations do not count as layers).

The following picture shows an example view of a 2D convolution with pooling:
![Example](docs/CNNOverview.png)

### Data, Weights, and Processors

Data memory, weight memory, and processors are interdependent.

In the AI8X accelerator, processors are organized as follows:

* Each processor is connected to its own dedicated weight memory instance.
* Four processors share one data memory instance.
* A group of sixteen processors shares certain common controls and can be operated as a slave to another group, or independently/separately.

Any given processor has visibility of:

* Its dedicated weight memory, and
* The data memory instance it shares with three other processors.

#### Weight Memory

For each of the four 16-processor groups, weight memory and processors can be visualized as follows. Assuming one input channel processed by processor 0, and 8 output channels, the 8 shaded kernels will be used:

![Weight Memory Map](docs/KernelMemory.png)

#### Data Memory

Data memory connections can be visualized as follows:

<img src="docs/DataMemory.png" alt="Data Memory Map" style="zoom: 67%;" />

All input data must be located in the data memory instance the processor can access. Conversely, output data can be written to any data memory instance inside the accelerator (but not to general purpose SRAM on the Arm microcontroller bus).

The data memory instances inside the accelerator are single-port memories. This means that only one access operation can happen per clock cycle. When using the HWC data format (see [Channel Data Formats](#Channel-Data-Formats)), this means that each of the four processors sharing the data memory instance will receive one byte of data per clock cycle (since each 32-bit data word consists of four packed channels).

### Streaming Mode

On AI85/AI86, the machine implements a streaming mode. Streaming allows input data dimensions that exceed the available per-channel data memory in the accelerator.

The following illustration shows the basic principle: In order to produce the first output pixel of the second layer, not all data needs to be present at the input. In the example, a 5×5 input needs to be available.

<img src="docs/Streaming.png"/>

In the accelerator implementation, data is shifted into the Tornado memory in a sequential fashion, so prior  rows will be available as well. In order to produce the _blue_ output pixel, input data up to the blue input pixel must be available.

![Streaming-Rows](docs/Streaming-Rows.png)

When the _yellow_ output pixel is produced, the first (_black_) pixel of the input data is no longer needed and its data can be discarded:

![Streaming-NextPixel](docs/Streaming-NextPixel.png)

The number of discarded pixels is network specific and dependent on pooling strides and the types of convolution. In general, streaming mode is only useful for networks where the output data dimensions decrease from layer to layer (for example, by using a pooling stride).

The AI85/AI86 accelerator has four dedicated FIFOs connected to processors 0, 16, 32, and 48. Since the data memory instances are single-port memories, software would otherwise have to temporarily disable the accelerator in order to feed it new data. Using the FIFOs, software can input available data while the accelerator is running. The accelerator will autonomously fetch data from the FIFOs when needed, and stall (pause) when no enough data is available.

### Accelerator Limits

* AI84:
  * The maximum number of layers is 32 (pooling layers do not count).
  * The maximum number of input or output channels in any layer is 64 each.
  * The weight memory supports up to 128 * 64 3×3 Q7 kernels (see [Number Format](#Number-Format)).
    However, weights must be arranged according to specific rules detailed below.
  * There are 16 instances of 16 KB data memory. Any data channel (input, intermediate, or output) must completely fit into one memory instance. This limits the first-layer input to 128×128 pixels per channel in the CHW format. However, when using more than one input channel, the HWC format may be preferred, and all layer output are in HWC format as well. In those cases, it is required that four channels fit into a single memory instance -- or 64×64 pixels per channel. Note that the first layer commonly creates a wide expansion (i.e., large number of output channels) that needs to fit into data memory, so the input size limit is mostly theoretical.
* AI85:
  * The maximum number of layers is 32 (pooling and element-wise layers do not count).
  * The maximum number of input or output channels in any layer is 512 each.
  * The weight memory supports up to 768 * 64 3×3 Q7 kernels (see [Number Format](#Number-Format)).
    When using 1-, 2- or 4 bit weights, the capacity increases accordingly.
    When using more than 64 input or output channels, weight memory is shared and effective capacity decreases.
    Weights must be arranged according to specific rules detailed below.
  * There are 16 instances of 32 KB data memory. When not using streaming mode, any data channel (input, intermediate, or output) must completely fit into one memory instance. This limits the first-layer input to 181×181pixels per channel in the CHW format. However, when using more than one input channel, the HWC format may be preferred, and all layer output are in HWC format as well. In those cases, it is required that four channels fit into a single memory instance -- or 91×90 pixels per channel.
    Note that the first layer commonly creates a wide expansion (i.e., large number of output channels) that needs to fit into data memory, so the input size limit is mostly theoretical.
  * When using streaming, the data sizes are limited to 512×512, subject to available TRAM. Streaming is limited to 8 layers or less, and to four FIFOs (up to 4 input channels in CHW and up to 16 channels in HWC format).

### Number Format

All weights, bias values and data are stored and computed in Q7 format (signed two’s complement 8-bit integers, [-128...+127]). See https://en.wikipedia.org/wiki/Q_%28number_format%29.

The 8-bit value $w$ is defined as:

$$ w = (-a_7 2^7+a_6 2^6+a_5 2^5+a_4 2^4+a_3 2^3+a_2 2^2+a_1 2^1+a_0)/128 $$

<img src="docs/Numbers.png" alt="76543210" style="zoom:67%;" />

Examples:
| Binary	| Value        |
|:---------:|-------------:|
| 0000 0000 | 0            |
| 0000 0001 | 1/128        |
| 0000 0010 | 2/128        |
| 0111 1110 | 126/128      |
| 0111 1111 | 127/128      |
| 1000 0000 | −128/128 (–1)|
| 1000 0001 | −127/128     |
| 1000 0010 | −126/128     |
| 1111 1110 | −2/128       |
| 1111 1111 | −1/128       |

On **AI85**, _weights_ can be 1, 2, 4, or 8 bits wide (configurable per layer using the `quantization` key). Bias values are always 8 bits wide. Data is 8 bits wide, except for the last layer that can optionally output 32 bits of unclipped data in Q25.7 format when not using activation.

|wt bits| min  | max  |
|:-----:|-----:|-----:|
|    8  | –128 | +127 |
|    4  |   –8 |    7 |
|    2  |   –2 |    1 |
|    1  |   –1 |    0 |

Note that 1-bit weights (and, to a lesser degree, 2-bit weights) require the use of bias to produce useful results. Without bias, all sums of products of activated data from a prior layer would be negative, and activation of that data would always be zero.

#### Rounding

AI8X rounding (for the CNN sum of products) uses “round half towards positive infinity”, i.e. $y=⌊0.5+x⌋$. This rounding method is not the default method in either Excel or Python/NumPy. The rounding method can be achieved in NumPy using `y = np.floor(0.5 + x)` and in Excel as `=FLOOR.PRECISE(0.5 + X)`.

By way of example:

| Input                    | Rounded |
|:-------------------------|:-------:|
| +3.5                     | +4      |
| +3.25, +3.0, +2.75, +2.5 | +3      |
| +2.25, +2.0, +1.75, +1.5 | +2      |
| +1.25, +1.0, +0.75, +0.5 | +1      |
| +0.25, 0, –0.25, –0.5    | 0       |
| –0.75, –1.0, –1.25, –1.5 | –1      |
| –1.75, –2.0, –2.25, –2.5 | –2      |
| –2.75, –3.0, –3.25, –3.5 | –3      |

#### Addition

Addition works similarly to regular two’s-complement arithmetic.

Example:
$$ w_0 = 1/64 → 00000010 $$
$$ w_1 = 1/2 → 01000000 $$
$$ w_0 + w_1 = 33/64 → 01000010 $$

#### Saturation and Clipping

Values smaller than $–128⁄128$ are saturated to $–128⁄128$ (1000 0000). Values larger than $+127⁄128$ are saturated to $+127⁄128$ (0111 1111).

The AI8X CNN sum of products uses full resolution for both products and sums, so the saturation happens only at the very end of the computation.

Example 1:

$$ w_0 = 127/128 → 01111111 $$
$$ w_1 = 127/128 → 01111111 $$
$$ w_0 + w_1 = 254/128 → saturate → 01111111 (= 127/128) $$

Example 2:

$$ w_0 = -128/128 → 10000000 $$
$$ w_1 = -128/128 → 10000000 $$
$$ w_0 + w_1 = -256/128 → saturate → 10000000 (= -128/128) $$

#### Multiplication

Since operand values are implicitly divided by 128, the product of two values has to be shifted in order to maintain magnitude when using a standard multiplier (e.g., 8×8):

$$ w_0 * w_1 = \frac{w'_0}{128} * \frac{w'_1}{128} = \frac{w'_0 * w'_1}{128} ≫ 7 $$

In software,
* Determine the sign bit: $s = sign(w_0) * sign(w_1)$
* Convert operands to absolute values: $w'_0 = abs(w_0); w'_1 = abs(w_1)$
* Multiply using standard multiplier: $w'_0 * w'_1 = w''_0/128 * w''_1/128; r' = w''_0 * w''_1$
* Shift: $r'' = r' ≫ 7$
* Round up/down depending on $r'[6]$
* Apply sign: $r = s * r''$

Example 1:

$$ w_0 = 1/64 → 00000010 $$
$$ w_1 = 1/2 → 01000000 $$
$$ w_0 * w_1 = 1/128 → shift, truncate → 00000001 (= 1/128) $$

A “standard” two’s-complement multiplication would return 00000000 10000000. The AI8X data format discards the rightmost bits.

Example 2:

$$ w_0 = 1/64 → 00000010 $$
$$ w_1 = 1/4 → 00100000 $$
$$ w_0 * w_1 = 1/256 → shift, truncate → 00000000 (= 0) $$

“Standard” two’s-complement multiplication would return 00000000 01000000, the AI8X result is truncated to 0 after the shift operation.

#### Sign Bit

Operations preserve the sign bit.

Example 1:

$$ w_0 = -1/64 → 11111110 $$
$$ w_1 = 1/4 → 00100000 $$
$$ w_0 * w_1 = -1/256 → shift, truncate → 00000000 (= 0) $$

* Determine the sign bit: $s = sign(-1/64) * sign(1/4) = -1 * 1 = -1$
* Convert operands to absolute values: $w'_0 = abs(-1/64); w'_1 = abs(1/4)$
* Multiply using standard multiplier: $r' = 1/64 ≪ 7 * 1/4 ≪ 7 = 2 * 32 = 64$
* Shift: $r'' = r' ≫ 7 = 64 ≫ 7 = 0$
* Apply sign: $r = s * r'' = -1 * 0 = 0$

Example 2:

$$ w_0 = -1/64 → 11111110 $$
$$ w_1 = 1/2 → 01000000 $$
$$ w_0 * w_1 = -1/128 → shift, truncate → 11111111 (= -1/128) $$

* Determine the sign bit: $s = sign(-1/64) * sign(1/2) = -1 * 1 = -1$
* Convert operands to absolute values: $w'_0 = abs(-1/64); w'_1 = abs(1/2)$
* Multiply using standard multiplier: $r' = 1/64 ≪ 7 * 1/2 ≪ 7 = 2 * 64 = 128$
* Shift: $r'' = r' ≫ 7 = 128 ≫ 7 = 1$
* Apply sign: $r = s * r'' = -1 * 1 ≫ 7 = -1/128$

Example 3:

$$ w_0 = 127/128 → 01111111 $$
$$ w_1 = 1/128 → 00000001 $$
$$ w_0 * w_1 = 128/128 → saturation → 01111111 (= 127/128) $$

### Channel Data Formats

#### HWC

All internal data are stored in HWC format, 4 channels per 32-bit word. Assuming 3-color (or 3-channel) input, one byte will be unused. Example:

![0BGR 0BGR 0 BGR 0BGR...](docs/HWC.png)

#### CHW

The input layer can also use the CHW format (sequence of channels), for example:

![RRRRRR...GGGGGG...BBBBBB...](docs/CHW.png)

### CHW Data Format and Consequences for Weight Memory Layout

When using the CHW data format, only one of the four processors sharing the data memory instance can be used. The next channel needs to use a processor connected to a different data memory instance, so that the machine can deliver one byte per clock cycle to each enabled processor.

Because of the fact that a processor has its own dedicated weight memory, this will introduce “gaps” in the weight memory map, as shown in the following illustration:

![Kernel Memory Gaps](docs/KernelMemoryGaps.png)


### Active Processors and Layers

For each layer, a set of active processors must be specified. The number of active processors must be the same as the number of input channels for the layer, and the input data for that layer must be located in data memory instances accessible to the selected processors.

It is possible to specify a relative offset into the data memory instance that applies to all processors. _Example:_ Assuming HWC data format, specifying the offset as 8192 bytes will cause processors 0-3 to read their input from the second half of data memory 0, processors 4-7 will read from the second half of data memory instance 1, etc.

For most simple networks with limited data sizes, it is easiest to ping-pong between the first and second halves of the data memories - specify the data offset as 0 for the first layer, 0x2000 for the second layer, 0 for the third layer, etc. This strategy avoids overlapping inputs and outputs when a given processor is used in two consecutive layers.

Even though it is supported by the accelerator, the Network Generator will not be able to check for inadvertent overwriting of unprocessed input data by newly generated output data when overlapping data or streaming data. Use the `--overlap-data` command line switch to disable these checks, and to allow overlapped data.

### Layers and Weight Memory

For each layer, the weight memory start column is automatically configured by the Network Loader. The start column must be a multiple of 4, and the value applies to all processors.

The following example shows the weight memory layout for two layers. The first layer (L0) has 7 inputs and 9 outputs, and the second layer (L1) has 9 inputs and 2 outputs.

![Layers and Weight Memory](docs/KernelMemoryLayers.png)


### Weight Storage Example

The file `ai84net.xlsx` contains an example for a single-channel CHW input using the `AI85Net5` network (this example also supports up to four channels in HWC).

*Note*: As described above, multiple CHW channels must be loaded into separate memory instances. When using a large number of channels, this can cause “holes” in the processor map, which in turn can cause subsequent layers’ kernels to require padding.

The Network Loader prints a kernel map that shows the kernel arrangement based on the provided network description. It will also flag cases where kernel or bias memories are exceeded.

### Example: `Conv2D`

The following picture shows an example of a `Conv2d` with 1×1 kernels, 5 input channels, 2 output channels and data size of 2×2. The inputs are shown on the left, and the outputs on the right, and the kernels are shown lined up with the associated inputs --- the number of kernel rows matches the number of input channels, and the number kernel columns matches the number of output channels. The lower half of the picture shows how the data is arranged in memory when HWC data is used for both input and output.

![Conv2Dk1x1](docs/Conv2Dk1x1.png)

### Limitations of AI84 Networks

The AI84 hardware does not support arbitrary network parameters. Specifically,
* Dilation, groups, element-wise addition, and batch normalization are not supported.
* `Conv2d`:
  * Input data must be square (i.e., rows == columns).
  * Kernel sizes must be 3×3.
  * Padding can be 0, 1, or 2.
  * Stride is fixed to 1.
* `Conv1d`:
  * Input data lengths must be a multiple of 3.
  * Kernel size must be 9.
  * Padding can be 0, 3, or 6.
  * Stride is fixed to 3.
* The only supported activation function is `ReLU`.
* Pooling:
  * Pooling is only supported for `Conv2d`.
  * Pooling is always combined with a convolution. Both max pooling and average pooling are available.
  * Pooling does not support padding.
  * Pooling strides can be 1, 2, or 4.
  * On AI84, three pooling sizes are supported:
    - 2×2 with stride 1
    - 2×2 with stride 2
    - and, for the last layer, 4×4 stride 4 using a custom unload function.
  * Pooling must cleanly divide the input data width and height. For example, a 2×2 pool on 17×17 data is not supported.
  * Average pooling does not support more than 2048 bits in the accumulator. This translates to a 4×4 pooling window if activation was used on the prior layer, 3×3 otherwise. Additionally, average pooling is currently implemented as a `floor()` operation. Since there is also a quantization step at the output of the average pooling, it may not perform as intended (for example, a 2×2 `AvgPool2d` of `[[0, 0], [0, 3]]` will return `0`).
  * Pooling window sizes must be even numbers, and have equal H and W dimensions.
* The number of input or output channels must not exceed 64.
* The number of layers must not exceed 32.
* The maximum dimension (number of rows or columns) for input or output data is 256.
* Overall weight storage is limited to 64*128 3×3 kernels. However, weights must be arranged in a certain order, see above.
* The hardware supports 1D and 2D convolution layers. For convenience, a single final fully connected layer with 8-bit inputs/weights/bias, and 16-bit output is supported in software, as well as a software `SoftMax` operator.
* Since the internal network format is HWC in groups of four channels, output concatenation only works properly when all components of the concatenation other than the last have multiples of four channels. _Output concatenation is not yet supported in the `ai84.py` primitives._
* It is recommended to not use bias on the AI84 convolutions when using post-training quantization as described in this document. The reason is that the bias is not shifted by the same amount as the weights. This could be mitigated using a more involved training procedure, but since bias values do not significantly improve performance in the example networks, and since this will be corrected for AI85, the recommendation is to not use bias values for now.

With the exception of weight storage, and bias use, most of these limitations will be flagged when using the network primitives from `ai84.py`, see [Model Training and Quantization](#AI84-Model-Training-and-Quantization).

### Limitations of AI85 Networks

The AI85 hardware does not support arbitrary network parameters. Specifically,
* Dilation, groups, and batch normalization are not supported.
* `Conv2d`:
  * Kernel sizes must be 1×1 or 3×3.
  * Padding can be 0, 1, or 2.
  * Stride is fixed to 1.
* `Conv1d`:
  * Kernel sizes must be 1 through 9.
  * Padding can be 0, 1, or 2.
  * Stride is fixed to 1.
* A programmable layer-specific shift operator is available at the output of a convolution.
* The only supported activation function is `ReLU`.
* Pooling:
  * Both max pooling and average pooling are available, with or without convolution.
  
  * Pooling does not support padding.
  
  * Pooling strides can be 1 through 16.
  
  * Average pooling is implemented both using `floor()`and using rounding (half towards positive infinity). Use the `—avg-pool-rounding` switch to turn on rounding in the Network Generator.
  
    Example:
  
    * _floor:_ Since there is a quantization step at the output of the average pooling, a 2×2 `AvgPool2d` of `[[0, 0], [0, 3]]` will return $\lfloor \frac{3}{4} \rfloor = 0$.
    * _rounding:_ 2×2 `AvgPool2d` of `[[0, 0], [0, 3]]` will return $\lfloor \frac{3}{4} \rceil = 1$.
* The number of input or output channels must not exceed 512.
* The number of layers must not exceed 32.
* The maximum dimension (number of rows or columns) for input or output data is 256.
  
  * When using data greater than 90×91, `streaming` mode must be used.
* Overall weight storage is limited to 64*768 3×3 8-bit kernels (and proportionally more when using smaller weights, or smaller kernels). However, weights must be arranged in a certain order, see above.
* The hardware supports 1D and 2D convolution layers, element-wise addition, subtraction, binary OR, binary XOR as well as fully connected layers (`Linear`) (implemented using 1×1 convolutions on 1×1 data):
  * The maximum number of input neurons is 512, and the maximum number of output neurons is 512 (8 each per processor used).
  *  `Flatten` functionality is available to convert 2D input data for use by fully connected layers.
  *  Element-wise operators support from 2 up to 16 inputs.
  *  Element-wise operators can be chained in-flight with pooling and 2D convolution (where the order of pooling and element-wise operations can be swapped).
  * For convenience, a `SoftMax` operator is supported in software.
* Since the internal network format is HWC in groups of four channels, output concatenation only works properly when all components of the concatenation other than the last have multiples of four channels.

### Fully Connected (Linear) Layers

On AI85/AI85, m×n fully connected layers can be realized in hardware by “flattening” 2D input data into m channels of 1×1 input data. The hardware will produce n channels of 1×1 output data. When chaining multiple fully connected layers, the flattening step is omitted. The following picture shows 2D data, the equivalent flattened 1D data, and the output.

![MLP](docs/MLP.png)

---

## Model Training and Quantization

The `ai84net.py` file contains models that fit into AI84’s weight memory. These models rely on the AI84 hardware operators that are defined in `ai84.py`.

To train the FP32 model for FashionMIST, run `go_fashionmnist.sh` in the `ai8x-training` project. This script will place checkpoint files into the log directory. Training makes use of the Distiller framework, but the `train.py` software has been modified slightly to improve it and add some AI84 specifics.

Training progress can be observed by starting TensorBoard and pointing a web browser to its local server.

```shell
(ai8x-training) $ tensorboard --logdir='./logs'
TensorBoard 1.14.0 at http://127.0.0.1:6006/ (Press CTRL+C to quit)
```


### Quantization

There are two main approaches to quantization — quantization-aware training and post-training quantization.

Since performance for 8-bit weights is decent enough, _naive post-training quantization_ is used in the `ai84ize.py` software. While several approaches are implemented, a simple fixed scale factor is used based on experimental results. The approach requires the clamping operators implemented in `ai84.py`.

The software quantizes an existing PyTorch checkpoint file and writes out a new PyTorch checkpoint file that can then be used to evaluate the quality of the quantized network, using the same PyTorch framework used for training. The same new checkpoint file will also be used to feed the [Network Loader](#Network-Loader).

Copy the weight files into the `trained/` folder of the `ai8x-synthesis` project.

Example:

```shell
(ai8x-synthesis) $ ./ai84ize.py logs/path-to-checkpoint/checkpoint.pth.tar ../ai8x-synthesis/trained/ai84-fashionmnist.pth.tar -v
```

To evaluate the quantized network:

```shell
(ai8x-synthesis) $ ./evaluate_fashionmnist.sh
```

### Alternative Quantization Approaches

Post-training quantization can be improved using more sophisticated methods. For example, see
https://github.com/pytorch/glow/blob/master/docs/Quantization.md,
https://github.com/ARM-software/ML-examples/tree/master/cmsisnn-cifar10,
https://github.com/ARM-software/ML-KWS-for-MCU/blob/master/Deployment/Quant_guide.md,
or Distiller’s approach (installed with this software).

Further, a quantized network can be refined using post-quantization training (see Distiller).

The software also includes an `AI84RangeLinear.py` training quantizer that plugs into the Distiller framework for quantization-aware training. However, it needs work as its performance is not good enough yet and the Distiller source needs to be patched to enable it (add `from range_linear_ai84 import QuantAwareTrainRangeLinearQuantizerAI84` to `distiller/config.py` and remove `False and` from `if False and args.ai84` in `train.py`).

Note that AI84 does not have a configurable per-layer output shift. The addition of this shift value will allow easier quantization on AI85, since fractional bits can be used if weights do not span the full 8-bit range (most read-made quantization approaches require a weight scale or output shift).

In all cases, ensure that the quantizer writes out a checkpoint file that the Network Loader can read.

### Adding Datasets and New Networks to the Training Process

The following steps are needed to add new data formats and networks:
1. Develop a data loader in PyTorch, see https://pytorch.org/tutorials/beginner/data_loading_tutorial.html.
2. Implement a new network model (see `ai84net.py` for an example).
3. Add data loader and network model to `train.py`.
4. Train the new network with the new data. See `go_cifar.sh` for a command line example.

---

## Network Loader

_The network loader currently depends on PyTorch and Nervana’s Distiller. This requirement will be removed in the long term._

The network loader creates C code that programs the AI8X (for embedded execution, or RTL simulation, or CMSIS NN comparison). Additionally, the generated code contains sample input data and the expected output for the sample, as well as code that verifies the expected output.

The `cnn-gen.py` program needs two inputs:
1. A quantized checkpoint file, generated by the AI84 model quantization program `ai84ize.py`.
2. A YAML description of the network.

An example network description for the ai84net5 architecture and FashionMNIST is shown below:

```yaml
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
```

To generate an embedded AI84 demo in the `demos/FashionMNIST/` folder, use the following command line:

```shell
$ ./cnn-gen.py --verbose -L --top-level cnn --test-dir demos --prefix FashionMNIST --checkpoint-file trained/ai84-mnist.pth.tar --config-file networks/fashionmnist-chw.yaml --fc-layer --embedded-code
```

Running this command will combine the network described above with a fully connected software classification layer. The generated code will include all loading, unloading, and configuration steps.

To generate an RTL simulation for the same network and sample data in the directory `tests/fmnist-....` (where .... is an autogenerated string based on the network topology), use:

```shell
$ ./cnn-gen.py --verbose --autogen rtlsim --top-level cnn -L --test-dir rtlsim --prefix fmnist --checkpoint-file trained/ai84-fashionmnist.pth.tar --config-file networks/fashionmnist-chw.yaml
```

### Network Loader Configuration Language

Network descriptions are written in YAML (see https://en.wikipedia.org/wiki/YAML). There are two sections in each file --- global statements and a sequence of layer descriptions.

#### Global Configuration

##### `arch` (Mandatory)

`arch` specifies the network architecture, for example `ai84net5`. This key is matched against the architecture embedded in the checkpoint file.

##### `bias` (Optional, Test Only)

`bias` is only used for test data.

##### `dataset` (Mandatory)

`dataset` configures the data set for the network. This determines the input data size and dimensions as well as the number of input channels.

Data sets are for example `mnist`, `fashionmnist`, and `cifar-10`.

##### `output_map` (Optional)

The global `output_map`, if specified, overrides the memory instances where the last layer outputs its results. If not specified, this will be either the `output_processors` specified for the last layer, or, if that key does not exist, default to the number of processors needed for the output channels, starting at 0.

Example:
	`output_map: 0x0000000000000ff0`

##### `layers` (Mandatory)

`layers` is a list that defines the per-layer description.

#### Per-Layer Configuration

Each layer in the `layers` list describes the layer’s processors, convolution type, activation, pooling, weight and output sizes, data input format, data memory offsets, and its processing sequence. Several examples are located in the `networks/` and `tests/` folders.

##### `sequence` (Optional)

This key allows overriding of the processing sequence. The default is `0` for the first layer, or the previous layer’s sequence + 1 for other layers.

`sequence` numbers may have gaps. The software will sort layers by their numeric value, with the lowest value first.

##### `processors` (Mandatory)

`processors` specifies which processors will handle the input data. The processor map must match the number of input channels, and the input data format. For example, in CHW format, processors must be attached to different data memory instances.

Example:
	 `processors: 0x0000000000000111`

##### `output_processors` (Optional)

`output_processors` specifies which data memory instances and 32-bit word offsets to use for the layer’s output data. When not specified, this key defaults to the next layer’s `processors`, or, for the last layer, to the global `output_map`.

##### `out_offset` (Optional)

`out_offset` specifies the relative offset inside the data memory instance where the output data should be written to. When not specified, `out_offset` defaults to `0`.

Example:
	 `out_offset: 0x2000`

##### `in_offset` (Optional)

`in_offset` specifies the offset into the data memory instances where the input data should be loaded from. When not specified, this key defaults to the previous layer’s `out_offset`, or `0` for the first layer.

Example:
	 `in_offset: 0x2000`

##### `output_width` (Optional)

On AI84, this value (if specified) has to be `8`.

On AI85, when __not__ using an `activation`, the last layer can output `32` bits of unclipped data in Q25.7 format. The default is `8` bits.

Example:
	`output_width: 32`

##### `data_format` (Optional)

When specified for the first layer only, `data_format` can be either `chw`/`big` or `hwc`/`little`. The default is `hwc`. Note that the data format interacts with `processors`. 

##### `operation` (Optional)

This key (which can also be specified using `op`, `operator`, or `convolution`) selects a layer’s main operation after the optional input pooling.
The default is `Conv2d`. AI84 only supports `Conv1d` and `Conv2d`.

| Operation                 | Description                                                  |
| :------------------------ | :----------------------------------------------------------- |
| `Conv1d`                  | 1D convolution over an input composed of several input planes |
| `Conv2d` (default)        | 2D convolution over an input composed of several input planes |
| `None` or `Passthrough`   | No operation                                                 |
| `Linear` or `FC` or `MLP` | Linear transformation to the incoming data                   |
| `Add`                     | Element-wise addition                                        |
| `Sub`                     | Element-wise subtraction                                     |
| `Xor`                     | Element-wise binary XOR                                      |
| `Or`                      | Element-wise binary OR                                      |

Element-wise operations default to two operands. This can be changed using the `operands` key.

##### `eltwise` (Optional)

Element-wise operations can also be added “in-flight” to `Conv2d`. In this case, the element-wise operation is specified using the `eltwise` key.

Example:
  `eltwise: add`

##### `pool_first` (Optional)

When using both pooling and element-wise operations, pooling is performed first by default. Optionally, the element-wise operation can be performed before the pooling operation by setting `pool_first` to `False`.

Example:
	`pool_first: false`

##### `operands` (Optional)

For any element-wise `operation`, this key configures the number of operands from `2` to `16` inclusive. The default is `2`.

Example:
	`operation: add`
	`operands: 4`

##### `activate` (Optional)

This key describes whether to activate the layer output (the default is to not activate). When specified, this key must be `ReLU`.

##### `quantization` (Optional)

On AI84, this key must always be `8` (the default value).

On AI85, this key describes the width of the weight memory in bits and can be `1`, `2`, `4`, or `8` (`8` is the default).

Example:
	`quantization: 4`

##### `kernel_size` (Optional)

2D convolutions:

​	On AI84, this key must always be `3x3` (the default).
​	On AI85, `1x1` is also permitted.

1D convolutions:

​	On AI84, this key must always be `9` (the default).
​	On AI85, `1` through `9` are permitted.

Example:
	`kernel_size: 1x1`

##### `stride` (Optional)

2D convolutions:

This key must always be `1`.

1D convolutions:

On AI84, this key must be set to `3`. On AI85, this key must be `1`.

##### `pad` (Optional)

`pad` sets the padding for the convolution.

* For `Conv2d`, this value can be `0`, `1` (the default), or `2`.
* For `Conv1d`, the value can be `0`, `3` (the default), or `6` on AI84 or `0`, `1`, `2` on AI85.
* For `Passthrough`, this value must be `0` (the default).

##### `max_pool` (Optional)

When specified, performs a `MaxPool` before the convolution. On AI84, `max_pool` is only supported for 2D convolutions. The pooling size can specified as an integer (when the value is identical for both dimensions, or for 1D convolutions), or as two values in order `[H, W]`.

Example:
	 `max_pool: 2`

##### `avg_pool` (Optional)

When specified, performs an `AvgPool` before the convolution. On AI84, `avg_pool` is only supported for 2D convolutions. The pooling size can specified as an integer (when the value is identical for both dimensions, or for 1D convolutions), or as two values in order `[H, W]`.

Example:
	 `avg_pool: 2`

##### `pool_stride` (Optional)

When performing a pooling operation, this key describes the pool stride. The pooling stride can be specified as an integer (when the value is identical for both dimensions, or for 1D convolutions), or as two values in order `[H, W]`. The default is `1` or `[1, 1]`.

Example:
	 `pool_stride: 2`

##### `in_dim` (Optional)

`in_dim` specifies the dimensions of the input data. This is usually automatically computed; however, when merging layers or flattening data, this key allows overriding of the dimensions.

Example:
  `in_dim: [64, 64]` 

##### `streaming` (Optional)

`streaming` specifies that the layer is using streaming mode. this is necessary when the input data dimensions exceed the available data memory. When enabling `streaming`, all prior layers have to enable `streaming` as well. `streaming` can be enabled for up to 8 layers.

Example:
	`streaming: true`

##### `flatten` (Optional)

`flatten` specifies that 2D input data should be transformed to 1D data for use by a `Linear` layer.

Example:
	`flatten: true`


### Adding Datasets to the Network Loader

Adding new datasets to the Network Loader is implemented as follows:
1. Provide network model, its YAML description and quantized weights.
2. Provide a sample input -- add `sample_dset.npy` for the dataset named `dset` to the `tests` directory. This file is generated by saving a sample using `numpy.save()`. See `sampledata.py` for more information.

### Starting an Inference, Waiting for Completion, Multiple Inferences in Sequence

An inference is started by loading registers and kernels, loading the input, and enabling processing.  This code is automatically generated—see the `cnn_load()`, `load_kernels()`, and `load_input()` functions. The sample data can be used as a self-checking feature on device power-up since the output for the sample data is known.

The AI8X accelerator can generate an interrupt on completion, and it will set a status bit (see `cnn_wait()`). The resulting data can now be unloaded from the accelerator (code for this is also auto-generated in `unload()`).

To run another inference, ensure all groups are disabled (stopping the state machine, as shown in `cnn_load()`). Next, load the new input data and start processing.

---

### CMSIS5 NN Emulation

The Network Loader tool can also create code that executes the same exact network using Arm’s CMSISv5 Neural Networking library, optimized for Arm’s Microcontroller DSP (reportedly 4.6× faster than executing on the CPU), see https://developer.arm.com/solutions/machine-learning-on-arm/developer-material/how-to-guides/converting-a-neural-network-for-arm-cortex-m-with-cmsis-nn and https://github.com/ARM-software/CMSIS_5 for the source code. A version of this repository is automatically checked out as part of the `ai8x-synthesis` project.

The results of the generated code have been verified to match AI84 exactly and may be used to demonstrate the efficacy of the custom CNN accelerator.
Note there are minor rounding errors in the CMSIS average pooling code when enabling SIMD
(`-DARM_MATH_DSP`). If there are hard faults on the device, try limiting compiler optimizations to `-O1`.

The `Device/` folder contains a sample Makefile, and a custom fully connected layer in the file `arm_fully_connected_q7_q8p7_opt.c` that returns Q8.7 fixed-point outputs, and a custom `arm_softmax_q8p7_q15.c` which is aware of the fixed-point input (both of these files are also used for the software classification layer on AI84/AI85). Additionally, a number of files are provided that provide non-square pooling support, and activation support for more than 64 KB of data (these files are not needed for AI84/AI85 hardware).
The `tornadocnn.h` header file is included which helps both embedded examples as well as CMSIS NN code.

For example, the following command would generate code that performs a CIFAR-10 inference from the `trained/ai84-cifar10.pth.tar` checkpoint file and the `cifar10-hwc.yaml` network description file:

```shell
(ai8x-synthesis) $ ./cnn-gen.py --top-level cnn --test-dir demos --prefix CIFAR-10-Arm --checkpoint-file trained/ai84-cifar10.pth.tar --config-file networks/cifar10-hwc.yaml --fc-layer --embedded-code --cmsis-software-nn
```

When compiling the CMSIS code, you may have to disable compiler optimizations.

*Note*: The CMSIS code generator does not currently support the extended features of AI85 and AI86.

---

## AI84 SDK

Use SVN to check out the AI84 SDK from https://svn.maxim-ic.com/svn/mcbusw/Hardware/Micro/AI84/.

```shell
$ svn co https://svn.maxim-ic.com/svn/mcbusw/Hardware/Micro/AI84/ AI84SDK
```

Additionally, the Arm embedded compiler is required, it is available from https://developer.arm.com/tools-and-software/open-source-software/developer-tools/gnu-toolchain/gnu-rm/downloads.

In order for the debugger to work, the OpenOCD `max32xxx` branch from https://github.com/MaximIntegratedMicros/openocd.git must be installed, and an ai84.cfg file must be installed in `/usr/local/share/openocd/scripts/target/` (or `C:\Maxim\Toolchain\share\openocd\scripts\target\` on Windows). A copy of the file is contained in the `hardware` folder of the `ai8x-synthesis` project.

Windows binaries for OpenOCD can be installed via https://www.maximintegrated.com/en/design/software-description.html/swpart=SFW0001500A. This download also includes a version of the Arm compilers.

To compile and install this OpenOCD version on macOS, use:

```shell
$ git clone https://github.com/MaximIntegratedMicros/openocd.git
$ cd openocd
$ git checkout max32xxx
$ MAKEINFO=true LIBTOOL=/usr/local/bin/glibtool ./configure --disable-dependency-tracking --enable-dummy --enable-buspirate --enable-jtag_vpi --enable-remote-bitbang --enable-legacy-ft2232_libftdi
$ make
$ make install
```

Additional SDK instructions can be found in a separate document,
https://svn.maxim-ic.com/svn/mcbusw/Hardware/Micro/AI84/docs/trunk/AI84%20Test%20Board%20Setup%20Instructions.docx.

---

## AI85/AI86 Changes

The `ai84ize.py` quantization tool and the `cnn-gen.py` network generator both have an `--ai85` command line argument that enables code generation for the AI85 and AI86.

The `--ai85` option enables:
* Bias shift << 7.
* Per-layer support for 1, 2 and 4-bit weight sizes in addition to 8-bit weights (this is supported using the `quantization` keyword in the configuration file, and the configuration file can also be read by the quantization tool).
* A shift on the output of the convolution that allows for better use of the entire range of weight bits.
* Support for many more pooling sizes and pooling strides, and larger limits for average pooling.
* Support for pooling without convolution (passthrough mode), and optional rounding for average pooling.
* 1D convolutions.
* 1×1 kernels for 2D convolutions.
* Data “flattening”, allowing the use of 1×1 kernels to emulate fully connected layers.
* In-flight element-wise addition, subtraction, and binary or/xor.
* Support for more weight memory, and more input and output channels.
* Support for non-square data and non-square pooling kernels.
* Support for 32-bit Q25.7 data output for last layer when not using ReLU.
* Support for streaming mode with FIFOs to allow for larger data sizes.

---

## Updating the Project

To pull the latest code, in either project use:

```shell
$ git pull
$ git submodule update --init
```

## Contributing Code

### Linting

Both projects are set up for `flake8`, `pylint`, and `mypy`. The line width is related to 100 (instead of the default of 80), and the number of lines per module was increased.
Code should not generate any warnings in any of the tools (some of the components in the `ai8x-training` project will create warnings as they are based on third-party code).

`flake8`, `pylint` and `mypy` need to be installed into the virtual environment:

```shell
(ai8x-synthesis) $ pip3 install flake8 pylint mypy
```

### Submitting Changes

Do not try to push any changes into the master branch. Instead, create a local “feature” branch. The easiest way to do this is using a graphical client such as [Fork](#Recommended-Software). The command line equivalent is:

```shell
$ git checkout -b my-new-feature
$ git status
On branch my-new-feature
...
```

Commit changes and create a description of the changes:

```shell
$ git commit
```

Check the address for the Maxim server:

```shell
$ git remote -v
origin	https://first.last@gerrit.maxim-ic.com:8443/ai8x-synthesis (fetch)
origin	https://first.last@gerrit.maxim-ic.com:8443/ai8x-synthesis (push)
```

Install a commit hook:

```shell
$ GITDIR=$(git rev-parse --git-dir)
$ scp -p -P 29418 first.last@gerrit.maxim-ic.com:hooks/commit-msg ${GITDIR}/hooks/
```

And push the changes to Maxim’s Gerrit server (do not change “master” to anything else even though the local branch is called “my-new-feature”):

```shell
$ git push https://first.last@gerrit.maxim-ic.com:8443/ai8x-synthesis HEAD:refs/for/master
...
remote:
To URL
* [new branch] my-new-feature -> refs/for/master
```

Open the URL in a web browser and request a review for the change list.

---
