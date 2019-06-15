# AI84 Model Training and Quantization
# AI84 Network Loader and RTL Simulation Generator

This software consists of two related projects:
1. AI84 Model Training and Quantization
2. AI84 Network Loader and RTL Simulation Generator

## Installation

### Prerequisites

When going beyond simple tests, model training requires hardware acceleration (the network loader
does not require CUDA).
Install CUDA10 and CUDNN:
https://developer.nvidia.com/cuda-downloads
https://developer.nvidia.com/cudnn

### Project

*The software in this project requires Python 3.6.5 or a later 3.6.x version.*

It is not necessary to install Python 3.6.5 system-wide, or to rely on the system-provided
Python. To manage Python versions, use pyenv (https://github.com/pyenv/pyenv).

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

Next, install Python 3.6.5:

    $ pyenv install 3.6.5

Then,

    $ mkdir ai84
    $ cd ai84
    $ pyenv local 3.6.5
    $ python3 -m venv .
    $ source bin/activate
    (ai84) $ pip3 install -U pip setuptools
    (ai84) $ pip3 install numpy
    (ai84) $ pip3 install pyyaml tabulate future six typing
    (ai84) $ pip3 install scipy

For macOS:
    
    (ai84) $ pip3 install torch

For CUDA 10 on Linux (see https://pytorch.org/get-started/locally/):

    (ai84) $ pip3 install https://download.pytorch.org/whl/cu100/torch-1.1.0-cp36-cp36m-linux_x86_64.whl

On all systems, install Tensorflow:

    (ai84) $ pip3 install tensorflow

*NOTE*: On x86 systems, the pre-built Tensorflow wheels require AVX (`sysctl -n machdep.cpu.features`
to find out). Running an unsupported Tensorflow wheel that requires AVX instructions results
in `Illegal instruction: 4` on startup.

If the CPU does not support AVX, or to enable suport for AVX2, or CUDA, or AMD64, build
Tensorflow locally. This requires Java 8 and Bazel, and takes over two hours.

Building Tensorflow requires Bazel 0.21.0 (newer or much older versions do not work). See
https://docs.bazel.build/versions/master/install-compile-source.html#build-bazel-using-bazel for
build instructions.

Once Bazel is installed, compile and install Tensorflow. On Jetson TX1, disable S3. See 
https://github.com/peterlee0127/tensorflow-nvJetson/blob/master/patch/tensorflow1.12.patch
The removal of "-mfpu=neon" does not seem to be necessary.

    (ai84) $ pip3 install keras_preprocessing
    (ai84) $ git clone https://github.com/tensorflow/tensorflow 
    (ai84) $ cd tensorflow
    (ai84) $ git checkout tags/v1.13.1
    (ai84) $ ./configure
    (ai84) $ bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package
    (ai84) $ bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
    (ai84) $ pip3 install /tmp/tensorflow_pkg/tensorflow-1.13.1-cp36-cp36m-macosx_10_13_x86_64.whl 

To install torchvision (the wheels are frequently outdated):

    (ai84) $ git clone https://github.com/pytorch/vision.git
    (ai84) $ cd vision ; git checkout tags/v0.3.0 ; cd ..
    (ai84) $ pip3 install -e vision

To install Nervana's distiller:

    (ai84) $ git clone https://github.com/NervanaSystems/distiller.git
    (ai84) $ cp requirements.txt.distiller distiller/requirements.txt
    (ai84) $ pip3 install -e distiller

(macOS) Add to ~/.matplotlib/matplotrc:

    backend: TkAgg


## Usage: AI84 Model Training and Quantization

The `ai84net.py` file contains models that fit into AI84's weight memory. These models rely on
the AI84 hardware operators that are defined in `ai84.py`.

To train the FP32 model for FashionMIST, run `go_fashionmnist.sh`. This script will place
checkpoint files into the log directory. Training makes use of the Distiller framework, but
the `train.py` software has been modified slightly to improve it and add some AI84 specific
modifications.

### Quantization

There are two main approaches to quantization -- quantization aware training and post-training
quantization.

Since performance for 8-bit weights is decent enough, _naive post-training quantization_ is used
in the `ai84ize.py` software. While several approaches are implemented, a simple fixed scale
factor is used based on experimental results. The approach requires the clamping operators
implemented in `ai84.py` to be present.

The software quantizes an existing PyTorch checkpoint file and writes
out a new PyTorch checkpoint file that can then be used to evaluate the quality of the
quantized network, using the same PyTorch framework used for training.
The same new checkpoint file will also be used to feed the AI84 Network Loader.

Example:

    ./ai84ize.py logs/path-to-checkpoint/checkpoint.pth.tar trained/ai84-fashionmnist.pth.tar -v

To evaluate the quantized network:

    ./evaluate_fashionmnist.sh

### Alternative Quantization Approaches

Post-training quantization can be improved using more sophosticated methods. For example, see
https://github.com/ARM-software/ML-examples/tree/master/cmsisnn-cifar10,
https://github.com/ARM-software/ML-KWS-for-MCU/blob/master/Deployment/Quant_guide.md,
or Distiller's approach (installed with this software).

Further, a quantized network can be refined using post-quantization training (see Distiller).

The software also includes an `AI84RangeLinear` training quantizer that plugs into the Distiller
frameworkf for quantization aware training. However, it needs work as its performance is not
good enough yet.

Note that AI84 does not have a configurable per-layer output shift. The addition of this
shift value will allow easier quantization on AI85, since fractional bits can be used if weights
do not span the full 8-bit range.

In all cases, ensure that the quantizer writes out a checkpoint file that the AI84 Network Loader
can read.


## Limitations of AI84 Networks

The AI84 hardware does not support arbitrary network parameters. For example,
* Dilation, elementwise addition, and 1D convolutions are not supported.
* The `Conv2D` kernel size must be 3x3.
* `Conv2D` padding can be 0, 1, or 2.
* The only supported activation function is `ReLU`.
* Pooling is always combined with a convolution. Both `MaxPool2D` and `AvgPool2D` are available.
* Pooling does not support padding. Pooling must cleanly divide the input data width and height.
  For example, a 2x2 pool on 17x17 data is not supported.
* Average pooling does not support more than 2048 bits in the accumulator. This translates to
  a 4x4 pooling window if activation was used on the prior layer, 3x3 otherwise. Additionally,
  average pooling is currently implemented as a `floor()` operation. Since there is also a
  quantization step at the output of the average pooling, it may not perform as intended
  (for example, a 2x2 AvgPool of `[[0, 0], [0, 3]]` will return `0`).
* Pooling window sizes must be even numbers, and have equal H and W dimensions.
* The `Conv2D` stride is fixed to 1. However, the pooling stride can be 1, 2, or 4.
* The number of input or output channels must not exceed 64.
* The number of layers must not exceed 32.
* Overall weight storage is limited to 64*128 9x9 kernels. However, weights must be arranged
  in a certain order, see below.
* The hardware supports only 2D convolution layers. For convenience, a single final fully
  connected layer with 8-bit inputs/weights/bias, and 16-bit output is supported in software,
  as well as a software SoftMax operator.
* Since the internal network format is HWC in groups of four channels, output concatenation only
  works properly when all components of the concatenation other than the last have multiples
  of four channels. Output concatenation is not yet supported in the `ai84.py` primitives.
* It is recommended to not use bias on the AI84 convolutions when using post-training
  quantization as described in this document. The reason is that the bias is not shifted by the
  same amount as the weights. This could be mitigated using a more involved training procedure,
  but since bias values do not significantly improve performance in the example networks, and
  since this will be corrected for AI85, the string recommendation is to not use bias values
  for now.

With the exception of weight storage, and bias use, most of these limitations are flagged when
using the network primitives from `ai84.py`.


## Weight Storage

The file `ai84net.xlsx` contains an example for a single-channel CHW input (this example also
supports or up to four channels in HWC).

*NOTE*: Multiple CHW channels must be loaded into separate
memory instances. When using a large number of channels, this can cause 'holes' in the processor
map, which in turn can cause subsequent layers' kernels to require padding.

The AI84 Network Loader prints a kernel map that shows the kernel arrangement based on the provided
network description. It will also flag cases where kernel memory or bias memory is exceeded.


## Adding New Data and Networks

The following steps are needed to add new data formats and networks:
1. Develop a data loader in PyTorch.
2. Implement a new network model (see `ai84net.py` for an example).
3. Add data loader and network model to `train.py`.
4. Train the new network with the new data. See `go_cifar.sh` for a command line example.


## Usage: AI84 Network Loader

_The network loader currently depends on PyToch and Nervana's distiller. This requirement will be
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

    ./cnn-gen.py --verbose -L --top-level cnn --test-dir demos --prefix FashionMNIST --checkpoint-file trained/ai84-mnist.pth.tar --config-file fashionmnist-chw.yaml --fc-layer --embedded-code

In addition to the network described above, this invocation will also add a fully connected
classification layer in software. The generated code will include all loading, unloading, and
configuration steps.

To generate an RTL simulation for the same network and sample data in the directory
`tests/fmnist-....` (where .... is an autogenerated string based on the network topology), use:

    ./cnn-gen.py --verbose --autogen tests --top-level cnn -L --test-dir tests --prefix fmnist --checkpoint-file trained/ai84-fashionmnist.pth.tar --config-file fashionmnist-chw.yaml

Adding new datasets to the AI84 Network Loader is implemented by
1. Providing a network model, its YAML description and quantized weights.
2. Providing a sample input, and the expected output (modify `sampledata.py` as well as the
   `SUPPORTED_DATASETS` in `yamlcfg.py`).


## AI85

The `ai84ize.py` quantization tool and the `cnn-gen.py` network generator both have an 
`--ai85` command line argument that enables code generation for the AI85.

The `--ai85` option enables:
* Bias shift << 7.
* Per-layer support for 1, 2 and 4-bit weight sizes in addition to 8-bit weights.
* A scale factor on the output of Conv2D that allows for better use of the entire range of weight
  bits.
* Support for more memory.


## CMSIS5 NN Emulation

The AI84 Network Loader tool can _also_ create code that executes the same exact network using
Arm's CMSISv5 Neural Networking library, optimized for Arm's Microcontroller DSP (reportedly
4.6x faster than executing on the CPU), see
https://developer.arm.com/solutions/machine-learning-on-arm/developer-material/how-to-guides/converting-a-neural-network-for-arm-cortex-m-with-cmsis-nn
and https://github.com/ARM-software/CMSIS_5 for the source code.

The results of the generated code have been verified to match AI84 exactly and may be used to
demonstrate the efficacy of the custom CNN accelerator.

The `Device/` folder contains a sample Makefile, an edited `arm_math.h` header file (which allows
code execution on a host instead of a microcontroller), and a custom fully connected layer in the
file `arm_fully_connected_q7_q15_opt.c` that returns 16-bit outputs instead of 8-bit, and a custom
`arm_softmax_q7_q15.c` which is aware of the scaled input (both of these files are also used for
the software classification layer on AI84/AI85).
Additionally, a `tornadocnn.h` header file is included which helps for both embedded examples as
well as CMSIS NN code.

For example, the following command would generate code that performs a CIFAR-10 inference from the
`trained/ai84-cifar10.pth.tar` checkpoint file and the `cifar10-hwc.yaml` network description
file:

    ./cnn-gen.py --top-level cnn --test-dir demos --prefix CIFAR-10-Arm --checkpoint-file trained/ai84-cifar10.pth.tar --config-file cifar10-hwc.yaml --fc-layer --embedded-code --cmsis-software-nn


-----------------------
