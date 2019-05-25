# AI84 Model Training and Quantization

## Installation

### Prerequisites

Install CUDA10 and CUDNN:
https://developer.nvidia.com/cuda-downloads
https://developer.nvidia.com/cudnn

### Project

*The software in this project requires Python 3.6.*

To manage Python versions, use pyenv (https://github.com/pyenv/pyenv).

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


## Usage

The `ai84net.py` file contains models that fit into AI84's weight memory.

To train the FP32 model for FashionMIST, run `go.sh`. This script will place checkpoint files
into the log directory. To quantize the checkpoint file after training:

    ./ai84ize logs/path-to-checkpoint/checkpoint.pth.tar ai84.pth.tar -v

Now, evaluate the quantized network:

    ./evaluate.sh
