# AI84 Model Training

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
          xz-utils tk-dev libffi-dev liblzma-dev
        $ curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash

Then, add to ~/.bash_profile or ~/.profile (as shown by the previous step):

        eval "$(pyenv init -)"
        eval "$(pyenv virtualenv-init -)"

Next, install Python 3.6.5:

        $ pyenv install 3.6.5

Then,

    mkdir ai84
    cd ai84
    pyenv local 3.6.5
    python3 -m venv .
    source bin/activate
    pip3 install -U pip setuptools
    pip3 install numpy scipy scikit-learn matplotlib pyts pandas future xlrd scikit-image psutil

For macOS:
    
    pip3 install torch

For CUDA 10 on Linux (see https://pytorch.org/get-started/locally/):

    pip3 install https://download.pytorch.org/whl/cu100/torch-1.1.0-cp36-cp36m-linux_x86_64.whl

On all systems:

    pip3 install progressbar2 tabulate tensorboardX tensorboard tensorflow python_utils

To install torchvision:

    git clone https://github.com/pytorch/vision.git
    pip3 install -e vision

To install Nervana's distiller:

    git clone https://github.com/NervanaSystems/distiller.git
    cp requirements.txt.distiller distiller/requirements.txt
    pip3 install -e distiller

(macOS) Add to ~/.matplotlib/matplotrc:

    backend: TkAgg

Fix lib/python3.6/site-packages/pyts/image/image.py, line 321 and add `tuple()`:

    MTF[tuple(np.meshgrid(list_values[i], list_values[j]))] = MTM[i, j]

