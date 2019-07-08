On Windows, use Miniconda/Anaconda as package manager:
- Download the Miniconda/Anaconda installer for Windows
    (https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe)
- Click on .exe file and follow the instructions.

To create a virtual environment with conda (venv named ai8x-training):

    conda create -n ai8x-training python=3.6.5

Then,

    $ mkdir ai8x-training
    $ cd ai8x-training
    $ activate ai8x-training
    (ai8x-training) $ conda install pip
    (ai8x-training) $ conda install setuptools
    (ai8x-training) $ conda install numpy
    (ai8x-training) $ conda install pyyaml tabulate future six typing
    (ai8x-training) $ conda install scipy

For installing Torch on Windows; you can either choose the cpu version,

    (ai8x-training) $ conda install pytorch-cpu torchvision-cpu -c pytorch

or else the gpu version,

    (ai8x-training) $ conda install pytorch torchvision cudatoolkit=10.0 -c pytorch

For downloading tensorflow on Windows, you can use pip,

    (ai8x-training) $ pip install tensorflow

The rest of the document is the same for Windows, only differences are
- pip of conda enviroment should be used instead of the rest pip3 install commands. An example: 
    ```
    C:\Users\username\AppData\Local\Continuum\anaconda3\envs\ai8x-training\Scripts\pip install package_name
    ```
- Commands inside shell scripts should be called one by one or batch files can be created.



