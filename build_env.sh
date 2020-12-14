
#!/bin/bash

# Create the virtual environment with Python3 as default
virtualenv -p $(which python3) ml5g_bs_neu_venv

# Activate the just created virtual environment
source ./ml5g_bs_neu_venv/bin/activate

# Install Python's packages in the virtual environment
pip install tensorflow-gpu sklearn matplotlib tqdm

# Tell Python the location of CUDA and cuDNN libraries
# see: https://www.tensorflow.org/install/source
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64