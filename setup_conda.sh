#!/bin/bash

echo "install conda"
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh

source ~/miniconda3/bin/activate
export PATH=~/miniconda3/bin:$PATH

conda init --all 


echo "create rm env"
conda create -n rm python=3.10 
conda activate rm
echo "install stuff"
pip install -r requirements.txt
echo "done"

echo "adding .local to path (for accelerate and some other stuff)"
export PATH=~/.local/bin:$PATH

