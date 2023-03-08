#!/bin/bash

#git　install
sudo apt-get -y install git-all

#pip3 install
sudo apt-get -y install python3-pip
# pip自体のアップデート
pip3 install -U pip

# This script will install pytorch, torchvision, torchtext and spacy on nano. 
# If you have any of these installed already on your machine, you can skip those.

sudo apt-get -y update
sudo apt-get -y upgrade
#Dependencies
sudo apt-get -y install python3-setuptools

#python3.9 install
wget https://www.python.org/ftp/python/3.9.2/Python-3.9.2.tar.xz
tar xJf Python-3.9.2.tar.xz
cd Python-3.9.2
./configure --enable-optimizations
make
sudo make install

sudo pip install --upgrade pip


#Installing PyTorch
#For latest PyTorch refer original Nvidia Jetson Nano thread - https://devtalk.nvidia.com/default/topic/1049071/jetson-nano/pytorch-for-jetson-nano/.
wget https://nvidia.box.com/shared/static/yr6sjswn25z7oankw8zy1roow9cy5ur1.whl -O torch-1.6.0rc2-cp36-cp36m-linux_aarch64.whl
sudo apt-get -y install python3-pip libopenblas-base
#sudo pip3 install Cython
#sudo pip3 install numpy torch-1.6.0-cp36-cp36m-linux_aarch64.whl
sudo pip3 install Cython numpy
sudo pip3 install torch-1.6.0rc2-cp36-cp36m-linux_aarch64.whl

pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

#Installing torchvision
#For latest torchvision refer original Nvidia Jetson Nano thread - https://devtalk.nvidia.com/default/topic/1049071/jetson-nano/pytorch-for-jetson-nano/.
sudo apt-get -y install build-essential libssl-dev libffi-dev python3-dev libbz2-dev
sudo apt-get -y install libopenblas-base libopenmpi-dev
sudo apt-get -y install libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev libavformat-dev libswscale-dev
#git clone --branch v0.7.0 https://github.com/pytorch/vision torchvision   # see below for version of torchvision to download
#cd torchvision
#export BUILD_VERSION=0.7.0
#sudo python setup.py install
git clone --branch v0.7.0 https://github.com/pytorch/vision torchvision
cd torchvision
export BUILD_VERSION=0.7.0
python3 setup.py install --user
cd ../  # attempting to load torchvision from build dir will result in import error

#Installing spaCy
#Installing dependency sentencepiece
sudo apt-get -y install cmake build-essential pkg-config libgoogle-perftools-dev
git clone https://github.com/google/sentencepiece.git
cd sentencepiece
mkdir build
cd build
cmake ..
make -j $(nproc)
sudo make install
sudo ldconfig -v
cd python
python3 setup.py build
sudo python3 setup.py install
cd ../../

git clone https://github.com/explosion/spaCy
cd spaCy/
export PYTHONPATH=`pwd`
export BLIS_ARCH=generic
sudo pip3 install -r requirements.txt
sudo python3 setup.py build_ext --inplace
sudo python3 setup.py install
python3 -m spacy download en_core_web_sm
cd ../

#Installing torchtext
git clone https://github.com/pytorch/text.git
cd text
sudo pip3 install -r requirements.txt
sudo python3 setup.py install
cd ../

pyenv install 3.9.2
pyenv local 3.9.2
pyenv global 3.9.2

#モジュールインストール
python3 -m pip install numpy
python3 -m pip install typing_extensions
python3 -m pip install requests
python3 -m pip install tqdm

#cudaインストール
sudo apt purge -y nvidia-*
sudo apt purge -y cuda-*
sudo apt autoremove -y --purge

#wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
#sudo dpkg -i cuda-keyring_1.0-1_all.deb
#sudo apt-get -y update
#sudo apt install -y cuda-11-7


echo "done installing PyTorch, torchvision, spaCy, torchtext"
