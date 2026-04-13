# Orin Installation Guide For Co-SLAM

Co-SLAM Installation Guide: Orin with Jetpack 6.2 and cuda 12.6
Co-slam autors recommend a setup with a conda env with python=3.7 because it is compatible with the libriraies they need. Because on Orin python 3.7 is very problematic to use (due to unexisting wheels for this version), I have tested a new setup with python 3.10 for Orin, that can also be used on other platforms. The changes concern the versions of the requirements mainly.
For Orin, I have jetpack 6.2 and cuda 12.6, but the wheels for torch and torchvision I use are for jetpack 6.0 and cuda 12.4 or 12.2, since jetpack supports backward compatibility

## Environment Setup

```bash
# Create conda environment
conda create -n coslam python=3.10 
conda activate coslam

# Clone Co-SLAM (you can clone https://github.com/yhaddouda/Co-SLAM/tree/profiling and you will find the requirements.txt edited)
git clone https://github.com/HengyiWang/Co-SLAM.git 
cd Co-SLAM

## torch 2.3
wget https://nvidia.box.com/shared/static/zvultzsmd4iuheykxy17s4l2n91ylpl8.whl -O torch-2.3.0-cp310-cp310-linux_aarch64.whl
## torchvision
wget https://nvidia.box.com/shared/static/u0ziu01c0kyji4zz3gxam79181nebylf.whl -O torchvision-0.18.0a0+6043bc2-cp310-cp310-linux_aarch64.whl

pip install numpy ./torch-2.3.0-cp310-cp310-linux_aarch64.whl
pip install ./torchvision-0.18.0a0+6043bc2-cp310-cp310-linux_aarch64.whl 


# Install Co-SLAM dependencies
pip install -r requirements.txt
```
The changes are mathutils==3.3.0, typing_extensions==4.8.0, #git+https://github.com/facebookresearch/pytorch3d.git
#git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch  these two librairies should be commented and built from source (see below)

## tiny-cuda-nn Installation 

```bash
$ git clone --recursive https://github.com/nvlabs/tiny-cuda-nn
$ cd tiny-cuda-nn
tiny-cuda-nn$ cmake . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
tiny-cuda-nn$ cmake --build build --config RelWithDebInfo -j
then go to bindings/torch and do : 
$ python setup.py install
```

## Pytorch3d
```bash
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```


# Build marching cubes extension
```bash
cd external/NumpyMarchingCubes
python setup.py install
```
