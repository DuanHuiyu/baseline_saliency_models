# Baseline deep saliency models

## pytorch environment
create environment
```
conda create -n saliency python=3.8 pip
conda activate saliency
conda install visdom dominate -c conda-forge
```
CUDA 11.3
```
conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
```
CUDA 10.2
``` 
conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=10.2 -c pytorch
```
install requirement
```
pip install -r requirements.txt
```