# GAN
Tensorflow implement of Advanced GANs 

# Requirements
Python 3.5
TensorFlow 1.12.0
Opencv-python 3.4.5.20

# Usage
Each GAN can be trained by run "train_XXX.py" and evaled by "eval_XXX.py" directly.

# DTN(Domain Transfer Net)
transfer SVHN data to MNIST data 
![](https://github.com/qzq2514/GAN/blob/master/GANs_Advanced/pictures/DTN_result.jpg)

# DiscoGAN  
|   Edge2Shoes   | Shoes2Edge|  
|:------------:|:-------------------:|  
| ![](https://github.com/qzq2514/GAN/blob/master/GANs_Advanced/pictures/DiscoGAN_A2B.jpg)    |![](https://github.com/qzq2514/GAN/blob/master/GANs_Advanced/pictures/DiscoGAN_B2A.jpg)        | 

# DualGAN  
|   Edge2Shoes   | Shoes2Edge|  
|:------------:|:-------------------:|  
| ![](https://github.com/qzq2514/GAN/blob/master/GANs_Advanced/pictures/DualGAN_A2B.jpg)    |![](https://github.com/qzq2514/GAN/blob/master/GANs_Advanced/pictures/DualGAN_B2A.jpg)        | 

# CycleGAN  
|   Edge2Shoes   | Shoes2Edge|  
|:------------:|:-------------------:|  
| ![](https://github.com/qzq2514/GAN/blob/master/GANs_Advanced/pictures/CycleGAN_A2B.jpg)    |![](https://github.com/qzq2514/GAN/blob/master/GANs_Advanced/pictures/CycleGAN_B2A.jpg)        | 

# Pix2Pix         
|   Edge2Shoes   | Shoes2Edge|  
|:------------:|:-------------------:|  
| ![](https://github.com/qzq2514/GAN/blob/master/GANs_Advanced/pictures/Pix2Pix_A2B.jpg)    |![](https://github.com/qzq2514/GAN/blob/master/GANs_Advanced/pictures/Pix2Pix_B2A.jpg)        |  

Unlike DiscoGAN、DualGAN and CycleGAN,Pix2Pix needs paired data to train and it can only generate picture from one domain to another,so I train Edge2Shoes and Shoes2Edge Separately.

# CartoonGAN
![](https://github.com/qzq2514/GAN/blob/master/GANs_Advanced/pictures/CartoonGAN.jpg)
Every two images form left to right in each grid represent realistic picture and its cartoonized counterpart.

