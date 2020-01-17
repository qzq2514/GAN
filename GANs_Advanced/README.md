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

# SRGAN         
|   Down sampling   | INTER_LINEAR |  SRGAN | Ground Truth
|:------------:|:-------------------:|:-------------------:|:-------------------:|
| ![](https://github.com/qzq2514/GAN/blob/master/GANs_Advanced/pictures/SRGAN/104500_3_real_LR.jpg)    |![](https://github.com/qzq2514/GAN/blob/master/GANs_Advanced/pictures/SRGAN/104500_3_real_4x.jpg) | ![](https://github.com/qzq2514/GAN/blob/master/GANs_Advanced/pictures/SRGAN/104500_3_fake_HR.jpg) |![](https://github.com/qzq2514/GAN/blob/master/GANs_Advanced/pictures/SRGAN/104500_3_real_HR.jpg)|
| ![](https://github.com/qzq2514/GAN/blob/master/GANs_Advanced/pictures/SRGAN/105000_5_real_LR.jpg)    |![](https://github.com/qzq2514/GAN/blob/master/GANs_Advanced/pictures/SRGAN/105000_5_real_4x.jpg) | ![](https://github.com/qzq2514/GAN/blob/master/GANs_Advanced/pictures/SRGAN/105000_5_fake_HR.jpg) |![](https://github.com/qzq2514/GAN/blob/master/GANs_Advanced/pictures/SRGAN/105000_5_real_HR.jpg)|
| ![](https://github.com/qzq2514/GAN/blob/master/GANs_Advanced/pictures/SRGAN/106500_7_real_LR.jpg)    |![](https://github.com/qzq2514/GAN/blob/master/GANs_Advanced/pictures/SRGAN/106500_7_real_4x.jpg) | ![](https://github.com/qzq2514/GAN/blob/master/GANs_Advanced/pictures/SRGAN/106500_7_fake_HR.jpg) |![](https://github.com/qzq2514/GAN/blob/master/GANs_Advanced/pictures/SRGAN/106500_7_real_HR.jpg)|
| ![](https://github.com/qzq2514/GAN/blob/master/GANs_Advanced/pictures/SRGAN/107000_4_real_LR.jpg)    |![](https://github.com/qzq2514/GAN/blob/master/GANs_Advanced/pictures/SRGAN/107000_4_real_4x.jpg) | ![](https://github.com/qzq2514/GAN/blob/master/GANs_Advanced/pictures/SRGAN/107000_4_fake_HR.jpg) |![](https://github.com/qzq2514/GAN/blob/master/GANs_Advanced/pictures/SRGAN/107000_4_real_HR.jpg)|
| ![](https://github.com/qzq2514/GAN/blob/master/GANs_Advanced/pictures/SRGAN/107500_3_real_LR.jpg)    |![](https://github.com/qzq2514/GAN/blob/master/GANs_Advanced/pictures/SRGAN/107500_3_real_4x.jpg) | ![](https://github.com/qzq2514/GAN/blob/master/GANs_Advanced/pictures/SRGAN/107500_3_fake_HR.jpg) |![](https://github.com/qzq2514/GAN/blob/master/GANs_Advanced/pictures/SRGAN/107500_3_real_HR.jpg)|
| ![](https://github.com/qzq2514/GAN/blob/master/GANs_Advanced/pictures/SRGAN/108000_0_real_LR.jpg)    |![](https://github.com/qzq2514/GAN/blob/master/GANs_Advanced/pictures/SRGAN/108000_0_real_4x.jpg) | ![](https://github.com/qzq2514/GAN/blob/master/GANs_Advanced/pictures/SRGAN/108000_0_fake_HR.jpg) |![](https://github.com/qzq2514/GAN/blob/master/GANs_Advanced/pictures/SRGAN/108000_0_real_HR.jpg)|
| ![](https://github.com/qzq2514/GAN/blob/master/GANs_Advanced/pictures/SRGAN/108000_2_real_LR.jpg)    |![](https://github.com/qzq2514/GAN/blob/master/GANs_Advanced/pictures/SRGAN/108000_2_real_4x.jpg) | ![](https://github.com/qzq2514/GAN/blob/master/GANs_Advanced/pictures/SRGAN/108000_2_fake_HR.jpg) |![](https://github.com/qzq2514/GAN/blob/master/GANs_Advanced/pictures/SRGAN/108000_2_real_HR.jpg)|
| ![](https://github.com/qzq2514/GAN/blob/master/GANs_Advanced/pictures/SRGAN/108000_6_real_LR.jpg)    |![](https://github.com/qzq2514/GAN/blob/master/GANs_Advanced/pictures/SRGAN/108000_6_real_4x.jpg) | ![](https://github.com/qzq2514/GAN/blob/master/GANs_Advanced/pictures/SRGAN/108000_6_fake_HR.jpg) |![](https://github.com/qzq2514/GAN/blob/master/GANs_Advanced/pictures/SRGAN/108000_6_real_HR.jpg)|
Super resolution is good on cartoon dataset here, experiments on real datasets will be later...
