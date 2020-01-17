import sys
import numpy as np
sys.path.append("..")

import tensorflow as tf
import tensorflow.contrib.slim as slim
from Vgg19 import Vgg19

batch_norm = tf.layers.batch_normalization 
vgg_file_path = '/home/cgim/桌面/tensorflow_train/GAN/vgg19.npy'
class SRGAN:
    def __init__(self,is_training,vgg_weight):
        self.is_training = is_training
        self.epsilon = 1e-5
        self.weight_decay = 0.00001
        self.vgg_weight = vgg_weight
        self.REAL_LABEL=0.9

    def preprocess(self,images,scale=False):
        images = tf.to_float(images)
        if scale:
            images = tf.div(images, 127.5)
            images = tf.subtract(images, 1.0)
        return images

    def sample(self, input, type="down",sample_size=4):
        shape = input.get_shape().as_list()  # NHWC
        if (type == "down"):
            h = int(shape[1] // sample_size)
            w = int(shape[1] // sample_size)
        else:
            h = int(shape[1] * sample_size)
            w = int(shape[1] * sample_size)
        resized_image = tf.image.resize_images(input, [h, w],
                                               tf.image.ResizeMethod.BILINEAR)
        return resized_image

    def Resblock(self,inputs,scope_name):
        with tf.variable_scope(scope_name+"/layer1") as scope:
            conv1 = slim.conv2d(inputs,64,[3,3],[1,1])
            norm1 = slim.batch_norm(conv1)
            relu1 = tf.nn.relu(norm1)
        with tf.variable_scope(scope_name+"/layer2") as scope:
            conv2 = slim.conv2d(relu1,64,[3,3],[1,1])
            norm2 = slim.batch_norm(conv2)
        return inputs+norm2

    def get_content_loss(self,feature_a,feature_b,type="VGG"):
        if type=="VGG":
            print("Using VGG loss as content loss!")
            vgg_a, vgg_b = Vgg19(vgg_file_path), Vgg19(vgg_file_path)
            vgg_a.build(feature_a)
            vgg_b.build(feature_b)
            VGG_loss = tf.reduce_mean(tf.losses.absolute_difference(vgg_a.conv4_4, vgg_b.conv4_4))
            h = tf.cast(tf.shape(vgg_a.conv4_4)[1], tf.float32)
            w = tf.cast(tf.shape(vgg_a.conv4_4)[2], tf.float32)
            c = tf.cast(tf.shape(vgg_a.conv4_4)[3], tf.float32)
            content_loss = VGG_loss/(h*w*c)
        else:
            print("Using MSE loss of images as content loss!")
            content_loss=tf.reduce_mean(tf.losses.absolute_difference(feature_a , feature_b))
        return content_loss

    def pixel_shuffle_layer(self,x, r, n_split):
        def PS(x, r):
            bs, a, b, c = x.get_shape().as_list()
            x = tf.reshape(x, (-1, a, b, r, r))
            x = tf.transpose(x, [0, 1, 2, 4, 3])
            x = tf.split(x, a, 1)
            x = tf.concat([tf.squeeze(x_) for x_ in x], 2)
            x = tf.split(x, b, 1)
            x = tf.concat([tf.squeeze(x_) for x_ in x], 2)
            return tf.reshape(x, (-1, a * r, b * r, 1))
        xc = tf.split(x, n_split, 3)
        return tf.concat([PS(x_, r) for x_ in xc], 3)

    #(64*64--->256*256)
    def generator(self,inputs,name_scope,reuse=False):
        print("SRGAN_onlyMSE_generator")
        with tf.variable_scope(name_scope,reuse=reuse) as scope:
            w_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
            with slim.arg_scope([slim.conv2d],weights_initializer=w_init,padding="SAME",activation_fn=None):
                with slim.arg_scope([slim.conv2d_transpose],weights_initializer=w_init,padding="SAME",activation_fn=None):
                    with slim.arg_scope([slim.batch_norm], decay=0.9, epsilon=1e-5, scale=True,
                                        activation_fn=None,is_training=self.is_training):
                        print("inputs:",inputs)
                        net = slim.conv2d(inputs,64,[3,3],[1,1])
                        net = tf.nn.relu(net)

                        short_cut = net
                        print("net1:", net)
                        #8 Resblock
                        for i in range(6):
                            net = self.Resblock(net, "ResBlock{}".format(i))
                        
                        #DeConv
                        net = slim.conv2d_transpose(net, 64, [3,3], [1,1])
                        net = slim.batch_norm(net)
                        net = net+short_cut

                        print("net2:",net)
                        net = slim.conv2d_transpose(net, 256, [3, 3], [1, 1])
                        print("net3:",net)
                        net = self.pixel_shuffle_layer(net, 2, 64)
                        net = tf.nn.relu(net)
                        print("net4:",net)
                        net = slim.conv2d_transpose(net, 256, [3, 3], [1, 1])
                        print("net5:",net)
                        net = self.pixel_shuffle_layer(net, 2, 64)
                        net = tf.nn.relu(net)

                        net = slim.conv2d(net, 3, [3, 3], [1, 1],activation_fn=tf.nn.tanh)
                        return net

    # (256*256--->1)
    def discriminator(self,inputs,name_scope,reuse=False):
        print("SRGAN_onlyMSE_discriminator")
        with tf.variable_scope(name_scope,reuse=reuse) as scope:
            w_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
            with slim.arg_scope([slim.conv2d], weights_initializer=w_init, padding="SAME", activation_fn=None):
                with slim.arg_scope([slim.conv2d_transpose], weights_initializer=w_init, padding="SAME",activation_fn=None):
                    with slim.arg_scope([slim.batch_norm], decay=0.9, epsilon=1e-5, scale=True,activation_fn=None,is_training=self.is_training):
                        nfg = 64
                        net = slim.conv2d(inputs,nfg,[3,3],[1,1])
                        net = tf.nn.leaky_relu(net)
                        print("net:",net)

                        for i in range(1,5):
                            net = slim.conv2d(net, nfg, [3, 3], [2, 2])
                            net = slim.batch_norm(net)
                            net = tf.nn.leaky_relu(net)

                            net = slim.conv2d(net, nfg*2, [3, 3], [1, 1])
                            net = slim.batch_norm(net)
                            net = tf.nn.leaky_relu(net)
                            nfg *= 2
                            print("dis{}:".format(i),net)

                        net = slim.conv2d(net, nfg, [3, 3], [2, 2])
                        net = slim.batch_norm(net)
                        logits = tf.nn.leaky_relu(net)

                        net_flat = tf.layers.flatten(net)
                        dense_net = slim.fully_connected(net_flat,1024)
                        dense_net = tf.nn.leaky_relu(dense_net)
                        logits = slim.fully_connected(dense_net, 1)
                        return logits

    def get_vars(self):
        all_vars = tf.trainable_variables()
        dis_vars = [var for var in all_vars if 'discriminator' in var.name]
        gen_vars = [var for var in all_vars if 'generator' in var.name]
        
        return gen_vars,dis_vars

    def build_CartoonGAN(self,LR,HR):
        #归一化
        LR_pre = self.preprocess(LR, scale=True)
        HR_pre = self.preprocess(HR, scale=True)

        #reality --> cartoon
        fake_HR = self.generator(LR_pre,"generator")

        fake_HR_logits = self.discriminator(fake_HR, "discriminator", reuse=False)
        real_HR_logits = self.discriminator(HR_pre, "discriminator", reuse=True)

        #GAN损失
        real_dis_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                                logits=real_HR_logits,labels=tf.ones_like(real_HR_logits)))
        fake_dis_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                                logits=fake_HR_logits,labels=tf.zeros_like(fake_HR_logits)))
        fake_gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                                logits=fake_HR_logits,labels=tf.ones_like(fake_HR_logits)))
        dis_loss = real_dis_loss+fake_dis_loss

        print("size:",HR_pre,fake_HR)
        content_loss = self.get_content_loss(HR_pre, fake_HR,"no_VGG")
        psnr = self.get_PSNR(HR_pre,fake_HR)
        gen_loss = fake_gen_loss + self.vgg_weight*content_loss

        return gen_loss,dis_loss,content_loss,psnr

    def get_PSNR(self,real, fake):
        mse = tf.reduce_mean(tf.square(127.5 * (real - fake) + 127.5), axis=(-3, -2, -1))
        psnr = tf.reduce_mean(10 * (tf.log(255 * 255 / tf.sqrt(mse)) / np.log(10)))
        return psnr
    def sample_generate(self,LR):
        LR_pre = self.preprocess(LR, scale=True)
        HR_out = self.generator(LR_pre,name_scope="generator",reuse=True)
        return HR_out



