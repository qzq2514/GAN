import sys
sys.path.append("..")

import tensorflow as tf
import tensorflow.contrib.slim as slim
from Vgg19 import Vgg19

batch_norm = tf.layers.batch_normalization 
vgg_file_path = '../vgg19.npy'
class CartoonGAN:
    #is_training
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

    def Resblock(self,inputs,k,scope_name):
        with tf.variable_scope(scope_name+"/layer1") as scope:
            conv1 = slim.conv2d(inputs,k,[3,3],[1,1],padding="SAME")
            norm1 = slim.batch_norm(conv1)
            relu1 = tf.nn.relu(norm1)
        with tf.variable_scope(scope_name+"/layer2") as scope:
            conv2 = slim.conv2d(relu1,k,[3,3],[1,1],padding="SAME")
            norm2 = slim.batch_norm(conv2)
        return norm2+inputs

    def instance_norm(self,inputs,scope_name):
        with tf.variable_scope(scope_name):
            epsilon = 1e-5
            mean, var = tf.nn.moments(inputs, [1, 2], keep_dims=True)
            scale = tf.get_variable('scale',[inputs.get_shape()[-1]], 
                initializer=tf.truncated_normal_initializer(mean=1.0, stddev=0.02))
            offset = tf.get_variable('offset',[inputs.get_shape()[-1]],initializer=tf.constant_initializer(0.0))
            out = scale*tf.div(inputs-mean, tf.sqrt(var+epsilon)) + offset
            return out

    def vgg_loss(self,image_a, image_b):
        vgg_a, vgg_b = Vgg19(vgg_file_path), Vgg19(vgg_file_path)
        vgg_a.build(image_a)
        vgg_b.build(image_b)
        VGG_loss = tf.reduce_mean(tf.losses.absolute_difference(vgg_a.conv4_4, vgg_b.conv4_4))
        h = tf.cast(tf.shape(vgg_a.conv4_4)[1], tf.float32)
        w = tf.cast(tf.shape(vgg_a.conv4_4)[2], tf.float32)
        c = tf.cast(tf.shape(vgg_a.conv4_4)[3], tf.float32)
        VGG_loss = VGG_loss/(h*w*c)
        return VGG_loss

    def res_block(self,input_tensor, channel, is_train=False):
        short_cut = input_tensor
        res_conv1 = slim.conv2d(input_tensor, channel, [3, 3], activation_fn=None)
        res_norm1 = batch_norm(res_conv1, training=is_train)
        res_relu1 = tf.nn.relu(res_norm1)
        res_conv2 = slim.conv2d(res_relu1, channel, [3, 3], activation_fn=None)
        res_norm2 = batch_norm(res_conv2, training=is_train)
        return res_norm2 + short_cut

    def generator(self,inputs,name_scope,reuse=False):
        print("CartoonGAN_dis_loss_generator")
        with tf.variable_scope(name_scope,reuse=reuse) as scope:
            w_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
            with slim.arg_scope([slim.conv2d],weights_initializer=w_init,padding="SAME",activation_fn=None):
                with slim.arg_scope([slim.conv2d_transpose],weights_initializer=w_init,padding="SAME",activation_fn=None):
                    with slim.arg_scope([slim.batch_norm], decay=0.9, epsilon=1e-5, scale=True,
                                        activation_fn=None,is_training=self.is_training):
                        print("inputs:",inputs)
                        nfg = 32
                        net = slim.conv2d(inputs,nfg,[7,7],[1,1])
                        net = slim.batch_norm(net)
                        net = tf.nn.relu(net)
                        print("net0:", net)
                        # Down-sampling
                        for i in range(1,3):
                            net = slim.conv2d(net, nfg*2, [3, 3], [2, 2])
                            net = slim.conv2d(net, nfg*2, [3, 3], [1, 1])
                            net = slim.batch_norm(net)
                            net = tf.nn.relu(net)
                            nfg *= 2
                            print("net{}:".format(i), net)

                        #8 Resblock
                        channals = net.get_shape().as_list()[-1]
                        for i in range(4):
                            net = self.Resblock(net, channals, "R{}_{}".format(channals, i))
                            print("Resblock{}:".format(i),net)
                        #up-Sampling
                        for i in range(2):
                            net = slim.conv2d_transpose(net, nfg//2, [3,3],[2,2])
                            net = slim.conv2d(net, nfg // 2, [3, 3], [1, 1])
                            net = slim.batch_norm(net)
                            net = tf.nn.relu(net)
                            nfg = nfg // 2
                            print("up{}:".format(i), net)
                        net = slim.conv2d(net, 3, [7, 7], [1, 1],activation_fn=None)#tf.nn.tanh
                        return net

    def l2_norm(self,v, eps=1e-12):
        return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)

    def spectral_norm(self, w, iteration=1):
        w_shape = w.shape.as_list()
        w = tf.reshape(w, [-1, w_shape[-1]])

        u = tf.get_variable("u", [1, w_shape[-1]], initializer=
                            tf.truncated_normal_initializer(), trainable=False)

        u_hat = u
        v_hat = None
        for i in range(iteration):
            v_ = tf.matmul(u_hat, tf.transpose(w))
            v_hat = self.l2_norm(v_)

            u_ = tf.matmul(v_hat, w)
            u_hat = self.l2_norm(u_)

        sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
        w_norm = w / sigma

        with tf.control_dependencies([u.assign(u_hat)]):
            w_norm = tf.reshape(w_norm, w_shape)
        return w_norm

    def conv_sn(self,x, channels, k_size, stride=1, name='conv2d'):
        with tf.variable_scope(name):
            w = tf.get_variable("kernel", shape=[k_size, k_size, x.get_shape()[-1], channels],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))

            output = tf.nn.conv2d(input=x, filter=self.spectral_norm(w), 
                            strides=[1, stride, stride, 1], padding='SAME') + b
            return output

    def leaky_relu(self,x, leak=0.2):
        return tf.maximum(x, leak*x)

    #using Spectral Normalization in discriminator
    #Reference: https://github.com/SystemErrorWang/CartoonGAN/blob/master/old_code/model.py
    def discriminator(self,input_tensor,name_scope,reuse=False):
        print("CartoonGAN_dis_loss_discriminator")
        use_bn=True
        with tf.variable_scope(name_scope,reuse=reuse) as scope:
            patach_conv_layers = []
            for i in range(4):
                batch_size = tf.shape(input_tensor)[0]
                patch = tf.random_crop(input_tensor, [batch_size, 16, 16, 3])
                patch_conv = self.conv_sn(patch, 32, 3, name='patch_conv'+str(i))
                norm_p = batch_norm(patch_conv, training=True)
                relu_p = self.leaky_relu(norm_p)
                patach_conv_layers.append(relu_p)

            patch_concat = tf.concat(patach_conv_layers, axis=-1)

            conv1 = self.conv_sn(patch_concat, 128, 3, stride=2, name='conv1')
            norm1 = batch_norm(conv1, training=True)
            relu1 = self.leaky_relu(norm1)
            
            conv2 = self.conv_sn(relu1, 256, 3, name='conv2')
            norm2 = batch_norm(conv2, training=True)
            relu2 = self.leaky_relu(norm2)
            
            conv3 = self.conv_sn(relu2, 256, 3, stride=2, name='conv3')
            norm3 = batch_norm(conv3, training=True)
            relu3 = self.leaky_relu(norm3)
            
            conv4 = self.conv_sn(relu3, 512, 3,name='conv4')
            norm4 = batch_norm(conv4, training=True)
            relu4 = self.leaky_relu(norm4)
            
            conv_out = self.conv_sn(relu4, 1, 1, name='conv7')
            avg_pool = tf.reduce_mean(conv_out, axis=[1, 2])
            
            return avg_pool

    def get_vars(self):
        all_vars = tf.trainable_variables()
        dis_vars = [var for var in all_vars if 'discriminator' in var.name]
        gen_vars = [var for var in all_vars if 'generator' in var.name]
        
        return gen_vars,dis_vars

    def build_CartoonGAN(self,reality,real_cartoon,cartoon_smooth):
        #归一化
        reality_pre = self.preprocess(reality, scale=True)
        real_cartoon_pre = self.preprocess(real_cartoon, scale=True)
        cartoon_smooth_pre = self.preprocess(cartoon_smooth, scale=True)

        #reality --> cartoon
        fake_cartoon = self.generator(reality_pre,"generator")

        fake_cartoon_logits = self.discriminator(fake_cartoon, "discriminator", reuse=False)
        real_cartoon_logits = self.discriminator(real_cartoon_pre, "discriminator", reuse=True)
        cartoon_smooth_logits = self.discriminator(cartoon_smooth_pre, "discriminator", reuse=True)

        #GAN损失
        vgg_loss = self.vgg_loss(reality_pre, fake_cartoon)
        print("self.vgg_weight:",self.vgg_weight)
        gen_loss = -tf.reduce_mean(tf.log(tf.nn.sigmoid(fake_cartoon_logits))) + 5e3*vgg_loss
        
        dis_loss = -tf.reduce_mean(tf.log(tf.nn.sigmoid(real_cartoon_logits))
                                + tf.log(1. - tf.nn.sigmoid(fake_cartoon_logits))
                                + tf.log(1. - tf.nn.sigmoid(cartoon_smooth_logits)))
        return gen_loss,dis_loss,vgg_loss

    def sample_generate(self,reality):
        reality_pre = self.preprocess(reality, scale=True)
        cartoon_out = self.generator(reality_pre,name_scope="generator",reuse=True)
        return cartoon_out



