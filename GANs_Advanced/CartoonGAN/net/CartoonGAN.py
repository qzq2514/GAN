import sys
sys.path.append("..")

import tensorflow as tf
import tensorflow.contrib.slim as slim
from Vgg19 import Vgg19

batch_norm = tf.layers.batch_normalization 
vgg_file_path = '../vgg19.npy'
class CartoonGAN:
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

    def generator(self,inputs,name_scope,reuse=False):
        print("CartoonGAN_loss_generator")
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

    def discriminator(self,inputs,name_scope,reuse=False):
        print("CartoonGAN_loss_discriminator")
        with tf.variable_scope(name_scope,reuse=reuse) as scope:
            w_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
            with slim.arg_scope([slim.conv2d], weights_initializer=w_init, padding="SAME", activation_fn=None):
                with slim.arg_scope([slim.conv2d_transpose], weights_initializer=w_init, padding="SAME",activation_fn=None):
                    with slim.arg_scope([slim.batch_norm], decay=0.9, epsilon=1e-5, scale=True,activation_fn=None,is_training=self.is_training):
                        nfg = 32
                        net = slim.conv2d(inputs,nfg,[3,3],[1,1])
                        net = slim.batch_norm(net)
                        net = tf.nn.leaky_relu(net)
                        print("net:",net)

                        for i in range(1,3):
                            net = slim.conv2d(net, nfg * 2, [3, 3], [2, 2])
                            net = tf.nn.leaky_relu(net)
                            net = slim.conv2d(net, nfg * 4, [3, 3], [1, 1])
                            net = slim.batch_norm(net)
                            net = tf.nn.leaky_relu(net)
                            nfg *= 2
                            print("dis{}:".format(i),net)

                        net = slim.conv2d(net, nfg * 2, [3, 3], [1, 1])
                        net = slim.batch_norm(net)
                        net = tf.nn.leaky_relu(net)
                        print("net:",net)

                        logits = slim.conv2d(net, 1, [3, 3], [1, 1])
                        return logits

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
        real_dis_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                                logits=real_cartoon_logits,labels=tf.ones_like(real_cartoon_logits)))
        fake_dis_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                                logits=fake_cartoon_logits,labels=tf.zeros_like(fake_cartoon_logits)))
        smooth_dis_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                                logits=cartoon_smooth_logits,labels=tf.zeros_like(cartoon_smooth_logits)))
        dis_loss = real_dis_loss+fake_dis_loss+smooth_dis_loss

        vgg_loss = self.vgg_loss(reality_pre, fake_cartoon)
        fake_gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                                logits=fake_cartoon_logits,labels=tf.ones_like(fake_cartoon_logits)))
        gen_loss = fake_gen_loss + self.vgg_weight*vgg_loss
        

        return gen_loss,dis_loss,vgg_loss

    def sample_generate(self,reality):
        reality_pre = self.preprocess(reality, scale=True)
        cartoon_out = self.generator(reality_pre,name_scope="generator",reuse=True)
        return cartoon_out



