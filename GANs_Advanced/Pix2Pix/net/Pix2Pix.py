import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

class Pix2Pix:
    def __init__(self,is_training,keep_prob,lambda_reconst):
        self.is_training = is_training
        self.epsilon = 1e-5
        self.weight_decay = 0.00001
        self.keep_prob = keep_prob
        self.lambda_reconst = lambda_reconst

    def preprocess(self,images,scale=False):
        images = tf.to_float(images)
        if scale:
            images = tf.div(images, 127.5)
            images = tf.subtract(images, 1.0)
        return images

    #[None,64,64,3]-->[None,64,64,3]
    def generator(self,inputs,reuse=False):
        with tf.variable_scope("generator",reuse=reuse) as scope:
            w_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
            with slim.arg_scope([slim.conv2d], padding="SAME", activation_fn=None, stride=2,kernel_size=[4,4],
                weights_initializer=w_init,weights_regularizer=slim.l2_regularizer(self.weight_decay)):
                with slim.arg_scope([slim.conv2d_transpose], padding="SAME", activation_fn=None, stride=2,kernel_size=[4,4],
                    weights_initializer=w_init,weights_regularizer=slim.l2_regularizer(self.weight_decay)):
                    # 使用updates_collections=None强制更新参数
                    with slim.arg_scope([slim.batch_norm], decay=0.9, epsilon=1e-5, scale=True,updates_collections=None,
                                        activation_fn=None,is_training=self.is_training):
                        #Encode
                        e1 = slim.conv2d(inputs,64,activation_fn=None)  #[None,32,32,64]
                        e2 = slim.conv2d(tf.nn.leaky_relu(e1), 64*2)  #[None,16,16,128]
                        e2 = slim.batch_norm(e2)
                        e3 = slim.conv2d(tf.nn.leaky_relu(e2), 64*4)  #[None,8,8,256]
                        e3 = slim.batch_norm(e3)
                        e4 = slim.conv2d(tf.nn.leaky_relu(e3), 64*8)  #[None,4,4,512]
                        e4 = slim.batch_norm(e4)
                        e5 = slim.conv2d(tf.nn.leaky_relu(e4), 64*8)  # [None,2,2,512]
                        e5 = slim.batch_norm(e5)
                        e6 = slim.conv2d(tf.nn.leaky_relu(e5), 64*8)  # [None,1,1,512]
                        e6 = slim.batch_norm(e6)
                        # e7 = slim.conv2d(tf.nn.leaky_relu(e6), 64*8)  # [None,1,1,512]
                        # e7 = slim.batch_norm(e7)

                        #Decode
                        d1 = slim.conv2d_transpose(tf.nn.relu(e6), 64 * 8)
                        d1 = tf.nn.dropout(slim.batch_norm(d1),self.keep_prob)
                        d1 = tf.concat([d1, e5],3)                          # [None,2,2,512*2]
                        d2 = slim.conv2d_transpose(tf.nn.relu(d1), 64 * 8)
                        d2 = tf.nn.dropout(slim.batch_norm(d2), self.keep_prob)
                        d2 = tf.concat([d2, e4], 3)                         # [None,4,4,512*2]
                        d3 = slim.conv2d_transpose(tf.nn.relu(d2), 64 * 8)
                        d3 = tf.nn.dropout(slim.batch_norm(d3), self.keep_prob)
                        d3 = tf.concat([d3, e3], 3)                         # [None,8,8,512*2]
                        d4 = slim.conv2d_transpose(tf.nn.relu(d3), 64 * 4)
                        d4 = tf.nn.dropout(slim.batch_norm(d4), self.keep_prob)
                        d4 = tf.concat([d4, e2], 3)                         # [None,16,16,256*2]
                        d5 = slim.conv2d_transpose(tf.nn.relu(d4), 64 * 2)
                        d5 = tf.nn.dropout(slim.batch_norm(d5), self.keep_prob)
                        d5 = tf.concat([d5, e1], 3)                         # [None,32,32,128*2]
                        # d6 = slim.conv2d_transpose(tf.nn.relu(d5), 64 * 2)
                        # d6 = tf.nn.dropout(slim.batch_norm(d6), self.keep_prob)
                        # d6 = tf.concat([d6, e1], 3)
                        d6 = slim.conv2d_transpose(tf.nn.relu(d5), 3)       # [None,64,64,3]
                        generate_out = tf.nn.tanh(d6)
                        return generate_out

    # [None,128,128,3]-->[None,1,1,1]
    def discriminator(self,inputs,reuse=False):
        with tf.variable_scope("discriminator",reuse=reuse) as scope:
            w_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
            with slim.arg_scope([slim.conv2d], padding="SAME", activation_fn=None, stride=2,kernel_size=[5,5],
                                weights_initializer=w_init,weights_regularizer=slim.l2_regularizer(self.weight_decay)):
                # 使用updates_collections=None强制更新参数
                with slim.arg_scope([slim.batch_norm], decay=0.9, epsilon=1e-5, scale=True,updates_collections=None,
                                    activation_fn=tf.nn.leaky_relu, is_training=self.is_training):
                    feature1 = slim.conv2d(inputs, 64,activation_fn = tf.nn.leaky_relu)  # [None,32,32,64]
                    feature2 = slim.conv2d(feature1, 128)  # [None,16,16,128]
                    feature2 = slim.batch_norm(feature2)
                    feature3 = slim.conv2d(feature2, 256)  # [None,8,8,256]
                    feature3 = slim.batch_norm(feature3)
                    feature4 = slim.conv2d(feature3, 512,stride=1)  # [None,8,8,512]
                    feature4 = slim.batch_norm(feature4)
                    out_logits = slim.conv2d(feature4, 1,stride=1)  # [None,8,8,1]
                    return out_logits

    def get_vars(self):
        all_vars = tf.trainable_variables()
        g_vars = [var for var in all_vars if var.name.startswith("generator")]
        d_vars = [var for var in all_vars if var.name.startswith("discriminator")]
        return g_vars, d_vars

    def build_Pix2Pix(self,input_A,input_B):
        #归一化
        input_A_pre = self.preprocess(input_A, scale=True)
        input_B_pre = self.preprocess(input_B, scale=True)

        #Domain A --> Domain B
        AB = self.generator(input_A_pre)

        #原论文pix2pix对在语义图像转现实图像的实验上,分别需要将语义图和生成的现实图像和原本的真实图像在
        #通道维度上拼接再送入判别器判别(这和之前使用cGAN生成MNIST是一样的想法)
        #本实验以shoes2edge和edge2shoe为例,不牵扯语义图,故未进行拼接
        #AB = tf.concat([input_A_pre, AB], 3)
        #input_B_pre = tf.concat([input_A_pre, input_B_pre], 3)
        AB_logits = self.discriminator(AB)
        B_logits = self.discriminator(input_B_pre, reuse=True)
        
        reconst_B_loss = tf.reduce_mean(tf.abs(input_B_pre - AB))
        fake_Gen_AB_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=AB_logits, labels=tf.ones_like(AB_logits)))
        reconst_B_loss = self.lambda_reconst*reconst_B_loss
        fake_Dis_AB_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=AB_logits, labels=tf.zeros_like(AB_logits)))
        real_Dis_B_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=B_logits, labels=tf.ones_like(B_logits)))

        Dis_loss = fake_Dis_AB_loss + real_Dis_B_loss
        Gen_loss = fake_Gen_AB_loss + reconst_B_loss

        return Gen_loss,Dis_loss

    def sample_generate(self,input):
        input_pre = self.preprocess(input, scale=True)
        generated_out = self.generator(input_pre,reuse=True)
        return generated_out



