import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

class DiscoGAN:
    def __init__(self,is_training,reconst_rate):
        self.is_training = is_training
        self.epsilon = 1e-5
        self.weight_decay = 0.00001
        self.rate = reconst_rate

    def preprocess(self,images,scale=False):
        images = tf.to_float(images)
        if scale:
            images = tf.div(images, 127.5)
            images = tf.subtract(images, 1.0)
        return images

    #[None,64,64,3]-->[None,64,64,3]
    def generator(self,inputs,name_scope,reuse=False):
        with tf.variable_scope(name_scope,reuse=reuse) as scope:
            w_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
            with slim.arg_scope([slim.conv2d], padding="SAME", activation_fn=None, stride=2,
                weights_initializer=w_init,weights_regularizer=slim.l2_regularizer(self.weight_decay)):
                with slim.arg_scope([slim.conv2d_transpose], padding="SAME", activation_fn=None, stride=2,
                                    weights_initializer=w_init,weights_regularizer=slim.l2_regularizer(self.weight_decay)):
                    #使用updates_collections=None强制更新参数
                    with slim.arg_scope([slim.batch_norm], decay=0.9, epsilon=1e-5, scale=True,updates_collections=None,
                                        activation_fn=tf.nn.leaky_relu, is_training=self.is_training):
                        #Encode
                        net = slim.conv2d(inputs,64,[4, 4],activation_fn=tf.nn.leaky_relu)  #[None,32,32,64]
                        net = slim.conv2d(net, 128, [4, 4])  #[None,16,16,128]
                        net = slim.batch_norm(net)
                        net = slim.conv2d(net, 256, [4, 4])  #[None,8,8,256]
                        net = slim.batch_norm(net)
                        net = slim.conv2d(net, 512, [4, 4])  #[None,4,4,512]
                        net = slim.batch_norm(net)

                        #Decode
                        net = slim.conv2d_transpose(net, 256, [4, 4])  # [None,8,8,64]
                        net = slim.batch_norm(net,activation_fn=tf.nn.relu)
                        net = slim.conv2d_transpose(net, 128, [4, 4])  # [None,16,16,128]
                        net = slim.batch_norm(net,activation_fn=tf.nn.relu)
                        net = slim.conv2d_transpose(net, 64, [4, 4])  # [None,32,32,64]
                        net = slim.batch_norm(net,activation_fn=tf.nn.relu)
                        net = slim.conv2d_transpose(net, 3, [4, 4])  # [None,64,64,3]
                        net = tf.nn.tanh(net)
                        return net

    # [None,64,64,3]-->[None,1,1,1]
    def discriminator(self,inputs,name_scope,reuse=False):
        with tf.variable_scope(name_scope,reuse=reuse) as scope:
            w_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
            with slim.arg_scope([slim.conv2d], padding="SAME", activation_fn=None, stride=2,
                                weights_initializer=w_init,weights_regularizer=slim.l2_regularizer(self.weight_decay)):
                # 使用updates_collections=None强制更新参数
                with slim.arg_scope([slim.batch_norm], decay=0.9, epsilon=1e-5, scale=True,updates_collections=None,
                                    activation_fn=tf.nn.leaky_relu, is_training=self.is_training):
                    feature1 = slim.conv2d(inputs, 64, [4, 4],activation_fn = tf.nn.leaky_relu)  # [None,32,32,64]

                    feature2 = slim.conv2d(feature1, 128, [4, 4])  # [None,16,16,128]
                    feature2 = slim.batch_norm(feature2)

                    feature3 = slim.conv2d(feature2, 256, [4, 4])  # [None,8,8,256]
                    feature3 = slim.batch_norm(feature3)

                    feature4 = slim.conv2d(feature3, 512, [4, 4])  # [None,4,4,512]
                    feature4 = slim.batch_norm(feature4)

                    out_logits = slim.conv2d(feature4, 1, [4, 4],padding="VALID")  # [None,1,1,1]
                    out_logits = slim.flatten(out_logits)

                    return out_logits,[feature2,feature3,feature4]

    def feature_loss(self,features_real,features_fake):
        total_loss = 0
        for feature_real,feature_fake in zip(features_real,features_fake):
            l2_loss = tf.square(tf.reduce_mean(feature_real) - tf.reduce_mean(feature_fake))
            total_loss += l2_loss
        return total_loss

    def huber_loss(self, logits, labels, max_gradient=1.0):
        err = tf.abs(labels - logits)
        mg = tf.constant(max_gradient)
        lin = mg * (err - 0.5 * mg)
        quad = 0.5 * err * err
        return tf.where(err < mg, quad, lin)

    def get_vars(self):
        all_vars = tf.trainable_variables()
        g_vars = [var for var in all_vars if var.name.startswith("generator_")]
        d_vars = [var for var in all_vars if var.name.startswith("discriminator_")]
        return g_vars, d_vars

    def build_DiscoGAN(self,input_A,input_B):
        #归一化
        input_A_pre = self.preprocess(input_A, scale=True)
        input_B_pre = self.preprocess(input_B, scale=True)

        #Domain A --> Domain B
        AB = self.generator(input_A_pre,"generator_AB")
        ABA = self.generator(AB,"generator_BA")

        AB_logits,features_AB = self.discriminator(AB, "discriminator_B")
        B_logits,features_B = self.discriminator(input_B_pre, "discriminator_B", reuse=True)
        features_B_loss = self.feature_loss(features_B,features_AB)
        reconst_A_loss = tf.reduce_mean(tf.losses.mean_squared_error(ABA, input_A_pre))
        fake_Gen_AB_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=AB_logits, labels=tf.ones_like(AB_logits)))
        Gen_AB_loss = (fake_Gen_AB_loss*0.1+features_B_loss*0.9)*(1-self.rate)+reconst_A_loss*self.rate
        fake_Dis_AB_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=AB_logits, labels=tf.zeros_like(AB_logits)))
        real_Dis_B_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=B_logits, labels=tf.ones_like(B_logits)))
        Dis_B_loss = fake_Dis_AB_loss + real_Dis_B_loss

        # Domain B --> Domain A
        BA  = self.generator(input_B_pre, "generator_BA",reuse=True)
        BAB  = self.generator(BA, "generator_AB",reuse=True)

        BA_logits, features_BA = self.discriminator(BA, "discriminator_A")
        A_logits, features_A = self.discriminator(input_A_pre, "discriminator_A", reuse=True)
        features_A_loss = self.feature_loss(features_A, features_BA)
        reconst_B_loss = tf.reduce_mean(tf.losses.mean_squared_error(BAB, input_B_pre))
        fake_Gen_BA_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=BA_logits, labels=tf.ones_like(BA_logits)))
        Gen_BA_loss = (fake_Gen_BA_loss * 0.1 + features_A_loss * 0.9) * (1 - self.rate) +reconst_B_loss * self.rate
        fake_Dis_BA_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=BA_logits, labels=tf.zeros_like(BA_logits)))
        real_Dis_A_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=A_logits, labels=tf.ones_like(A_logits)))
        Dis_A_loss = fake_Dis_BA_loss + real_Dis_A_loss

        Gen_loss = Gen_AB_loss + Gen_BA_loss
        Dis_loss = Dis_B_loss + Dis_A_loss

        #正则化损失
        reg_loss_gen_AB = tf.add_n(slim.losses.get_regularization_losses(scope="generator_AB"))
        reg_loss_gen_BA = tf.add_n(slim.losses.get_regularization_losses(scope="generator_BA"))
        reg_loss_dis_A  = tf.add_n(slim.losses.get_regularization_losses(scope="discriminator_A"))
        reg_loss_dis_B  = tf.add_n(slim.losses.get_regularization_losses(scope="discriminator_B"))

        Gen_loss += reg_loss_gen_AB + reg_loss_gen_BA
        Dis_loss += reg_loss_dis_A + reg_loss_dis_B
        return Gen_loss,Dis_loss

    def sample_generate(self,input,type="A2B"):
        if type=="A2B":
            name_scope_first = "generator_AB"
            name_scope_second = "generator_BA"
        else:
            name_scope_first = "generator_BA"
            name_scope_second = "generator_AB"
        input_pre = self.preprocess(input, scale=True)
        generated_out = self.generator(input_pre,name_scope=name_scope_first,reuse=True)
        reconst_image = self.generator(generated_out,name_scope=name_scope_second,reuse=True)
        return generated_out,reconst_image



