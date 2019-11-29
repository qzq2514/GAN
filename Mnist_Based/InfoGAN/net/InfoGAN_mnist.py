import tensorflow as tf

class InfoGAN_mnist:
    def __init__(self,smooth,is_training,latent_code_size,num_class,batch_size):
        self.smooth = smooth
        self.is_training = is_training
        self.latent_code_size = latent_code_size
        self.num_class = num_class
        self.batch_size = batch_size

    def leakyReLU(self,inputs,leak=0.01):
        return tf.maximum(leak * inputs, inputs)

    # z_prior:[None,prior_size]
    # latent_code:[None,laten_code_size]
    # label:[None,]
    # 原始的InfoGAN的生成器是只有噪声和潜在编码的,不带label,即不是一个类条件GAN
    def generator(self,z_prior,latent_code,label,reuse=False):
        with tf.variable_scope("generator",reuse=reuse):
            w_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
            b_init = tf.constant_initializer(0.0)

            ont_hot_label = tf.one_hot(label, depth=self.num_class)
            input_cat = tf.concat(values=[ont_hot_label, latent_code, z_prior], axis=1)

            #使用全连接:
            # fc1 = tf.layers.dense(input_cat,1024)
            # fc1_lrelu = self.leakyReLU(tf.layers.batch_normalization(fc1, training=self.is_training))
            #
            # fc2 = tf.layers.dense(fc1_lrelu, 7*7*128)
            # fc2_lrelu = self.leakyReLU(tf.layers.batch_normalization(fc2, training=self.is_training))
            #
            # reshaped = tf.reshape(fc2_lrelu,[-1,7,7,128])

            #不使用全连接:
            channals_num = input_cat.get_shape().as_list()
            reshaped = tf.reshape(input_cat,shape=[-1,1,1,channals_num[-1]])

            deconv1 = tf.layers.conv2d_transpose(reshaped, 256, [7, 7], strides=[1, 1], padding="valid",
                                                 kernel_initializer=w_init, bias_initializer=b_init)
            lrelu1 = self.leakyReLU(tf.layers.batch_normalization(deconv1, training=self.is_training))

            #公共部分
            deconv2 = tf.layers.conv2d_transpose(lrelu1, 64, [4, 4], strides=[2, 2], padding="same",
                                                 kernel_initializer=w_init, bias_initializer=b_init)
            lrelu2 = self.leakyReLU(tf.layers.batch_normalization(deconv2, training=self.is_training))

            deconv3 = tf.layers.conv2d_transpose(lrelu2, 1, [4, 4], strides=(2, 2), padding='same',
                                                 kernel_initializer=w_init, bias_initializer=b_init)
            output = tf.nn.tanh(deconv3)

            return output

    #inputs:[None,image_height,image_width,1]
    #原始的InfoGAN的判别器输入也没有label,这里因为想要实现标签控制,所以加上了label
    def discriminator(self,inputs,label,reuse=False):
        with tf.variable_scope("discriminator", reuse=reuse):
            w_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
            b_init = tf.constant_initializer(0.0)

            ont_hot_label = tf.one_hot(label, depth=self.num_class)
            label_fill = tf.reshape(ont_hot_label,shape=[self.batch_size,1,1,10])* \
                         tf.ones(shape=[self.batch_size,28,28,10])

            input_cat = tf.concat(values=[inputs, label_fill], axis=3)

            conv1 = tf.layers.conv2d(input_cat, 64, [4, 4], strides=[2, 2], padding="same",
                                                 kernel_initializer=w_init, bias_initializer=b_init)

            lrelu1 = self.leakyReLU(tf.layers.batch_normalization(conv1, training=self.is_training))

            conv2 = tf.layers.conv2d(lrelu1, 128, [4, 4], strides=[2, 2], padding="same",
                                                 kernel_initializer=w_init, bias_initializer=b_init)
            lrelu2 = self.leakyReLU(tf.layers.batch_normalization(conv2, training=self.is_training))



            with tf.variable_scope("D", reuse=reuse):
                # conv3:[None, 1, 1, 1] 每个元素就单纯表示每张图片是真实图片的概率
                d_logits = tf.layers.conv2d(lrelu2, 1, [7, 7], strides=(1, 1), padding='valid',
                                         kernel_initializer=w_init)
                d_sigmoid = tf.nn.sigmoid(d_logits)

            with tf.variable_scope("Q", reuse=reuse):
                #不使用全连接
                q_share_conv1 = tf.layers.conv2d(lrelu2, 128, [4, 4], strides=(2, 2), padding='same',
                                            kernel_initializer=w_init)
                q_share_lrelu1 = self.leakyReLU(tf.layers.batch_normalization(q_share_conv1,
                                                                    training=self.is_training))
                feature_height,feature_width = q_share_lrelu1.get_shape().as_list()[1:3]

                Q_cat_logit = tf.layers.conv2d(q_share_lrelu1, self.num_class, [feature_height, feature_height],
                                             strides=(1, 1), padding='valid',kernel_initializer=w_init)
                Q_cat_logit = tf.squeeze(Q_cat_logit,axis=[1,2])

                Q_latent_logit = tf.layers.conv2d(q_share_lrelu1, self.latent_code_size, [4, 4],
                                             strides=(1, 1), padding='valid',kernel_initializer=w_init)
                Q_latent_logit = tf.squeeze(Q_latent_logit, axis=[1, 2])
                Q_latent_sigmod = tf.nn.sigmoid(Q_latent_logit)
                print(Q_cat_logit,Q_latent_logit)

                # 使用全连接
                # flatten = tf.layers.flatten(lrelu2)
                # fc = tf.layers.dense(flatten, 1024)
                # fc_lrelu = self.leakyReLU(tf.layers.batch_normalization(fc, training=self.is_training))
                #
                # Q_fc = tf.layers.dense(fc_lrelu, 128)
                # Q_fc_lrelu = self.leakyReLU(tf.layers.batch_normalization(Q_fc, training=self.is_training))
                # Q_cat_logit = tf.layers.dense(Q_fc_lrelu, self.num_class)
                # Q_latent_logit = tf.layers.dense(Q_fc_lrelu, self.latent_code_size)
                # Q_latent_sigmoid = tf.nn.sigmoid(Q_latent_logit)
        return d_sigmoid,Q_cat_logit,Q_latent_sigmod

    def get_vars(self):
        all_vars = tf.trainable_variables()
        g_vars = [var for var in all_vars if var.name.startswith("generator")]
        d_vars = [var for var in all_vars if var.name.startswith("discriminator")]
        return g_vars,d_vars

    def loss(self,prob_real,prob_fake,real_label,cat_logit_real,cat_logit_fake,real_latent,pred_latent):
        #一般的损失,同DCGAN
        d_loss = - tf.reduce_mean((1-self.smooth)*tf.log(tf.clip_by_value(prob_real,1e-8,1.0))
                                  + tf.log(tf.clip_by_value(1 - prob_fake,1e-8,1.0)))
        g_loss = - tf.reduce_mean(tf.log(tf.clip_by_value(prob_fake,1e-8,1.0)))

        #使用"Improved Techniques for Training GANs_2016"论文中的半监督思想
        #加入分类损失,原InfoGAN中是没有分类损失的
        cat_loss_real = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                      logits=cat_logit_real, labels=real_label))
        cat_loss_fake = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=cat_logit_fake, labels=real_label))
        cat_loss = (cat_loss_real + cat_loss_fake)/2

        #潜在编码损失,直接使用MSE损失即可
        latent_loss = tf.reduce_mean(tf.square(real_latent - pred_latent))

        g_loss_all = g_loss + cat_loss + latent_loss
        d_loss_all = d_loss + cat_loss + latent_loss

        return g_loss_all,d_loss_all

    def loss_tf(self,y_data,y_generated):
        d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=y_data,
                                                    labels=tf.ones_like(y_data) * (1 - self.smooth)))
        d_loss_generated = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=y_generated,
                                                    labels=tf.zeros_like(y_generated)))

        d_loss = d_loss_real + d_loss_generated

        g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=y_generated,
                                                    labels=tf.ones_like(y_generated)))
        return g_loss, d_loss


