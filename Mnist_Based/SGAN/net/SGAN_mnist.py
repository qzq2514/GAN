import tensorflow as tf

class SGAN_mnist:
    def __init__(self,smooth,is_training,num_class,batch_size):
        self.smooth = smooth
        self.is_training = is_training
        self.num_class = num_class
        self.batch_size = batch_size

    def leakyReLU(self,inputs,leak=0.01):
        return tf.maximum(leak * inputs, inputs)

    # z_prior:[None,prior_size]
    # label:[None,]
    # 原始SGAN的重要特点是利用类别损失进一步优化判别器,从而更好"监督"生成器生成更真实的图像
    # 但是原始SGAN是一个非条件GAN,即其仅用判别器输出的类别新信息优化判别器和生成器,但是生成器是不接受类别条件的
    # 这里保留了SGAN的最终特点:即类别损失,但是在此基础上添加了类条件输入,可以控制生成标签的类别
    def generator(self,z_prior,label,reuse=False):
        with tf.variable_scope("generator",reuse=reuse):
            w_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
            b_init = tf.constant_initializer(0.0)

            ont_hot_label = tf.one_hot(label, depth=self.num_class)
            input_cat = tf.concat(values=[ont_hot_label, z_prior], axis=1)

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
                d_logits = tf.layers.conv2d(lrelu2, 1, [7, 7], strides=(1, 1), padding='valid',
                                         kernel_initializer=w_init)
                d_sigmoid = tf.nn.sigmoid(d_logits)

            with tf.variable_scope("Q", reuse=reuse):
                q_share_conv1 = tf.layers.conv2d(lrelu2, 128, [4, 4], strides=(2, 2), padding='same',
                                            kernel_initializer=w_init)
                q_share_lrelu1 = self.leakyReLU(tf.layers.batch_normalization(q_share_conv1,
                                                                    training=self.is_training))
                feature_height,feature_width = q_share_lrelu1.get_shape().as_list()[1:3]

                Q_cat_logit = tf.layers.conv2d(q_share_lrelu1, self.num_class, [feature_height, feature_height],
                                             strides=(1, 1), padding='valid',kernel_initializer=w_init)
                Q_cat_logit = tf.squeeze(Q_cat_logit,axis=[1,2])

        return d_sigmoid,Q_cat_logit

    def get_vars(self):
        all_vars = tf.trainable_variables()
        g_vars = [var for var in all_vars if var.name.startswith("generator")]
        d_vars = [var for var in all_vars if var.name.startswith("discriminator")]
        return g_vars,d_vars

    def loss(self,prob_real,prob_fake,real_label,cat_logit_real,cat_logit_fake):
        #一般的损失,同DCGAN
        d_loss = - tf.reduce_mean((1-self.smooth)*tf.log(tf.clip_by_value(prob_real,1e-8,1.0))
                                  + tf.log(tf.clip_by_value(1 - prob_fake,1e-8,1.0)))
        g_loss = - tf.reduce_mean(tf.log(tf.clip_by_value(prob_fake,1e-8,1.0)))

        #SAGN的核心:使用类别信息辅助更新判别器,保证在增强判别器的同时可以“监督”生成更真实的图片
        #其实我觉得和"Improved Techniques for Training GANs_2016"论文中的半监督思想下使用类别损失是一样的
        #这一点在原论文中其实也提到了,说在该篇论文之前improvedGA也提出的相同的思想
        cat_loss_real = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                      logits=cat_logit_real, labels=real_label))
        cat_loss_fake = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=cat_logit_fake, labels=real_label))
        cat_loss = (cat_loss_real + cat_loss_fake)/2

        g_loss_all = g_loss + cat_loss
        d_loss_all = d_loss + cat_loss

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


