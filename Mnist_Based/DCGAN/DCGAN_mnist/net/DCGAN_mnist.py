import tensorflow as tf

class DCGAN_mnist:
    def __init__(self,smooth,is_training,num_class,batch_size,
                 image_width,image_height,image_channal):
        self.smooth=smooth
        self.is_training=is_training
        self.num_class=num_class
        self.batch_size=batch_size
        self.image_width=image_width
        self.image_height=image_height
        self.image_channal=image_channal

    def leakyReLU(self,inputs,leak=0.01):
        return tf.maximum(leak * inputs, inputs)

    #z_prior:[None,1,1,100]
    #label:[None,1,1,10]
    def generator(self,z_prior,label,reuse=False):
        with tf.variable_scope("generator",reuse=reuse):
            w_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
            b_init = tf.constant_initializer(0.0)

            label_one_hot = tf.one_hot(label, depth=self.num_class)
            input_cat = tf.concat([z_prior,label_one_hot],axis=1)
            channals_num = input_cat.get_shape().as_list()
            reshaped = tf.reshape(input_cat, shape=[-1, 1, 1, channals_num[-1]])

            deconv1 = tf.layers.conv2d_transpose(reshaped,256,[7,7],strides=[1,1],padding="valid",
                                                 kernel_initializer=w_init, bias_initializer=b_init)
            lrelu1 = self.leakyReLU(tf.layers.batch_normalization(deconv1, training=self.is_training))

            deconv2 = tf.layers.conv2d_transpose(lrelu1, 128, [5, 5], strides=[2, 2], padding="same",
                                                 kernel_initializer=w_init, bias_initializer=b_init)
            lrelu2 = self.leakyReLU(tf.layers.batch_normalization(deconv2, training=self.is_training))

            deconv3 = tf.layers.conv2d_transpose(lrelu2, 1, [5, 5], strides=(2, 2), padding='same',
                                                 kernel_initializer=w_init, bias_initializer=b_init)
            output = tf.nn.tanh(deconv3)

            return output

    #inputs:[None,image_height,image_width,1]
    #label_fill:[None,image_height,image_width,num_class]
    def discriminator(self,inputs,label,reuse=False):
        with tf.variable_scope("discriminator", reuse=reuse):
            w_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
            b_init = tf.constant_initializer(0.0)

            label_one_hot = tf.one_hot(label,depth=self.num_class)
            label_fill = tf.reshape(label_one_hot, shape=[self.batch_size, 1, 1, self.num_class]) * \
                         tf.ones(shape=[self.batch_size, self.image_height, self.image_width, self.num_class])

            cat = tf.concat([inputs, label_fill], axis=3)

            conv1 = tf.layers.conv2d(cat, 128, [5, 5], strides=[2, 2], padding="same",
                                                 kernel_initializer=w_init, bias_initializer=b_init)
            lrelu1 = self.leakyReLU(tf.layers.batch_normalization(conv1, training=self.is_training))

            conv2 = tf.layers.conv2d(lrelu1, 256, [5, 5], strides=[2, 2], padding="same",
                                                 kernel_initializer=w_init, bias_initializer=b_init)
            lrelu2 = self.leakyReLU(tf.layers.batch_normalization(conv2, training=self.is_training))

            logits = tf.layers.conv2d(lrelu2, 1, [7, 7], strides=(1, 1), padding='valid',
                                                 kernel_initializer=w_init)
            output = tf.nn.sigmoid(logits)

            return output,logits

    def get_vars(self):
        all_vars = tf.trainable_variables()
        g_vars = [var for var in all_vars if var.name.startswith("generator")]
        d_vars = [var for var in all_vars if var.name.startswith("discriminator")]
        return g_vars,d_vars

    def loss(self,y_data,y_generated):
        #又因为判别器只是用来判断输入的数据是真实数据还是生成器生成的数据
        #所以这里损失不用考虑样本的具体类别信息(即到底是哪个字符)
        #所以这里的损失和一般的GAN损失是一样的
        d_loss = - tf.reduce_mean((1-self.smooth)*tf.log(tf.clip_by_value(y_data,1e-8,1.0))
                                  + tf.log(tf.clip_by_value(1 - y_generated,1e-8,1.0)))
        g_loss = - tf.reduce_mean(tf.log(tf.clip_by_value(y_generated,1e-8,1.0)))
        return g_loss,d_loss

    def loss_tf(self,y_data,y_generated):
        d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=y_data,labels=tf.ones_like(y_data) * (1 - self.smooth)))
        d_loss_generated = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=y_generated,labels=tf.zeros_like(y_generated)))

        d_loss = d_loss_real + d_loss_generated

        g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=y_generated,
                                                    labels=tf.ones_like(y_generated)))
        return g_loss, d_loss


