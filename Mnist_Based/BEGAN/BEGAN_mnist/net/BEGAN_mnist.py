import tensorflow as tf

class BEGAN_mnist:
    def __init__(self,is_training,num_class,batch_size,
                 image_height,image_width,channels,gamma,lamda):
        self.is_training = is_training
        self.num_class = num_class
        self.batch_size = batch_size
        self.image_height = image_height
        self.image_width = image_width
        self.channels = channels
        self.k = tf.Variable(0.0, trainable=False)
        self.gamma = gamma
        self.lamda = lamda

    def leakyReLU(self,inputs,leak=0.01):
        return tf.maximum(leak * inputs, inputs)

    #BEGAN:和EBGAN一样,使用自编码器作为判别器
    def discriminator(self, inputs, reuse=False):
        with tf.variable_scope("discriminator", reuse=reuse):
            w_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
            b_init = tf.constant_initializer(0.0)

            net = tf.layers.conv2d(inputs, 64, [4, 4], strides=[2, 2], padding="same",
                                   kernel_initializer=w_init, bias_initializer=b_init)
            net = tf.nn.relu(net)
            feature_height, feature_width = net.get_shape().as_list()[1:3]
            net = tf.layers.flatten(net)
            code = tf.layers.dense(net, 32)
            net = tf.contrib.layers.batch_norm(code, decay=0.9, updates_collections=None,
                                               epsilon=1e-5, scale=True, is_training=self.is_training)
            net = tf.nn.relu(net)
            net = tf.layers.dense(net, feature_height * feature_width * 64)
            net = tf.contrib.layers.batch_norm(net, decay=0.9,updates_collections=None,
                                               epsilon=1e-5,scale=True,is_training=self.is_training)
            net = tf.nn.relu(net)
            net = tf.reshape(net, shape=[-1, feature_height, feature_width, 64])
            net = tf.layers.conv2d_transpose(net, 1, [4, 4], strides=[2, 2], padding="same",
                                             kernel_initializer=w_init, bias_initializer=b_init)
            #因为原图和生成的图像的像素值都是0~1范围,所以这里要用sigmod激活函数,不要用tanh
            #除非图像像素值全部统一到-1~1,就可以用tanh
            recon_out = tf.sigmoid(net)
            recon_error = tf.sqrt(2 * tf.nn.l2_loss(recon_out - inputs)) / self.batch_size
            # recon_error = tf.reduce_mean(tf.square(recon_out - x))
            return recon_out, recon_error, code

    def generator(self, z_prior, reuse=False):
        with tf.variable_scope("generator", reuse=reuse):
            w_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
            b_init = tf.constant_initializer(0.0)

            net = tf.layers.dense(z_prior, 1024)
            relu1 = tf.nn.relu(tf.layers.batch_normalization(net, training=self.is_training))

            net = tf.layers.dense(relu1, 128 * 7 * 7)
            relu2 = tf.nn.relu(tf.layers.batch_normalization(net, training=self.is_training))

            net = tf.reshape(relu2, [-1, 7, 7, 128])
            deconv1 = tf.layers.conv2d_transpose(net, 128, [4, 4], strides=[2, 2], padding="same",
                                                 kernel_initializer=w_init, bias_initializer=b_init)
            deconv_relu1 = tf.nn.relu(tf.layers.batch_normalization(deconv1, training=self.is_training))

            deconv2 = tf.layers.conv2d_transpose(deconv_relu1, 1, [4, 4], strides=(2, 2), padding='same',
                                                 kernel_initializer=w_init, bias_initializer=b_init)
            output = tf.nn.sigmoid(deconv2)
            return output

    def get_vars(self):
        all_vars = tf.trainable_variables()
        g_vars = [var for var in all_vars if var.name.startswith("generator")]
        d_vars = [var for var in all_vars if var.name.startswith("discriminator")]
        return g_vars,d_vars

    def loss(self,real_recon_error,fake_recon_error):
        #BEGAN结合ENGAN和WGAN,使用自编码器作为判别器,并使用Wassertein距离损失优化重构误差
        d_loss = real_recon_error - self.k * fake_recon_error
        g_loss = fake_recon_error

        #BEGAN中提出的一个收敛指标:越小表示网络训练的越好,最终的生成图像质量越高
        #gamma代表生成图像的重构误差和真实图像重构误差的比例,同时其越高,
        #在优化损失中生成图像的损失占比就越高,故其也可以代表生成图像的多样性比例
        #k:用来影响生成图像重构误差损失的梯度
        #gamma:用更新K
        metric = real_recon_error + tf.abs(self.gamma * real_recon_error - fake_recon_error)

        #跟新K,供下一次计算损失使用
        update_k = self.k.assign(
            tf.clip_by_value(self.k + self.lamda * (self.gamma * real_recon_error
                                                    - fake_recon_error), 0, 1))
        return g_loss,d_loss,metric,update_k



