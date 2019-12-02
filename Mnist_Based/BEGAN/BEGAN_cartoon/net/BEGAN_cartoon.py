import tensorflow as tf

class BEGAN_cartoon:
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

            #[None,32,32,32]
            net = tf.layers.conv2d(inputs, 32, [5, 5], strides=[2, 2], padding="same",
                                   kernel_initializer=w_init, bias_initializer=b_init)
            # [None,16,16,64]
            net = tf.layers.conv2d(net, 64, [5, 5], strides=[2, 2], padding="same",
                                   kernel_initializer=w_init, bias_initializer=b_init)
            net = tf.contrib.layers.batch_norm(net, decay=0.9, updates_collections=None,
                                               epsilon=1e-5, scale=True, is_training=self.is_training)
            net = tf.nn.relu(net)
            # [None,8,8,128]
            net = tf.layers.conv2d(net, 128, [5, 5], strides=[2, 2], padding="same",
                                   kernel_initializer=w_init, bias_initializer=b_init)
            net = tf.contrib.layers.batch_norm(net, decay=0.9, updates_collections=None,
                                               epsilon=1e-5, scale=True, is_training=self.is_training)
            net = tf.nn.relu(net)

            feature_height, feature_width = net.get_shape().as_list()[1:3]

            net = tf.layers.flatten(net)
            #[None,128]
            code = tf.layers.dense(net, 256)
            net = tf.contrib.layers.batch_norm(code, decay=0.9, updates_collections=None,
                                               epsilon=1e-5, scale=True, is_training=self.is_training)
            net = tf.nn.relu(net)
            # [None,8*8*128]
            net = tf.layers.dense(net, feature_height * feature_width * 128)
            net = tf.contrib.layers.batch_norm(net, decay=0.9,updates_collections=None,
                                               epsilon=1e-5,scale=True,is_training=self.is_training)
            net = tf.nn.relu(net)
            # [None,8,8,128]
            net = tf.reshape(net, shape=[-1, feature_height, feature_width, 128])
            # [None,16,16,64]
            net = tf.layers.conv2d_transpose(net, 64, [4, 4], strides=[2, 2], padding="same",
                                             kernel_initializer=w_init, bias_initializer=b_init)
            net = tf.contrib.layers.batch_norm(net, decay=0.9, updates_collections=None,
                                               epsilon=1e-5, scale=True, is_training=self.is_training)
            net = tf.nn.relu(net)
            # [None,32,32,32]
            net = tf.layers.conv2d_transpose(net, 32, [4, 4], strides=[2, 2], padding="same",
                                             kernel_initializer=w_init, bias_initializer=b_init)
            net = tf.contrib.layers.batch_norm(net, decay=0.9, updates_collections=None,
                                               epsilon=1e-5, scale=True, is_training=self.is_training)
            net = tf.nn.relu(net)

            # [None,64,64,3]
            net = tf.layers.conv2d_transpose(net,self.channels, [4, 4], strides=[2, 2], padding="same",
                                             kernel_initializer=w_init, bias_initializer=b_init)
            recon_out = tf.nn.tanh(net)
            recon_error = tf.sqrt(2 * tf.nn.l2_loss(recon_out - inputs)) / self.batch_size
            return recon_out, recon_error, code

    def generator(self, z_prior, reuse=False):
        with tf.variable_scope("generator", reuse=reuse):
            w_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
            b_init = tf.constant_initializer(0.0)

            #[None,1024]
            net = tf.layers.dense(z_prior, 1024)
            relu1 = tf.nn.relu(tf.layers.batch_normalization(net, training=self.is_training))

            # [None,128 * 4 * 4]
            net = tf.layers.dense(relu1, 128 * 4 * 4)
            relu2 = tf.nn.relu(tf.layers.batch_normalization(net, training=self.is_training))

            # [None,4,4,128]
            net = tf.reshape(relu2, [-1, 4, 4, 128])

            # [None,8,8,128]
            deconv1 = tf.layers.conv2d_transpose(net, 128, [4, 4], strides=[2, 2], padding="same",
                                                 kernel_initializer=w_init, bias_initializer=b_init)
            deconv_relu1 = tf.nn.relu(tf.layers.batch_normalization(deconv1, training=self.is_training))
            # [None,16,16,64]
            deconv2 = tf.layers.conv2d_transpose(deconv_relu1, 64, [4, 4], strides=[2, 2], padding="same",
                                                 kernel_initializer=w_init, bias_initializer=b_init)
            deconv_relu2 = tf.nn.relu(tf.layers.batch_normalization(deconv2, training=self.is_training))
            # [None,32,32,32]
            deconv3 = tf.layers.conv2d_transpose(deconv_relu2, 32, [4, 4], strides=[2, 2], padding="same",
                                                 kernel_initializer=w_init, bias_initializer=b_init)
            deconv_relu3 = tf.nn.relu(tf.layers.batch_normalization(deconv3, training=self.is_training))
            # [None,64,64,3]
            deconv4 = tf.layers.conv2d_transpose(deconv_relu3, self.channels, [4, 4], strides=[2, 2], padding="same",
                                                 kernel_initializer=w_init, bias_initializer=b_init)
            output = tf.nn.tanh(deconv4)
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



