import tensorflow as tf

class EBGAN_mnist:
    def __init__(self,is_training,num_class,batch_size,
                 image_height,image_width,channels,PT_loss_weight,margin):
        self.is_training = is_training
        self.num_class = num_class
        self.batch_size = batch_size
        self.image_height = image_height
        self.image_width = image_width
        self.channels = channels
        self.PT_loss_weight = PT_loss_weight
        self.margin = margin

    def leakyReLU(self,inputs,leak=0.01):
        return tf.maximum(leak * inputs, inputs)

    def pullaway_loss(self, embeddings):
        # norm:[batch_size,1]每个数字表示对应行样本的元素平方和
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        # normalized_embeddings:[batch_size,code_dims]:每个样本各自归一化后的新矩阵
        normalized_embeddings = embeddings / norm
        # similarity:归一化后的新矩阵两两行样本相乘即可得两两行样本的余弦相似度
        similarity = tf.matmul(normalized_embeddings, normalized_embeddings, transpose_b=True)
        # 转为float32型
        batch_size = tf.cast(tf.shape(embeddings)[0], tf.float32)
        # 减去对角线的值(对角线上样本与自身的相似度,都是1)
        pt_loss = (tf.reduce_sum(similarity) - batch_size) / (batch_size * (batch_size - 1))
        return pt_loss

    # EBGAN的核心:在判别器中使用自编码器作为判别,计算重构后的图像和原输入头像的重构误差(MSE误差)
    # 真实图像的重构误差小,生成图像的重构误差大
    def discriminator(self, x, reuse=False):
        with tf.variable_scope("discriminator", reuse=reuse):
            w_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
            b_init = tf.constant_initializer(0.0)

            net = tf.layers.conv2d(x, 64, [4, 4], strides=[2, 2], padding="same",
                                   kernel_initializer=w_init, bias_initializer=b_init)
            net = tf.nn.relu(net)
            feature_height, feature_width = net.get_shape().as_list()[1:3]
            net = tf.layers.flatten(net)
            code = tf.layers.dense(net, 32)
            net = tf.layers.dense(code, feature_height * feature_width * 64)
            net = tf.contrib.layers.batch_norm(net, decay=0.9,updates_collections=None,
                                               epsilon=1e-5,scale=True,is_training=self.is_training)
            net = tf.nn.relu(net)
            net = tf.reshape(net, shape=[-1, feature_height, feature_width, 64])
            net = tf.layers.conv2d_transpose(net, 1, [4, 4], strides=[2, 2], padding="same",
                                             kernel_initializer=w_init, bias_initializer=b_init)
            #因为原图和生成的图像的像素值都是0~1范围,所以这里要用sigmod激活函数,不要用tanh
            #除非图像像素值全部统一到-1~1,就可以用tanh
            recon_out = tf.sigmoid(net)
            recon_error = tf.sqrt(2 * tf.nn.l2_loss(recon_out - x)) / self.batch_size
            # recon_error = tf.reduce_mean(tf.square(recon_out - x))
            return recon_out, recon_error, code

    def generator(self, z, reuse=False):
        with tf.variable_scope("generator", reuse=reuse):
            w_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
            b_init = tf.constant_initializer(0.0)

            channals_num = z.get_shape().as_list()
            reshaped = tf.reshape(z, shape=[-1, 1, 1, channals_num[-1]])

            deconv1 = tf.layers.conv2d_transpose(reshaped, 256, [7, 7], strides=[1, 1], padding="valid",
                                                 kernel_initializer=w_init, bias_initializer=b_init)
            lrelu1 = tf.nn.relu(tf.layers.batch_normalization(deconv1, training=self.is_training))

            deconv2 = tf.layers.conv2d_transpose(lrelu1, 64, [4, 4], strides=[2, 2], padding="same",
                                                 kernel_initializer=w_init, bias_initializer=b_init)
            lrelu2 = tf.nn.relu(tf.layers.batch_normalization(deconv2, training=self.is_training))

            deconv3 = tf.layers.conv2d_transpose(lrelu2, 1, [4, 4], strides=(2, 2), padding='same',
                                                 kernel_initializer=w_init, bias_initializer=b_init)
            # 因为原来图像的像素值被归一化到0~1之间,且因为后面判别的时候计算输入图和原图的mse,
            # 所以要保证mse的两张图像的像素值是相同范围的,sigmod是0~1,tanh是-1~1,所以 不要用tanh
            # 或者将像素像之前一样归一化到-1~1,然后配合tanh使用也是可以的
            output = tf.nn.sigmoid(deconv3)
            return output

    def get_vars(self):
        all_vars = tf.trainable_variables()
        g_vars = [var for var in all_vars if var.name.startswith("generator")]
        d_vars = [var for var in all_vars if var.name.startswith("discriminator")]
        return g_vars,d_vars

    def loss(self,real_recon_error,fake_recon_error,fake_code):
        #判别器要减小真实图像的重构误差,增大生成图像的重构误差
        d_loss_all = real_recon_error + tf.maximum(self.margin - fake_recon_error, 0)
        #生成器要减小生成图像的重构误差,同时有一个pullaway_loss损失,保证生成图像的多样性,避免遇到模型坍塌(model collapse)
        g_loss_all = fake_recon_error + self.PT_loss_weight * self.pullaway_loss(fake_code)
        return g_loss_all,d_loss_all



