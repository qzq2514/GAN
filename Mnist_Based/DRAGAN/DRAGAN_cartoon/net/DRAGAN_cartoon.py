import tensorflow as tf
class DRAGAN_cartoon:
    def __init__(self,is_training,batch_size,image_width,
                 image_height,image_channal,lambd):
        self.lambd = lambd
        self.batch_size = batch_size
        self.image_width = image_width
        self.is_training = is_training
        self.image_height = image_height
        self.image_channal = image_channal

    def leakyReLU(self,inputs,leak=0.01):
        return tf.maximum(leak * inputs, inputs)

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
            deconv4 = tf.layers.conv2d_transpose(deconv_relu3, self.image_channal, [4, 4], strides=[2, 2], padding="same",
                                                 kernel_initializer=w_init, bias_initializer=b_init)
            output = tf.nn.tanh(deconv4)
            return output

    # inputs:[None,image_height,image_width,1]
    def discriminator(self, inputs, reuse=False):
        with tf.variable_scope("discriminator", reuse=reuse):
            w_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
            b_init = tf.constant_initializer(0.0)

            # [None,32,32,32]
            net = tf.layers.conv2d(inputs, 32, [4, 4], strides=[2, 2], padding="same",
                                   kernel_initializer=w_init, bias_initializer=b_init)
            net = tf.nn.relu(net)
            # [None,16,16,64]
            net = tf.layers.conv2d(net, 64, [4, 4], strides=[2, 2], padding="same",
                                   kernel_initializer=w_init, bias_initializer=b_init)
            net = tf.contrib.layers.batch_norm(net, decay=0.9, updates_collections=None, epsilon=1e-5,
                                               scale=True, is_training=self.is_training)
            net = tf.nn.relu(net)
            # [None,8,8,128]
            net = tf.layers.conv2d(net, 128, [4, 4], strides=[2, 2], padding="same",
                                   kernel_initializer=w_init, bias_initializer=b_init)
            net = tf.contrib.layers.batch_norm(net, decay=0.9, updates_collections=None, epsilon=1e-5,
                                               scale=True, is_training=self.is_training)
            net = tf.nn.relu(net)

            # 亲测用上全连接效果会更好一点
            net = tf.nn.relu(net)
            net = tf.layers.flatten(net)
            net = tf.layers.dense(net, 2048)
            net = tf.contrib.layers.batch_norm(net, decay=0.9, updates_collections=None, epsilon=1e-5,
                                               scale=True, is_training=self.is_training)
            net = tf.nn.relu(net)
            logits = tf.layers.dense(net, 1)
            output = tf.nn.sigmoid(logits)

            return output, logits

    def get_vars(self):
        all_vars = tf.trainable_variables()
        g_vars = [var for var in all_vars if var.name.startswith("generator")]
        d_vars = [var for var in all_vars if var.name.startswith("discriminator")]
        return g_vars,d_vars

    #给数据加上噪声
    def get_perturbed_batch_tf(self, real_image):
        mean, variance = tf.nn.moments(real_image,axes=[0,1,2])
        inputs_shape = [self.batch_size]+real_image.get_shape().as_list()[1:]
        return real_image + 0.5 * tf.sqrt(variance) * tf.random_normal(shape=inputs_shape)

    #传入判别器的prob形式,而非logits形式
    def loss(self,y_data,y_generated,real_image):

        #DRAGAN中除了梯度惩罚项以外的判别器损失和生成器损失是与原始的GAN或者DCGAN一样的, 不使用Wassertein距离
        d_loss = - tf.reduce_mean(tf.log(tf.clip_by_value(y_data, 1e-8, 1.0))
                                  + tf.log(tf.clip_by_value(1 - y_generated, 1e-8, 1.0)))
        g_loss = - tf.reduce_mean(tf.log(tf.clip_by_value(y_generated, 1e-8, 1.0)))

        #虽然DRAGAN中也使用了梯度惩罚,但是注意的是:原WGAN-GP中的梯度惩罚,是先计算真实图像和生成图像的的插值图像
        #然后使用判别器计算插值图像的得分,然后该得分对对插值图像的梯度,才是WGAN-GP中的梯度惩罚中的"梯度"
        #而DRAGAN中的梯度,是不使用生成图像,而是先对真实图像加噪声得到真实噪声图像(即这里的perturbed_image),
        #然后计算该真实噪声图像和真实图像的插值,之后步骤和WGAN-GP一样
        perturbed_image = self.get_perturbed_batch_tf(real_image)
        alpha = tf.random_uniform(shape=[self.batch_size]+real_image.get_shape().as_list()[1:], minval=0., maxval=1.)
        differences = perturbed_image - real_image
        interpolates = real_image + (alpha * differences)
        _, D_inter_logits = self.discriminator(interpolates, reuse=True)
        gradients = tf.gradients(D_inter_logits, [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
        # lambd:梯度惩罚在总体损失中的权重
        d_loss += self.lambd * gradient_penalty

        return g_loss,d_loss


