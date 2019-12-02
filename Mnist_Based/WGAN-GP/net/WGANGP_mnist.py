import tensorflow as tf

class WGANGP_mnist:
    def __init__(self,is_training,num_class,batch_size,
                 image_width,image_height,image_channal,lambd):
        self.lambd = lambd
        self.num_class = num_class
        self.batch_size = batch_size
        self.image_width = image_width
        self.is_training = is_training
        self.image_height = image_height
        self.image_channal = image_channal

    def leakyReLU(self,inputs,leak=0.01):
        return tf.maximum(leak * inputs, inputs)

    #z_prior:[None,100]
    #label:[None,]
    def generator(self,z_prior,reuse=False):
        with tf.variable_scope("generator", reuse=reuse):
            w_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
            b_init = tf.constant_initializer(0.0)

            # 亲测用上全连接效果会更好一点
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
            out = tf.nn.sigmoid(deconv2)

            return out

    #inputs:[None,image_height,image_width,1]
    def discriminator(self,inputs,reuse=False):
        with tf.variable_scope("discriminator", reuse=reuse):
            w_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
            b_init = tf.constant_initializer(0.0)

            net = tf.layers.conv2d(inputs, 64, [4, 4], strides=[2, 2], padding="same",
                                   kernel_initializer=w_init, bias_initializer=b_init)
            net = tf.nn.relu(net)

            net = tf.layers.conv2d(net, 128, [4, 4], strides=[2, 2], padding="same",
                                   kernel_initializer=w_init, bias_initializer=b_init)

            net = tf.contrib.layers.batch_norm(net, decay=0.9, updates_collections=None, epsilon=1e-5,
                                               scale=True, is_training=self.is_training)
            #亲测用上全连接效果会更好一点
            net = tf.nn.relu(net)
            net = tf.layers.flatten(net)

            net = tf.layers.dense(net, 1024)
            net = tf.contrib.layers.batch_norm(net, decay=0.9, updates_collections=None, epsilon=1e-5,
                                               scale=True, is_training=self.is_training)
            net = tf.nn.relu(net)
            logits = tf.layers.dense(net, 1)
            output = tf.nn.sigmoid(logits)

            return output, logits,

    def get_vars(self):
        all_vars = tf.trainable_variables()
        g_vars = [var for var in all_vars if var.name.startswith("generator")]
        d_vars = [var for var in all_vars if var.name.startswith("discriminator")]
        return g_vars,d_vars

    def loss(self,y_data,y_generated,real_image,generated_image):

        """WGAN-GP的核心: 使用梯度惩罚替代简单粗暴的权重剪枝,论文中使用生成图像和真实图像的线性插值作为生成图像,
        这里对于一般的GAN损失还是使用原本的生成图像,而计算后面的梯度惩罚时使用的线性插值后的图像"""

        d_loss_real = - tf.reduce_mean(y_data)
        d_loss_fake = tf.reduce_mean(y_generated)
        d_loss = d_loss_real + d_loss_fake
        g_loss = - d_loss_fake


        alpha = tf.random_uniform(shape=[self.batch_size]+real_image.get_shape().as_list()[1:], minval=0., maxval=1.)
        differences = generated_image - real_image
        interpolates = real_image + (alpha * differences)  #模拟生成图像和真实图像的线性插值(每个位置一个权重参数)
        _, D_inter_logits = self.discriminator(interpolates, reuse=True)
        gradients = tf.gradients(D_inter_logits, [interpolates])[0]   #计算判别器权重对生成图像的权重梯度
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        # 使用论文中所说的“two-sided penalty”,将梯度限制在1的两侧, #(one-sided penalty)则是限制梯度小于1
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
        # lambd:梯度惩罚在总体损失中的权重
        d_loss += self.lambd * gradient_penalty

        return g_loss,d_loss


