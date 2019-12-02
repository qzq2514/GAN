import tensorflow as tf

#WGAN相比较于之前一般GAN的重要改变:
'''
1.最后一层抛弃sigmod,即使用卷积或全连接出来的值直接计算损失
2.目标函数中不使用log
3.权重剪枝:每次更新后把权重限制在某个范围(一般是-0.01~0.01)
4.可以尽可能的把判别器训练的完美,即训练一次生成器就多训练几次判别器
'''
class WGAN_mnist:
    def __init__(self,is_training,num_class,batch_size,
                 image_width,image_height,image_channal,clip_value):
        self.is_training=is_training
        self.num_class=num_class
        self.clip_value=clip_value
        self.batch_size=batch_size
        self.image_width=image_width
        self.image_height=image_height
        self.image_channal=image_channal

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

    def loss(self,y_data,y_generated):

        #WGAN的核心之一:使用Wasserstein距离作为优化函数,优化目标中不使用log
        d_loss_real = - tf.reduce_mean(y_data)
        d_loss_fake = tf.reduce_mean(y_generated)

        d_loss = d_loss_real + d_loss_fake
        g_loss = - d_loss_fake
        return g_loss,d_loss


