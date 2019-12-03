import tensorflow as tf

class DCGAN_mnist:
    def __init__(self,smooth,is_training):
        self.smooth=smooth
        self.is_training=is_training

    def leakyReLU(self,inputs,leak=0.01):
        return tf.maximum(leak * inputs, inputs)

    def sample(self,input,type="down"):
        shape=input.get_shape().as_list()  #NHWC
        if(type=="down"):
            h = int(shape[1]//2)
            w = int(shape[1] // 2)
        else:
            h = int(shape[1] * 2)
            w = int(shape[1] * 2)
        resized_image = tf.image.resize_images(input, [h, w],
                                               tf.image.ResizeMethod.BILINEAR)
        return resized_image

    #制造模糊图像
    def get_corase_img(self,input):
        down_sample = self.sample(input,"down")
        corase_sample = self.sample(down_sample,"up")
        return corase_sample

    def generator(self,corase_img,z_prior,reuse=False):
        with tf.variable_scope("generator",reuse=reuse):

            cat = tf.concat([z_prior,corase_img],axis=3)

            net = tf.layers.conv2d(cat, 64 * 1, 5, 1, padding="same",name='gen-deconv2d-1')
            net = tf.nn.relu(net)
            net = tf.layers.conv2d(net, 64 * 1, 5, 1, padding="same",name='gen-deconv2d-2')
            net = tf.nn.relu(net)
            output = tf.layers.conv2d(net, 1, 5, 1,padding="same", name='gen-conv2d-3')
            # output = tf.nn.tanh(output)
            return output

    def discriminator(self,inputs,reuse=False):
        with tf.variable_scope("discriminator", reuse=reuse):
            with tf.variable_scope("generator", reuse=reuse):

                cat = tf.concat([inputs], axis=3)
                net = tf.layers.conv2d(cat, 64, 5, 1,
                                       activation=tf.nn.leaky_relu, padding='valid')
                net = tf.layers.conv2d(net, 64, 5, 1, activation=None, padding='valid')

                net = tf.layers.flatten(net)
                net = tf.nn.leaky_relu(net)
                net = tf.layers.dropout(net, 0.5, name='d-dropout-1')

                net = tf.layers.dense(net, 1, name='d-fc-2')
                output = tf.nn.sigmoid(net)

            return output,net

    def get_vars(self):
        all_vars = tf.trainable_variables()
        g_vars = [var for var in all_vars if var.name.startswith("generator")]
        d_vars = [var for var in all_vars if var.name.startswith("discriminator")]
        return g_vars,d_vars

    def loss(self,y_data,y_generated):
        d_loss = - tf.reduce_mean((1-self.smooth)*tf.log(tf.clip_by_value(y_data,1e-8,1.0))
                                  + tf.log(tf.clip_by_value(1 - y_generated,1e-8,1.0)))
        g_loss = - tf.reduce_mean(tf.log(tf.clip_by_value(y_generated,1e-8,1.0)))
        return g_loss,d_loss

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


