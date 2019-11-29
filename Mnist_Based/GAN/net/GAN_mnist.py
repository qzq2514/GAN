import tensorflow as tf

class GAN_mnist:
    def __init__(self,endode_size,smooth):
        self.encode_size=endode_size
        self.smooth=smooth

    def generator(self,z_prior,reuse=False):
        with tf.variable_scope("generator",reuse=reuse):
            h1 = tf.layers.dense(z_prior,128,activation=None)
            h1 = tf.maximum(0.01*h1,h1)    #Leaky ReLU激活函数

            logits = tf.layers.dense(h1,self.encode_size,activation=None)
            encode_feature = tf.tanh(logits)

            return encode_feature

    def discriminator(self,inputs,reuse=False):
        with tf.variable_scope("discriminator", reuse=reuse):
            h1 = tf.layers.dense(inputs, 128, activation=None)
            h1 = tf.maximum(0.01*h1,h1)    #Leaky ReLU激活函数

            logits = tf.layers.dense(h1, 1, activation=None)
            prob = tf.sigmoid(logits)
            return prob,logits

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


