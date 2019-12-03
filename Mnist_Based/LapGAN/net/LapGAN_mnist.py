import tensorflow as tf
import numpy as np

#正宗的LapGAN,适用于MNIST上的包含三级、二级甚至一级的模式
#使用dropout效果会更好
class LapGAN_mnist:
    def __init__(self,real_data_placeholder,z_prior_placeholders,
                      label_placeholder,keep_prob,is_training,
                      z_priors_size,smooth):
        self.real_data = real_data_placeholder
        self.channal = self.real_data.get_shape().as_list()[-1]
        self.z_priors = z_prior_placeholders
        self.pyramid_levels = len(self.z_priors)
        self.smooth=smooth
        self.keep_prob=keep_prob
        self.is_training = is_training
        self.z_priors_size=z_priors_size
        self.batch_size ,self.class_num = label_placeholder.get_shape().as_list()
        self.label_placeholder=label_placeholder

        self.g_pyramid,self.corase_pyramid,self.l_pyramid=self.get_pyramid()

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

    def get_pyramid(self):
        # 高斯金字塔,分辨率逐渐加减半
        g_pyramid = []

        g_pyramid.append(self.real_data)
        for level in range(1,self.pyramid_levels):
            g_pyramid.append(self.sample(g_pyramid[level-1],"down"))

        # 低频金字塔,就是高斯金字塔每一层的下采样再上采样,最后一层不需要模糊,为None
        corase_pyramid=[]
        for pyramid_level in g_pyramid[1:]:
            corase_pyramid.append(self.sample(pyramid_level,"up"))
        corase_pyramid.append(None)

        l_pyramid = []
        for level in range(0,self.pyramid_levels-1):
            residual_img = g_pyramid[level] - corase_pyramid[level]
            l_pyramid.append(residual_img)
        l_pyramid.append(g_pyramid[-1])

        return g_pyramid,corase_pyramid,l_pyramid

    def generator(self,coarse,z_prior,label,generator_ind,reuse=False):
        sqrt_= int(np.sqrt(self.z_priors_size[generator_ind]))
        scope_name = self.z_priors_size[-1] if generator_ind==self.pyramid_levels-1 \
                                    else "{}x{}".format(sqrt_,sqrt_)
        scale_h,scale_w = self.g_pyramid[generator_ind].get_shape().as_list()[1:3]

        pixels_num = scale_h*scale_w*self.channal
        with tf.variable_scope("generator_{}".format(scope_name),reuse=reuse):
            w_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
            b_init = tf.constant_initializer(0.0)

            if coarse is None:   #最低分辨率,不用添加当前层coarse的低频图像,且使用全连接生成图像
                input_cat = tf.concat([z_prior,label],axis=1)

                # FC Layers
                net = tf.layers.dense(input_cat, 128, activation=tf.nn.leaky_relu, name='g-fc-1')
                # 使用dropout可以提高效果
                net = tf.layers.dropout(net, 0.5, name='g-dropout-1')
                net = tf.layers.dense(net, 128 // 2, activation=tf.nn.leaky_relu, name='g-fc-2')
                net = tf.layers.dropout(net, 0.5, name='g-dropout-2')
                net = tf.layers.dense(net, pixels_num, name='g-fc-3')

                output = tf.reshape(net, [-1, scale_h, scale_w, self.channal])

            else:
                label = tf.reshape(label, shape=[-1, 1, 1, self.class_num])
                label = label * tf.ones(shape=[self.batch_size, scale_h, scale_w, self.class_num])

                z_prior = tf.reshape(z_prior, shape=[-1, scale_h, scale_w, self.channal])

                cat = tf.concat([z_prior, label, coarse,], axis=3)
                net = tf.layers.conv2d(cat, 64 * 1, 5, 1, padding="same", name='gen-deconv2d-1')
                net = tf.nn.relu(net)
                net = tf.layers.conv2d(net, 64 * 1, 5, 1, padding="same", name='gen-deconv2d-2')
                net = tf.nn.relu(net)
                output = tf.layers.conv2d(net, self.channal, 5, 1, padding="same", name='gen-conv2d-3')

                # cat = tf.concat([z_prior, label, coarse ], axis=3)
                # for idx in range(0, scale_h // 14):
                #     net = tf.layers.conv2d(cat, 64 * 1, 5, strides=1, padding="same",
                #                            name='gen-deconv2d-{}'.format(idx))
                #     net = tf.nn.relu(net)
                # output = tf.layers.conv2d(net, self.channal, 5, 1, padding="same", name='gen-conv2d-final')

            # net=tf.nn.tanh(net)

            return output

    def discriminator(self,res_image,coarse,label,discriminator_ind,reuse=False):
        sqrt_ = int(np.sqrt(self.z_priors_size[discriminator_ind]))
        scope_name = self.z_priors_size[-1] if discriminator_ind == self.pyramid_levels - 1 else \
            "{}x{}".format(sqrt_, sqrt_)
        scale_h, scale_w = self.g_pyramid[discriminator_ind].get_shape().as_list()[1:3]

        pixels_num=scale_h*scale_w*self.channal
        with tf.variable_scope("discriminator_{}".format(scope_name), reuse=reuse):

            if coarse is None:   #同样最低分辨率,不用添加当前层coarse的低频图像
                res_image = tf.reshape(res_image,shape=[-1,pixels_num])
                input_cat = tf.concat([res_image,label],axis=1)

                net = tf.layers.dense(input_cat, 128, activation=tf.nn.leaky_relu, name='d-fc-1')
                net = tf.layers.dropout(net, 0.5, name='d-dropout-1')
                net = tf.layers.dense(net, 128 // 2, activation=tf.nn.leaky_relu, name='d-fc-2')
                net = tf.layers.dropout(net, 0.5, name='d-dropout-2')
                logits = tf.layers.dense(net, 1, name='d-fc-3')

            else:
                label = tf.reshape(label, shape=[-1, 1, 1, self.class_num])
                label = label * tf.ones(shape=[self.batch_size, scale_h, scale_w, self.class_num])

                input_cat = tf.concat([res_image+coarse, label],axis=3)
                net = tf.layers.conv2d(input_cat, 64, 5, 1,
                                       activation=tf.nn.leaky_relu, padding='valid')
                net = tf.layers.conv2d(net, 64, 5, 1, activation=None, padding='valid')
                net = tf.layers.flatten(net)
                net = tf.nn.leaky_relu(net)
                net = tf.layers.dropout(net, 0.5, name='d-dropout-1')
                logits = tf.layers.dense(net, 1, name='d-fc-2')


                # net = tf.layers.conv2d(input_cat, 64, 5, 1,
                #                        activation=tf.nn.leaky_relu, padding='valid')
                # net = tf.layers.conv2d(net, 64, 5, 1, activation=None, padding='valid')
                #
                # net = tf.layers.flatten(net)
                # net = tf.nn.leaky_relu(net)
                # net = tf.layers.dropout(net, 0.5, name='d-dropout-1')
                # logits = tf.layers.dense(net, 1, name='d-fc-2')

            prob=tf.nn.sigmoid(logits)

            return prob,logits

    def build_LapGAN(self):
        generators=[]
        d_fake=[]
        d_real=[]
        gen_residual_image=[]
        #各层的生成器和判别器都是独立训练的,
        #某一层的生成器就用前一层真实的模糊图像作为条件进行引导
        for ind in range(self.pyramid_levels):

            coarse_img=self.corase_pyramid[ind]
            lap_img=self.l_pyramid[ind]

            g_img=self.generator(coarse_img,self.z_priors[ind],self.label_placeholder,ind)
            d_fake_prob,d_fake_logits=self.discriminator(g_img,coarse_img,self.label_placeholder,ind)
            d_real_prob,d_real_logits=self.discriminator(lap_img,coarse_img,self.label_placeholder,
                                                         ind,reuse=True)
            generators.append(g_img)
            d_fake.append(d_fake_logits)
            d_real.append(d_real_logits)

        self.generators = generators
        self.d_fake = d_fake
        self.d_real = d_real
        return

    def get_vars(self):
        g_vars=[]
        d_vars=[]
        all_vars = tf.trainable_variables()
        for ind in range(self.pyramid_levels):

            sqrt_ = int(np.sqrt(self.z_priors_size[ind]))
            scope_name = self.z_priors_size[-1] if ind == self.pyramid_levels - 1 else \
                "{}x{}".format(sqrt_, sqrt_)

            cur_g_vars = [var for var in all_vars if var.name.startswith("generator_{}".
                                                                         format(scope_name))]
            cur_d_vars = [var for var in all_vars if var.name.startswith("discriminator_{}".
                                                                         format(scope_name))]
            g_vars.append(cur_g_vars)
            d_vars.append(cur_d_vars)
        return g_vars,d_vars

    def calcu_loss_tf(self,y_data,y_generated):
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

    def calcu_loss(self,real_prob,generated_prob):
        d_loss = - tf.reduce_mean((1-self.smooth)*tf.log(tf.clip_by_value(real_prob,1e-8,1.0))
                                  + tf.log(tf.clip_by_value(1 - generated_prob,1e-8,1.0)))
        g_loss = - tf.reduce_mean(tf.log(tf.clip_by_value(generated_prob,1e-8,1.0)))
        return g_loss,d_loss

    def loss(self):
        d_loss = []
        g_loss = []

        for ind , g_level in enumerate(self.g_pyramid):
            shape = g_level.get_shape().as_list()  # NHWC
            scale_h, scale_w = shape[1], shape[2]

            with tf.variable_scope("loss_{}x{}".format(scale_h, scale_w)):
                generated_logit = self.d_fake[ind]
                real_logit = self.d_real[ind]
                cur_g_loss, cur_d_loss = self.calcu_loss_tf(real_logit, generated_logit)

                g_loss.append(cur_g_loss)
                d_loss.append(cur_d_loss)
        return g_loss,d_loss

    def generate_Laplace(self):
        generated_images=[]

        cur_coarse_img=None
        for ind in range(self.pyramid_levels)[::-1]:
            #生成器生产残差图像
            genreated_res_img = self.generator(cur_coarse_img, self.z_priors[ind],
                                               self.label_placeholder,ind,reuse=True)
            if cur_coarse_img is None:  #最顶层的生成器直接生成模糊图像
                generated_images.append(genreated_res_img)
                cur_coarse_img = self.sample(genreated_res_img,"up")
            else:
                #其余层生成残差图像,要和对应的模糊图像相加才能得到好图像
                fine_img = cur_coarse_img + genreated_res_img
                generated_images.append(fine_img)
                cur_coarse_img = self.sample(fine_img,"up")

        return generated_images

