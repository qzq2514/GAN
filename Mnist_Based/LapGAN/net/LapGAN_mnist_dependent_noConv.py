import tensorflow as tf
import numpy as np

#噪声为向量形式,各生成器之间独立训练,测试时联合生成数据,效果极差,几乎可以说无法用

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

        # 拉普拉斯金字塔,就是高斯金字塔每一层和上一层的上采样的差值,顶层和高斯金字塔顶层相同
        # 其实就是高斯金字塔和模糊金字塔对应层的差值(注意其最上一层就用高斯金字塔最上一层就行)
        l_pyramid = []
        for level in range(0,self.pyramid_levels-1):
            residual_img = g_pyramid[level] - corase_pyramid[level]
            l_pyramid.append(residual_img)
        l_pyramid.append(g_pyramid[-1])

        return g_pyramid,corase_pyramid,l_pyramid

    #coarse:金字塔当前层的低频图像 [batch_size,scale,scale,channal]
    #label:ont-hot形式标签  [batch_size,class_num]
    #z_prior:先验噪声  [batch_size,scale*scale*channal]
    #scale:最终二维图像大小
    #输出:当前生成器根据噪声输入,并以低频图像和标签信息作为条件生成图像[batch_size,scale,scale,channal]
    def generator(self,coarse,z_prior,label,generator_ind,reuse=False):
        sqrt_= int(np.sqrt(self.z_priors_size[generator_ind]))
        scope_name = self.z_priors_size[-1] if generator_ind==self.pyramid_levels-1 \
                                    else "{}x{}".format(sqrt_,sqrt_)
        scale_h,scale_w = self.g_pyramid[generator_ind].get_shape().as_list()[1:3]

        pixels_num = scale_h*scale_w*self.channal
        with tf.variable_scope("generator_{}".format(scope_name),reuse=reuse):
            if coarse is None:   #最低分辨率,不用添加当前层coarse的低频图像
                input_cat = tf.concat([z_prior,label],axis=1)
            else:       #其他层的生成器不仅要使用标签作为条件,还要使用当前层的coarse的低频图像作为条件
                # 因为使用全连接,所以这里要将每张图片数据展开成向量形式
                coarse = tf.reshape(coarse, shape=[-1, pixels_num])
                input_cat = tf.concat([z_prior,label ,coarse],axis=1)

            h1 = tf.layers.dense(input_cat, 128, activation=None)
            h1 = tf.maximum(0.01 * h1, h1)  # Leaky ReLU激活函数

            logits = tf.layers.dense(h1, pixels_num, activation=None)
            net = tf.tanh(logits)
            net = tf.reshape(net,shape=[-1,scale_h,scale_w,self.channal])

            return net

    #此处和原文有点有点不一样,原文中对真实和生成的残差图像,还有真实的模糊图像都进行判别
    #这里仅仅使用真实高斯图像与生成的残差图像和模糊图像相加构成的图像进行判别
    #res_image:生成器生成的残差图像或者真实的拉普拉斯图像-[batch_size,scale,scale,channal]
    #coarse:金字塔当前层的低频图像 [batch_size,scale,scale,channal]
    #label:ont-hot形式标签  [batch_size,class_num]
    def discriminator(self,res_image,coarse,label,discriminator_ind,reuse=False):
        sqrt_ = int(np.sqrt(self.z_priors_size[discriminator_ind]))
        scope_name = self.z_priors_size[-1] if discriminator_ind == self.pyramid_levels - 1 else \
            "{}x{}".format(sqrt_, sqrt_)
        scale_h, scale_w = self.g_pyramid[discriminator_ind].get_shape().as_list()[1:3]

        pixels_num=scale_h*scale_w*self.channal
        res_image = tf.reshape(res_image,shape=[-1, pixels_num])
        with tf.variable_scope("discriminator_{}".format(scope_name), reuse=reuse):
            if coarse is None:   #同样最低分辨率,不用添加当前层coarse的低频图像
                input_cat = tf.concat([res_image,label],axis=1)
            else:       #其他层的判别器要将
                # 因为使用全连接,所以这里要将每张图片数据展开成向量形式
                coarse = tf.reshape(coarse, shape=[-1, pixels_num])
                #残差图像和模糊图像相加得到精细的图像,再和one-hot形式的标签进行拼接后进行判别
                input_cat = tf.concat([res_image+coarse, label, ],axis=1)

            h1 = tf.layers.dense(input_cat, 128, activation=None)
            h1 = tf.maximum(0.01 * h1, h1)  # Leaky ReLU激活函数

            logits = tf.layers.dense(h1, 1, activation=None)
            prob = tf.sigmoid(logits)
            return prob,logits

    def build_LapGAN(self):
        generators=[]
        d_fake=[]
        d_real=[]

        #各层的生成器和判别器都是独立训练的,
        #某一层的生成器就用真实的模糊图像作为条件进行引导
        for ind in range(self.pyramid_levels):
            coarse_img=self.corase_pyramid[ind]
            lap_img=self.l_pyramid[ind]

            g_img=self.generator(coarse_img,self.z_priors[ind],self.label_placeholder,ind)
            d_fake_prob,d_fake_logits=self.discriminator(g_img,coarse_img,self.label_placeholder,ind)
            d_real_prob,d_real_logits=self.discriminator(lap_img,coarse_img,self.label_placeholder,
                                                         ind,reuse=True)
            generators.append(g_img)
            d_fake.append(d_fake_prob)
            d_real.append(d_real_prob)

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
                generated_prob = self.d_fake[ind]
                real_prob = self.d_real[ind]
                cur_g_loss, cur_d_loss = self.calcu_loss(real_prob, generated_prob)

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


