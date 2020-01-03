import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

class CycleGAN:
    def __init__(self,is_training,keep_prob,lambda_reconst):
        self.is_training = is_training
        self.epsilon = 1e-5
        self.weight_decay = 0.00001
        self.keep_prob = keep_prob
        self.lambda_reconst = lambda_reconst
        self.REAL_LABEL=0.9

    def preprocess(self,images,scale=False):
        images = tf.to_float(images)
        if scale:
            images = tf.div(images, 127.5)
            images = tf.subtract(images, 1.0)
        return images

    def c7s1_k(self,input,k,is_training,scope_name,isTanh=False):
    #A 7x7 Convolution-BatchNorm-ReLU layer with k filters and stride 1
        activation_func = tf.nn.tanh if isTanh else tf.nn.relu
        with tf.variable_scope(scope_name) as scope:
            padded = tf.pad(input,[[0,0],[3,3],[3,3],[0,0]],mode="REFLECT")
            net = slim.conv2d(padded,k,[7,7],[1,1],padding="VALID",activation_fn=None)
            net = self.instance_norm(net)
            output = activation_func(net)
            # net = slim.batch_norm(net,decay=0.9,is_training=is_training,epsilon=1e-5,
            #                       scale=True,updates_collections=None,activation_fn=activation_func)
            return output

    def dk(self,input,k,is_training,scope_name):
    # A 3x3 Convolution-BatchNorm-ReLU layer with k filters and stride 2
        with tf.variable_scope(scope_name) as scope:
            net = slim.conv2d(input, k, [3, 3], [2, 2], padding="SAME", activation_fn=None)
            net = self.instance_norm(net)
            net = tf.nn.relu(net)
            # net = slim.batch_norm(net, decay=0.9, is_training=is_training, epsilon=1e-5,
            #                     scale=True, updates_collections=None, activation_fn=tf.nn.relu)
            return net
    def Rk(self,input,k,is_training,scope_name):
    # A residual block that contains two 3x3 convolutional layers
    # with the same number of filters on both layer
        with tf.variable_scope(scope_name+"/layer1") as scope:
            padded1 = tf.pad(input,[[0,0],[1,1],[1,1],[0,0]],mode="REFLECT")
            conv1 = slim.conv2d(padded1,k,[3,3],[1,1],padding="VALID")
            norm1 = self.instance_norm(conv1)
            relu1 = tf.nn.relu(norm1)
            # norm1 = slim.batch_norm(conv1, decay=0.9, is_training=is_training, epsilon=1e-5,
            #                         scale=True, updates_collections=None, activation_fn=tf.nn.relu)
        with tf.variable_scope(scope_name+"/layer2") as scope:
            padded2 = tf.pad(norm1,[[0,0],[1,1],[1,1],[0,0]],mode="REFLECT")
            conv2 = slim.conv2d(padded2,k,[3,3],[1,1],padding="VALID")
            norm2 = self.instance_norm(conv2)
            # norm2 = slim.batch_norm(conv2, decay=0.9, is_training=is_training, epsilon=1e-5,
            #                         scale=True, updates_collections=None, activation_fn=None)
        return input+norm2

    #残差模块,不会改变特征图的宽高和通道数
    def n_res_block(self,input,n,is_training):
        channals = input.get_shape().as_list()[-1]
        for i in range(1,1+n):
            output = self.Rk(input,channals,is_training,"R{}_{}".format(channals,i))
            input = output
        return output

    def uk(self,input,k,is_training,scope_name):
    # A 3x3 fractional-strided-Convolution-BatchNorm-ReLU layer with k filters, stride 1/2
        with tf.variable_scope(scope_name) as scope:
            conv_tran = slim.conv2d_transpose(tf.nn.relu(input), k,[3,3],[2,2],padding="SAME")
            norm = self.instance_norm(conv_tran)
            output = tf.nn.relu(norm)
            # norm = slim.batch_norm(conv_tran, decay=0.9, is_training=is_training, epsilon=1e-5,
            #                     scale=True, updates_collections=None, activation_fn=tf.nn.relu)
        return output

    def Ck(self,input,k,is_training,scope_name,norm=True):
    #A 4x4 Convolution-BatchNorm-LeakyReLU layer with k filters and stride 2
        with tf.variable_scope(scope_name) as scope:
            net = slim.conv2d(input, k, [4, 4], [2, 2], padding="SAME", activation_fn=None)
            if norm:
            	net = self.instance_norm(net)
                # net = slim.batch_norm(net, decay=0.9, is_training=is_training, epsilon=1e-5,
                #         scale=True, updates_collections=None, activation_fn=tf.nn.leaky_relu)
            output = tf.nn.leaky_relu(net)
        return output

    def last_conv(self,inputs,use_sigmoid,scope_name):
    #Last convolutional layer of discriminator network  (1 filter with size 4x4, stride 1)
        with tf.variable_scope(scope_name) as scope:
            output = slim.conv2d(inputs, 1, [4, 4], [1, 1], padding="SAME", activation_fn=None)
            if use_sigmoid:
                output = tf.nn.sigmoid(output)
            return output

    def instance_norm(self,inputs):
        with tf.variable_scope("instance_norm"):
            epsilon = 1e-5
            mean, var = tf.nn.moments(inputs, [1, 2], keep_dims=True)
            scale = tf.get_variable('scale',[inputs.get_shape()[-1]], 
                initializer=tf.truncated_normal_initializer(mean=1.0, stddev=0.02))
            offset = tf.get_variable('offset',[inputs.get_shape()[-1]],initializer=tf.constant_initializer(0.0))
            out = scale*tf.div(inputs-mean, tf.sqrt(var+epsilon)) + offset
            return out

    #[None,64,64,3]-->[None,64,64,3]
    def generator(self,inputs,name_scope,reuse=False):
        print("CycleGAN_generator")
        with tf.variable_scope(name_scope,reuse=reuse) as scope:
            w_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
            with slim.arg_scope([slim.conv2d],weights_initializer=w_init,weights_regularizer=slim.l2_regularizer(self.weight_decay)):
                with slim.arg_scope([slim.conv2d_transpose],weights_initializer=w_init,weights_regularizer=slim.l2_regularizer(self.weight_decay)):
                    nfg = 32
                    #Encode
                    c7s1_32 = self.c7s1_k(inputs,nfg,is_training=self.is_training,scope_name="c7s1_32") #(?, 64, 64, nfg)
                    d64 = self.dk(c7s1_32,nfg*2,is_training=self.is_training,scope_name="d64") #(?, 32, 32, 2*nfg)
                    d128 = self.dk(d64,nfg*4,is_training=self.is_training,scope_name="d128") #(?, 16, 16, 4*nfg)
                    res_output = self.n_res_block(d128,n=6,is_training=self.is_training) #(?, 16, 16, 4*nfg)

                    # Decode
                    u64 = self.uk(res_output, nfg*2,is_training=self.is_training,scope_name="u64") #(?, 32, 32, nfg*2)
                    u32 = self.uk(u64, nfg, is_training=self.is_training, scope_name="u32") #(?, 64, 64, nfg)
                    generate_out = self.c7s1_k(u32, 3, is_training=self.is_training, scope_name="gen_output",isTanh=True) #(?, 64, 64, 3)
                    return generate_out

    # [None,64,64,3]-->[None,16,16,1]
    def discriminator(self,inputs,name_scope,reuse=False):
        print("CycleGAN_discriminator")
        with tf.variable_scope(name_scope,reuse=reuse) as scope:
            C64 = self.Ck(inputs, 64, is_training=self.is_training,scope_name="C64",norm=False) #(?, 32, 32, 64)
            C128 = self.Ck(C64, 128, is_training=self.is_training, scope_name="C128") #(?, 16, 16, 128)
            C256 = self.Ck(C128, 256, is_training=self.is_training, scope_name="C256") #(?, 8, 8, 256)
            C512 = self.Ck(C256, 512, is_training=self.is_training, scope_name="C512") #(?, 4, 4, 512)
            output = self.last_conv(C512,use_sigmoid=False,scope_name="dis_output") #(?, 4, 4, 1)
            return output

    def get_vars(self):
        all_vars = tf.trainable_variables()
        gen_A2B_vars = [var for var in all_vars if var.name.startswith("generator_AB")]
        gen_B2A_vars = [var for var in all_vars if var.name.startswith("generator_BA")]
        dis_A_vars = [var for var in all_vars if var.name.startswith("discriminator_A")]
        dis_B_vars = [var for var in all_vars if var.name.startswith("discriminator_B")]
        return gen_A2B_vars,gen_B2A_vars,dis_A_vars,dis_B_vars

    def build_DiscoGAN(self,input_A,input_B):
        #归一化
        input_A_pre = self.preprocess(input_A, scale=True)
        input_B_pre = self.preprocess(input_B, scale=True)

        #Domain A --> Domain B
        AB = self.generator(input_A_pre,"generator_AB")
        ABA = self.generator(AB,"generator_BA")

        AB_prob = self.discriminator(AB, "discriminator_B")
        B_prob = self.discriminator(input_B_pre, "discriminator_B", reuse=True)
        reconst_A_loss = tf.reduce_mean(tf.abs(ABA-input_A_pre))  # L1
        fake_Gen_AB_loss = tf.reduce_mean(tf.squared_difference(AB_prob, self.REAL_LABEL))
        fake_Dis_AB_loss = tf.reduce_mean(tf.square(AB_prob))
        real_Dis_B_loss  = tf.reduce_mean(tf.squared_difference(B_prob, self.REAL_LABEL))
        Dis_B_loss = fake_Dis_AB_loss + real_Dis_B_loss

        # Domain B --> Domain A
        BA  = self.generator(input_B_pre, "generator_BA",reuse=True)
        BAB  = self.generator(BA, "generator_AB",reuse=True)

        BA_porb = self.discriminator(BA, "discriminator_A")
        A_prob = self.discriminator(input_A_pre, "discriminator_A", reuse=True)
        reconst_B_loss = tf.reduce_mean(tf.abs(BAB-input_B_pre))
        fake_Gen_BA_loss = tf.reduce_mean(tf.squared_difference(BA_porb, self.REAL_LABEL))
        fake_Dis_BA_loss = tf.reduce_mean(tf.square(BA_porb))
        real_Dis_A_loss  = tf.reduce_mean(tf.squared_difference(A_prob, self.REAL_LABEL))
        Dis_A_loss = fake_Dis_BA_loss + real_Dis_A_loss

        #generate loss
        cycle_loss = reconst_A_loss+reconst_B_loss
        Gen_AB_loss = fake_Gen_AB_loss + self.lambda_reconst * cycle_loss
        Gen_BA_loss = fake_Gen_BA_loss + self.lambda_reconst * cycle_loss

        return Gen_AB_loss,Gen_BA_loss,Dis_A_loss,Dis_B_loss

    def sample_generate(self,input,type="A2B"):
        if type=="A2B":
            name_scope_first = "generator_AB"
            name_scope_second = "generator_BA"
        else:
            name_scope_first = "generator_BA"
            name_scope_second = "generator_AB"
        input_pre = self.preprocess(input, scale=True)
        generated_out = self.generator(input_pre,name_scope=name_scope_first,reuse=True)
        reconst_image = self.generator(generated_out,name_scope=name_scope_second,reuse=True)

        return generated_out,reconst_image



