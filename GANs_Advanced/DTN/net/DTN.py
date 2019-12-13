import tensorflow as tf
import tensorflow.contrib.slim as slim


class DTN(object):
    def __init__(self,is_training):
        self.is_training = is_training


    def preprocess(self,images,scale=False):
        images = tf.to_float(images)
        if scale:
            images = tf.div(images, 127.5)
            images = tf.subtract(images, 1.0)
        return images

    # pretrain:[None,32,32,3]--->[None,1,1,10]
    # train:[None,32,32,3]--->[None,1,1,128]
    def content_extractor(self,inputs,reuse=False,preTrain=False,is_training=True):
        if inputs.get_shape()[3] == 1:
            inputs = tf.image.grayscale_to_rgb(inputs)

        with tf.variable_scope("content_extractor",reuse=reuse):
            with slim.arg_scope([slim.conv2d],padding="SAME",activation_fn=None,stride=2,
                                weights_initializer=tf.contrib.layers.xavier_initializer()):
                with slim.arg_scope([slim.batch_norm],decay=0.95,center=True, scale=True,
                                    activation_fn=tf.nn.relu,is_training = is_training and self.is_training):
                    net = slim.conv2d(inputs, 64, [3,3])    #[None,16,16,64]
                    net = slim.batch_norm(net)
                    net = slim.conv2d(net, 128, [3, 3])    #[None,8,8,128]
                    net = slim.batch_norm(net)
                    net = slim.conv2d(net, 256, [3, 3])    #[None,4,4,256]
                    net = slim.batch_norm(net)
                    net = slim.conv2d(net, 128, [4, 4],padding = "VALID") #[None,1,1,128]
                    net = slim.batch_norm(net,activation_fn=tf.nn.tanh)

                    if preTrain:
                        net = slim.conv2d(net, 10, [1, 1],padding = "VALID")
                        net = slim.flatten(net)
                    return net

    # g:[None,1,1,128]--->[None,32,32,3]
    def reconstructor(self,inputs,reuse=False,is_training=True):
        with tf.variable_scope("reconstructor",reuse=reuse):
            with slim.arg_scope([slim.conv2d_transpose],padding="SAME",activation_fn=None,stride=2,
                                weights_initializer=tf.contrib.layers.xavier_initializer()):
                with slim.arg_scope([slim.batch_norm],decay=0.95,center=True, scale=True,
                                    activation_fn=tf.nn.relu,is_training = is_training and self.is_training):
                    net = slim.conv2d_transpose(inputs,512,[4,4],padding="VALID")
                    net = slim.batch_norm(net)
                    net = slim.conv2d_transpose(net, 256, [3, 3])
                    net = slim.batch_norm(net)
                    net = slim.conv2d_transpose(net, 128, [3, 3])
                    net = slim.batch_norm(net)
                    # 因为使用tanh激活函数,所以重构器图像输出范围为-1~1
                    net = slim.conv2d_transpose(net, 1, [3, 3],activation_fn=tf.nn.tanh)
                    return net


    # D:[None,32,32,1]--->[None,1]
    def discriminator(self,inputs,reuse=False):
        # if inputs.get_shape()[3] == 3:
        #     inputs = tf.image.rgb_to_grayscale(inputs)

        with tf.variable_scope("discriminator",reuse=reuse):
            with slim.arg_scope([slim.conv2d],padding="SAME",activation_fn=None,stride=2,
                                weights_initializer=tf.contrib.layers.xavier_initializer()):
                with slim.arg_scope([slim.batch_norm],decay=0.95,center=True, scale=True, 
                                    activation_fn=tf.nn.relu,is_training = self.is_training):
                    net = slim.conv2d(inputs,128,[3,3],activation_fn=tf.nn.relu)
                    net = slim.batch_norm(net)
                    net = slim.conv2d(net, 256, [3, 3])
                    net = slim.batch_norm(net)
                    net = slim.conv2d(net, 512, [3, 3])
                    net = slim.batch_norm(net)    
                    net = slim.conv2d(net, 1, [4, 4],padding="VALID") #activation_fn=tf.nn.sigmoid
                    net = slim.flatten(net)
                    return net

    def get_vars(self):
        all_vars = tf.trainable_variables()
        f_vars = [var for var in all_vars if var.name.startswith("content_extractor")]
        r_vars = [var for var in all_vars if var.name.startswith("reconstructor")]
        d_vars = [var for var in all_vars if var.name.startswith("discriminator")]
        return f_vars,r_vars, d_vars

    #预训练对源域进行分类,SVHN:元素范围0~255
    def build_model_preTrain(self,source_image_place,source_label_place):
        preprecossed = self.preprocess(source_image_place,scale=True) 
        print("preprecossed:",preprecossed)
        logits = self.content_extractor(preprecossed, reuse=False, preTrain=True)
        f_vars, r_vars, d_vars = self.get_vars()
        # print("len:",len(f_vars),len(r_vars),len(d_vars))

        preTrain_loss, preTrain_accu = self.preTrain_class_loss_accu(logits, source_label_place)
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(0.025, global_step, 150, 0.9)
        optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            preTrain_op = optimizer.minimize(preTrain_loss, global_step,var_list=f_vars)
        return preTrain_op,preTrain_loss,preTrain_accu

    def preTrain_class_loss_accu(self,logits,labels):
        loss = slim.losses.sparse_softmax_cross_entropy(logits=logits,labels=labels)

        pred_label = tf.argmax(logits,axis=1)
        correct = tf.equal(pred_label,labels)
        accu = tf.reduce_mean(tf.cast(correct,tf.float32))
        return loss,accu

    def build_model_train(self,source_image_place,target_image_place):
        source_image_place = self.preprocess(source_image_place,scale=True)
        target_image_place = self.preprocess(target_image_place)

        #source domain
        src_fx = self.content_extractor(source_image_place,reuse=True)
        src_fake_imgs = self.reconstructor(src_fx)
        src_logits = self.discriminator(src_fake_imgs)
        src_fgfx = self.content_extractor(src_fake_imgs,reuse=True)

        self.d_loss_src = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                     logits=src_logits,labels=tf.zeros_like(src_logits)))
        self.g_loss_src = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                     logits=src_logits,labels=tf.ones_like(src_logits)))
        self.f_loss_src = tf.reduce_mean(tf.square(src_fx-src_fgfx))*15.0

        f_vars, g_vars, d_vars = self.get_vars()

        learninr_rate = 0.0003
        with tf.variable_scope('source_train_op', reuse=False):
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.d_train_op_src = tf.train.AdamOptimizer(learninr_rate).\
                    minimize(self.d_loss_src, var_list=d_vars)
                self.g_train_op_src = tf.train.AdamOptimizer(learninr_rate).\
                    minimize(self.g_loss_src, var_list=g_vars)
                self.f_train_op_src =tf.train.AdamOptimizer(learninr_rate).\
                    minimize(self.f_loss_src, var_list=f_vars)

        #target domain
        trg_fx = self.content_extractor(target_image_place,reuse=True)
        trg_fake_imgs = self.reconstructor(trg_fx,reuse=True)
        trg_fake_logits = self.discriminator(trg_fake_imgs,reuse=True)
        print("target_image_place:",target_image_place)
        trg_real_logits = self.discriminator(target_image_place, reuse=True)

        trg_d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                          logits=trg_fake_logits,labels=tf.zeros_like(trg_fake_logits)))
        trg_d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                          logits=trg_real_logits,labels=tf.ones_like(trg_real_logits)))
        self.d_loss_trg = trg_d_loss_fake + trg_d_loss_real

        trg_g_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                          logits=trg_fake_logits,labels=tf.ones_like(trg_fake_logits)))
        trg_g_loss_reconst = tf.reduce_mean(tf.square(target_image_place-trg_fake_imgs))*15.0
        self.g_loss_trg = trg_g_loss_fake+trg_g_loss_reconst

        with tf.variable_scope('target_train_op', reuse=False):
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.d_train_op_trg = tf.train.AdamOptimizer(learninr_rate).minimize(self.d_loss_trg, var_list=d_vars)
                self.g_train_op_trg = tf.train.AdamOptimizer(learninr_rate).minimize(self.g_loss_trg, var_list=g_vars)
        return

    def generateor(self,source_image_place):
        source_image_place = self.preprocess(source_image_place,scale=True)
        fx = self.content_extractor(source_image_place, reuse=True,is_training=True)
        generated = self.reconstructor(fx,reuse=True,is_training=True)
        return generated







