import tensorflow as tf
import os
from tensorflow.python.framework import graph_util
import matplotlib.pyplot as plt
from net import WGAN_mnist
import cv2
import time
import numpy as np
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data

image_height = 28
image_width = 28
prior_size=62

class_num=10
channels=1
batch_size = 64
epochs = 2000
learning_rate = 0.0002
discriminator_train_intervel=1
clip_value=0.01
snapshot = 100
beta1 = 0.5

model_path="models/"
pb_path=os.path.join(model_path,"pb/")
ckpt_path=os.path.join(model_path,"ckpt/")
result_dir = "fashion_result"
if not os.path.exists(pb_path):
    os.makedirs(pb_path)
if not os.path.exists(ckpt_path):
    os.makedirs(ckpt_path)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
data_set_path = "MNIST_data"  # MNIST_fashion MNIST_data
model_name="WGAN_mnist"

def train():
    real_data_placeholder = tf.placeholder(tf.float32,shape=[None,image_height,image_width,channels],name="real_data")
    z_prior_placeholder = tf.placeholder(tf.float32,shape=[None,prior_size],name="z_prior")
    is_training_placeholder = tf.placeholder(tf.bool,shape=(),name="is_training")

    GANer=WGAN_mnist.WGAN_mnist(is_training_placeholder,class_num,batch_size,
                                image_height,image_width,channels,clip_value)

    generated_data = GANer.generator(z_prior_placeholder)
    _generated_data = tf.identity(generated_data,name="generated_output")

    real_prop,real_logist = GANer.discriminator(real_data_placeholder)
    generated_prop,generated_logits = GANer.discriminator(generated_data,reuse=True)

    #WGAN的核心之一:不使用使用判别器的sigmod形式计算损失
    g_loss, d_loss = GANer.loss(real_logist, generated_logits)
    g_vars, d_vars = GANer.get_vars()
    global_step = tf.Variable(0, trainable=False)

    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        g_train_opt = tf.train.AdamOptimizer(learning_rate,beta1=beta1).\
                          minimize(g_loss,global_step=global_step, var_list = g_vars)
        d_train_opt = tf.train.AdamOptimizer(learning_rate*5,beta1=beta1).\
                          minimize(d_loss, var_list = d_vars)

    #WGAN的核心之一:权重剪枝,将判别器的参数限制在一个范围内
    clip_d_vars = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in d_vars]

    #保存所有参数,以便后面进行断点重新训练
    saver = tf.train.Saver()
    mnist = input_data.read_data_sets(data_set_path)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        ckpt = tf.train.get_checkpoint_state(ckpt_path)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join(ckpt_path, ckpt_name))

        var_list = tf.trainable_variables()
        g_list = tf.global_variables()  # 从全局变量中获得batch norm的缩放和偏差
        bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
        bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
        var_list += bn_moving_vars


        slim.model_analyzer.analyze_vars(var_list, print_info=True)

        step_per_epoch = mnist.train.num_examples // batch_size
        start_time = time.time()

        for _ in range(epochs):
            is_break=False
            for _ in range(step_per_epoch):
                batch = mnist.train.next_batch(batch_size)

                # 原始数据:[128,784] 因为使用卷积形式的GAN,输入要变成一般图像形式:[128, 28, 28, 1]
                batch_images = batch[0].reshape((batch_size,image_height,image_width,channels))
                #和生成器的输出对应,生成器输出使用sigmod激活,所以这里元素范围在0~1
                batch_images = batch_images #*2-1
                batch_z_prior = np.random.uniform(-1, 1,[batch_size, prior_size]).astype(np.float32)

                #更新判别器
                feed_dict={real_data_placeholder:batch_images,
                                    z_prior_placeholder:batch_z_prior,
                                    is_training_placeholder:True}

                # WGAN的核心之一:在使用Wasserstein距离作为目标函数时,可以尽可能完美的优化判别器
                for _ in range(discriminator_train_intervel):
                    _,_,global_step_,train_loss_d = sess.run([d_train_opt,clip_d_vars,global_step,d_loss],
                                            feed_dict=feed_dict)

                train_loss_g, _ = sess.run([g_loss, g_train_opt],
                                           feed_dict=feed_dict)
                epoche_num = global_step_ // step_per_epoch
                step = global_step_ % step_per_epoch
                print("Epoch:[{}] ,Step:[{}/{}] ,time:{} s ,D_Loss: {:.4f} ,G_Loss: {:.4f}".format(
                    epoche_num, step, step_per_epoch,
                    time.time() - start_time, train_loss_d, train_loss_g))

                #使用batch_norm就要查看其中norm的均值或方差有没有稳定
                one_moving_meaning_show = "No mean or variance"
                if len(bn_moving_vars) > 0:
                    one_moving_meaning = sess.graph.get_tensor_by_name(bn_moving_vars[0].name)
                    one_moving_meaning_show = np.mean(one_moving_meaning.eval())
                print("one_moving_meaning:", one_moving_meaning_show)

                if global_step_ % snapshot == 0:
                    # 显示中间结果
                    samples = sess.run(generated_data, feed_dict={z_prior_placeholder: batch_z_prior,
                                                                  is_training_placeholder: False})
                    for ind in range(len(samples))[:20]:
                        image_reshape_org = samples[ind].reshape((image_height, image_width))
                        image_reshape = image_reshape_org * 255.0
                        cv2.imwrite(result_dir + "/epoch{}_{}_{}.jpg".format(
                            epoche_num, step, ind), image_reshape)

                    # 保存PB
                    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ["generated_output"])
                    save_model_name = model_name + "-" + str(global_step_) + ".pb"
                    with tf.gfile.FastGFile(pb_path + save_model_name, mode="wb") as fw:
                        fw.write(constant_graph.SerializeToString())
                    # 保存CKPT
                    saver.save(sess, ckpt_path + model_name + ".ckpt", global_step=global_step_)
                    print("Successfully saved model {}".format(save_model_name))

if __name__ == '__main__':
    train()