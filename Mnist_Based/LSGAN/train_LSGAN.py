import tensorflow as tf
import os
from tensorflow.python.framework import graph_util
import matplotlib.pyplot as plt
from net import LSGAN_mnist
import cv2
import time
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

image_height = 28
image_width = 28
prior_size=100

class_num=10
channels=1
batch_size = 64
epochs = 2000
learning_rate = 0.001
discriminator_train_intervel=1
smooth=0.1
snapshot = 100
one_hot=np.eye(class_num)

model_path="models_fashion/"
image_path = "../Data/MNIST_fashion"
pb_path=os.path.join(model_path,"pb/")
ckpt_path=os.path.join(model_path,"ckpt/")
result_dir="result_fashion"
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
if not os.path.exists(pb_path):
    os.makedirs(pb_path)
if not os.path.exists(ckpt_path):
    os.makedirs(ckpt_path)
model_name="LSGAN_fashion"

def train():

    real_data_placeholder = tf.placeholder(tf.float32,shape=(None,image_height,image_width,channels),name="real_data")
    z_prior_placeholder = tf.placeholder(tf.float32,shape=(None,prior_size),name="z_prior")
    label_placeholder = tf.placeholder(tf.int32,shape=[None,],name="label")
    is_training_placeholder = tf.placeholder_with_default(False,shape=(),name="is_training")

    GANer=LSGAN_mnist.LSGAN_mnist(smooth,is_training_placeholder,class_num,
                                        batch_size,image_height,image_width,channels)

    generated_data = GANer.generator(z_prior_placeholder,label_placeholder)
    _generated_data = tf.identity(generated_data,name="generated_output")

    real_prop,real_logist = GANer.\
        discriminator(real_data_placeholder,label_placeholder)
    generated_prop,generated_logits = GANer.\
        discriminator(generated_data,label_placeholder,reuse=True)

    g_vars, d_vars = GANer.get_vars()
    #LSGAN中:使用logits计算损失,不要用sigmoid后的值计算损失
    g_loss, d_loss = GANer.loss(real_logist, generated_logits)
    # g_loss, d_loss = mnist_GANer.loss_tf(real_logist,generated_logits)

    global_step = tf.Variable(0, trainable=False)
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        g_train_opt = tf.train.AdamOptimizer(learning_rate,beta1=0.5).\
                          minimize(g_loss,global_step=global_step, var_list = g_vars)
        d_train_opt = tf.train.AdamOptimizer(learning_rate,beta1=0.5).\
                          minimize(d_loss, var_list = d_vars)

    saver = tf.train.Saver()   #只保存生成器的参数即可
    mnist = input_data.read_data_sets(image_path)
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

        step_per_epoch = mnist.train.num_examples // batch_size
        start_time = time.time()

        for _ in range(epochs):
            for _ in range(mnist.train.num_examples//batch_size):
                batch = mnist.train.next_batch(batch_size)

                # 原始数据:[128,784] 因为使用卷积形式的GAN,输入要变成一般图像形式:[128, 28, 28, 1]
                batch_images = batch[0].reshape((batch_size,image_height,image_width,channels))
                batch_labels = np.array(batch[1])
                # 将像素值转到-1~1范围,这很有必要,不然后期生成效果很差
                batch_images = batch_images*2-1
                batch_z_prior = np.random.normal(0,1,size=(batch_size,prior_size))

                #更新判别器
                discriminator_dict={real_data_placeholder:batch_images,
                                    z_prior_placeholder:batch_z_prior,
                                    label_placeholder:batch_labels,
                                    is_training_placeholder:True}

                global_step_,train_loss_d,_ = sess.run([global_step,d_loss,d_train_opt],
                                          feed_dict=discriminator_dict)
                # 更新生成器
                batch_z_prior = np.random.normal(0, 1, (batch_size,prior_size))

                # 随机赋予标签,要保证每个标签都被充分地用过
                batch_labels = np.random.randint(0, 10, (batch_size,))

                # 虽然生成器实际用不到real_data_placeholder和fill_label_placeholder,
                # 但是好像这里必须feed进去
                generator_dict={real_data_placeholder:batch_images,
                                z_prior_placeholder:batch_z_prior,
                                label_placeholder: batch_labels,
                                is_training_placeholder: True}
                train_loss_g, _ = sess.run([g_loss, g_train_opt],
                                           feed_dict=generator_dict)

                epoche_num = global_step_ // step_per_epoch
                step = global_step_ % step_per_epoch
                print("Epoch:[{}] ,Step:[{}/{}] ,time:{} s ,D_Loss: {:.4f} ,G_Loss: {:.4f}".format(
                    epoche_num, step, step_per_epoch, time.time() - start_time, train_loss_d, train_loss_g))

                #使用batch_norm就要查看其中norm的均值或方差有没有稳定
                one_moving_meaning_show = "No mean or variance"
                if len(bn_moving_vars) > 0:
                    one_moving_meaning = sess.graph.get_tensor_by_name(bn_moving_vars[0].name)
                    one_moving_meaning_show = np.mean(one_moving_meaning.eval())
                print("one_moving_meaning:", one_moving_meaning_show)

                if global_step_ % snapshot == 0:
                    # 显示中间结果
                    samples = sess.run(generated_data, feed_dict={z_prior_placeholder: batch_z_prior,
                                                                  label_placeholder: batch_labels,
                                                                  is_training_placeholder: False})
                    for ind in range(len(samples))[:20]:
                        image_reshape_org = samples[ind].reshape((image_height, image_width))
                        image_reshape = (image_reshape_org + 1) / 2 * 255.0
                        cv2.imwrite(result_dir + "/epoch{}_{}_{}_{}.jpg".format(
                            epoche_num, batch_labels[ind], step, ind), image_reshape)

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