import tensorflow as tf
import os
from tensorflow.python.framework import graph_util
import matplotlib.pyplot as plt
from net import cGAN_mnist
import cv2
import time
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


class_num=10
image_flatten_size = 784
prior_size=100
batch_size = 64
epochs = 2000
learning_rate = 0.001
discriminator_train_intervel=1
smooth=0.1
snapshot = 100
one_hot=np.eye(10)

image_path = "../Data/MNIST_fashion"
model_path="models_fashion/"
pb_path=os.path.join(model_path,"pb/")
ckpt_path=os.path.join(model_path,"ckpt/")
result_dir="result"
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
if not os.path.exists(pb_path):
    os.makedirs(pb_path)
if not os.path.exists(ckpt_path):
    os.makedirs(ckpt_path)
model_name="cGAN_fashion"

def train():

    real_data_placeholder = tf.placeholder(tf.float32,shape=(None,image_flatten_size),name="real_data")
    z_prior_placeholder = tf.placeholder(tf.float32,shape=(None,prior_size),name="z_prior")
    label_placeholder = tf.placeholder(tf.float32,shape=(None,class_num),name="label")

    mnist_GANer=cGAN_mnist.cGAN_mnist(image_flatten_size,smooth)

    generated_data = mnist_GANer.generator(z_prior_placeholder,label_placeholder)
    _generated_data = tf.identity(generated_data,name="generated_output")

    real_prop,real_logist = mnist_GANer.\
        discriminator(real_data_placeholder,label_placeholder)
    generated_prop,generated_logits = mnist_GANer.\
        discriminator(generated_data,label_placeholder,reuse=True)

    g_vars, d_vars = mnist_GANer.get_vars()
    g_loss, d_loss = mnist_GANer.loss(real_prop, generated_prop)
    # g_loss, d_loss = mnist_GANer.loss_tf(real_logist,generated_logits)

    global_step = tf.Variable(0, trainable=False)
    #生成器和判别器各自更新自己的参数
    g_train_opt = tf.train.AdamOptimizer(learning_rate).\
        minimize(g_loss,global_step=global_step, var_list = g_vars)
    d_train_opt = tf.train.AdamOptimizer(learning_rate).\
        minimize(d_loss,global_step=global_step, var_list = d_vars)

    saver = tf.train.Saver()
    mnist = input_data.read_data_sets(image_path)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        ckpt = tf.train.get_checkpoint_state(ckpt_path)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join(ckpt_path, ckpt_name))

        step_per_epoch = mnist.train.num_examples // batch_size
        start_time = time.time()

        for _ in range(epochs):
            for _ in range(mnist.train.num_examples//batch_size):
                batch = mnist.train.next_batch(batch_size)
                batch_images = batch[0].reshape((batch_size,image_flatten_size))
                batch_labels = one_hot[np.array(batch[1])]  #注意要变成one-hot形式

                # 将像素值转到-1~1范围,这很有必要,不然后期生成效果很差
                batch_images = batch_images*2-1
                batch_z_prior = np.random.normal(0,1,size=(batch_size,prior_size))

                #更新判别器
                discriminator_dict={real_data_placeholder:batch_images,
                                    z_prior_placeholder:batch_z_prior,
                                    label_placeholder:batch_labels}

                global_step_,train_loss_d,_ = sess.run([global_step,d_loss,d_train_opt],
                                          feed_dict=discriminator_dict)
                # 更新生成器
                batch_z_prior = np.random.normal(0, 1, (batch_size, prior_size))
                # 随机赋予标签,要保证每个标签都被充分地用过
                # 本人做过实验,如果有一个标签一直没有出现过,那么最终该标签下的数据生成效果极差
                # 甚至如果每次这里都赋予一个固定的标签,如每次都赋值0标签,那么就不会生成其他标签的数据
                # 可自行尝试cGAN_mnist_noLabel-300.pb
                batch_labels = np.random.randint(0, 10, (batch_size, 1))
                batch_labels =one_hot[batch_labels].squeeze()

                generator_dict={z_prior_placeholder:batch_z_prior,
                                label_placeholder: batch_labels}
                train_loss_g, _ = sess.run([g_loss, g_train_opt],
                                           feed_dict=generator_dict)

            epoch_num = int(global_step_ / (mnist.train.num_examples // batch_size))//2
            print("Epoch:[{}]  ,time:{} s ,D_Loss: {:.4f} ,G_Loss: {:.4f}".format(
                epoch_num, step_per_epoch,
                time.time() - start_time, train_loss_d, train_loss_g))

            if epoch_num % snapshot == 0:
                # 显示中间结果
                samples = sess.run(generated_data, feed_dict={z_prior_placeholder: batch_z_prior,
                                                              label_placeholder: batch_labels})
                for ind in range(len(samples))[:20]:
                    image_reshape_org = samples[ind].reshape((28, 28))
                    image_reshape = (image_reshape_org + 1) / 2 * 255.0
                    cv2.imwrite(result_dir + "/epoch{}_{}_{}.jpg".format(
                        epoch_num, np.argmax(batch_labels[ind]), ind), image_reshape)

                # 保存PB
                constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ["generated_output"])
                save_model_name = model_name + "-" + str(epoch_num) + ".pb"
                with tf.gfile.FastGFile(pb_path + save_model_name, mode="wb") as fw:
                    fw.write(constant_graph.SerializeToString())
                # 保存CKPT
                saver.save(sess, ckpt_path + model_name + ".ckpt", global_step=epoch_num)
                print("Successfully saved model {}".format(save_model_name))

if __name__ == '__main__':
    train()