import tensorflow as tf
import os
from tensorflow.python.framework import graph_util
import matplotlib.pyplot as plt
from net import GAN_mnist
import numpy as np
import time
from tensorflow.examples.tutorials.mnist import input_data
import cv2

image_flatten_size = 784
prior_size=100
batch_size = 128
epochs = 2000
learning_rate = 0.001
discriminator_train_intervel=1
smooth=0.1
snapshot = 100

image_path = "../Data/MNIST_fashion"
model_path="models_fashion/"
pb_path=os.path.join(model_path,"pb/")
ckpt_path=os.path.join(model_path,"ckpt/")
result_dir="result_fashion"
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
if not os.path.exists(pb_path):
    os.makedirs(pb_path)
if not os.path.exists(ckpt_path):
    os.makedirs(ckpt_path)
model_name="GAN_fashion"

def train():
    real_data_placeholder = tf.placeholder(tf.float32,shape=(None,image_flatten_size),name="real_data")
    z_prior_placeholder = tf.placeholder(tf.float32,shape=(None,prior_size),name="z_prior")

    mnist_GANer=GAN_mnist.GAN_mnist(image_flatten_size,smooth)

    generated_data = mnist_GANer.generator(z_prior_placeholder)
    _generated_data = tf.identity(generated_data,name="generated_output")

    real_prop,real_logist = mnist_GANer.discriminator(real_data_placeholder)
    generated_prop,generated_logits = mnist_GANer.discriminator(generated_data,reuse=True)

    g_vars, d_vars = mnist_GANer.get_vars()

    g_loss, d_loss = mnist_GANer.loss(real_prop, generated_prop)
    # g_loss, d_loss = mnist_GANer.loss_tf(real_logist,generated_logits)

    global_step = tf.Variable(0, trainable=False)
    #生成器和判别器各自更新自己的参数
    g_train_opt = tf.train.AdamOptimizer(learning_rate).\
        minimize(g_loss,global_step=global_step, var_list = g_vars)
    d_train_opt = tf.train.AdamOptimizer(learning_rate).\
        minimize(d_loss, var_list = d_vars)

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
                # 将像素值转到-1~1范围,这很有必要,不然后期生成效果很差
                batch_images = batch_images*2-1

                batch_z_prior = np.random.uniform(-1,1,size=(batch_size,prior_size))

                discriminator_dict={real_data_placeholder:batch_images,
                                    z_prior_placeholder:batch_z_prior}
                generator_dict={z_prior_placeholder:batch_z_prior}

                for _ in range(discriminator_train_intervel):
                    _ = sess.run(d_train_opt,feed_dict=discriminator_dict)
                _ = sess.run(g_train_opt,feed_dict=generator_dict)

            global_step_,train_loss_d = sess.run([global_step,d_loss], feed_dict=discriminator_dict)
            train_loss_g = sess.run(g_loss, feed_dict=generator_dict)

            epoch_num = int(global_step_/(mnist.train.num_examples//batch_size))
            print("Global_step:{},Epoch:[{}]  ,time:{} s ,D_Loss: {:.4f} ,G_Loss: {:.4f}".format(
                global_step_, epoch_num, step_per_epoch,
                time.time() - start_time, train_loss_d, train_loss_g))

            if epoch_num % snapshot == 0:
                # 显示中间结果
                samples = sess.run(generated_data, feed_dict={z_prior_placeholder: batch_z_prior})
                for ind in range(len(samples))[:20]:
                    image_reshape_org = samples[ind].reshape((28, 28))
                    image_reshape = (image_reshape_org + 1) / 2 * 255.0
                    cv2.imwrite(result_dir + "/epoch{}_{}.jpg".format(epoch_num, ind), image_reshape)

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