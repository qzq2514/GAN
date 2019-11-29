import tensorflow as tf
import os
from tensorflow.python.framework import graph_util
import matplotlib.pyplot as plt
from net import cGAN_mnist_14x14 as cGAN_mnist
import numpy as np
import cv2
from tensorflow.examples.tutorials.mnist import input_data


class_num=10
image_width_org = 28
image_height_org = 28
down_sample=2

gen_image_height=int(image_height_org/down_sample)
gen_image_width=int(image_width_org/down_sample)

image_flatten_size = gen_image_height*gen_image_width
prior_size=100
batch_size = 64
epochs = 2000
learning_rate = 0.001
discriminator_train_intervel=1
smooth=0.1
snapshot = 100
one_hot=np.eye(10)

model_path="models/"
pb_path=os.path.join(model_path,"pb/")
ckpt_path=os.path.join(model_path,"ckpt/")
if not os.path.exists(pb_path):
    os.makedirs(pb_path)
if not os.path.exists(ckpt_path):
    os.makedirs(ckpt_path)
model_name="cGAN_mnist_14x14"

def train():

    real_data_placeholder = tf.placeholder(tf.float32,shape=(batch_size,image_height_org,image_width_org,1),name="real_data")
    z_prior_placeholder = tf.placeholder(tf.float32,shape=(batch_size,prior_size),name="z_prior")
    label_placeholder = tf.placeholder(tf.float32,shape=(batch_size,class_num),name="label")

    mnist_GANer=cGAN_mnist.cGAN_mnist(image_flatten_size,smooth)

    generated_data = mnist_GANer.generator(z_prior_placeholder,label_placeholder)
    _generated_data = tf.identity(generated_data,name="generated_output")

    down_sample_image = mnist_GANer.get_downSample_img(real_data_placeholder)
    real_prop,real_logist = mnist_GANer.\
        discriminator(down_sample_image,label_placeholder)
    generated_prop,generated_logits = mnist_GANer.\
        discriminator(generated_data,label_placeholder,reuse=True)

    g_vars, d_vars = mnist_GANer.get_vars()
    g_loss, d_loss = mnist_GANer.loss(real_prop, generated_prop)
    # g_loss, d_loss = mnist_GANer.loss_tf(real_logist,generated_logits)

    #生成器和判别器各自更新自己的参数
    g_train_opt = tf.train.AdamOptimizer(learning_rate).minimize(g_loss, var_list = g_vars)
    d_train_opt = tf.train.AdamOptimizer(learning_rate).minimize(d_loss, var_list = d_vars)
    saver = tf.train.Saver(var_list=g_vars)   #只保存生成器的参数即可
    mnist = input_data.read_data_sets("MNIST_data")

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch_num in range(epochs):
            for _ in range(mnist.train.num_examples//batch_size):
                batch = mnist.train.next_batch(batch_size)
                batch_images = batch[0].reshape((batch_size,image_height_org,image_width_org,1))
                batch_labels = one_hot[np.array(batch[1])]  #注意要变成one-hot形式

                # 将像素值转到-1~1范围,这很有必要,不然后期生成效果很差
                batch_images = batch_images*2-1
                batch_z_prior = np.random.normal(0,1,size=(batch_size,prior_size))

                #更新判别器
                discriminator_dict={real_data_placeholder:batch_images,
                                    z_prior_placeholder:batch_z_prior,
                                    label_placeholder:batch_labels}

                train_loss_d,_ = sess.run([d_loss,d_train_opt],
                                          feed_dict=discriminator_dict)
                # 更新生成器
                batch_z_prior = np.random.normal(0, 1, (batch_size, prior_size))

                batch_labels = np.random.randint(0, 10, (batch_size, 1))
                batch_labels =one_hot[batch_labels].squeeze()

                generator_dict={z_prior_placeholder:batch_z_prior,
                                label_placeholder: batch_labels}
                train_loss_g, _ = sess.run([g_loss, g_train_opt],
                                           feed_dict=generator_dict)

            print("Epoch:{}/{},D_Loss: {:.4f},G_Loss: {:.4f}".format(
                epoch_num, epochs, train_loss_d, train_loss_g))

            if epoch_num % snapshot == 0:
                if not os.path.exists("gen_pics"):
                    os.makedirs("gen_pics")

                down_sample_img,generated_out = sess.run([down_sample_image,generated_data],
                                                         feed_dict=discriminator_dict)

                down_sample_img = (down_sample_img+1)/2*255.0
                generated_out = (generated_out+1)/2*255.0
                for ind in range(10):
                    cur_generated_out = generated_out[ind].reshape((
                        gen_image_height, gen_image_width))
                    cur_real_image = down_sample_img[ind].reshape((
                        gen_image_height, gen_image_width))

                    cv2.imwrite("gen_pics/real_{}_{}.jpg".format(
                        batch[1][ind],epoch_num),cur_real_image)
                    cv2.imwrite("gen_pics/gen_{}_{}.jpg".format(
                        batch[1][ind], epoch_num), cur_generated_out)


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