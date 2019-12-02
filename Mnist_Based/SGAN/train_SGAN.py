from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import os
from tensorflow.python.framework import graph_util
from net import SGAN_mnist
import numpy as np
import cv2
import time


image_height = 28
image_width = 28
prior_size=100
num_class=10

channels=1
batch_size = 64
epochs = 2000
max_step = 2000
d_learning_rate = 0.001
g_learning_rate = 0.001

discriminator_train_intervel=1
smooth=0.1
snapshot = 200
one_hot=np.eye(num_class)

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
model_name="SGAN_fashion"

def train():

    real_data_placeholder = tf.placeholder(tf.float32,shape=[None,image_height,image_width,channels],name="real_data")
    z_prior_placeholder = tf.placeholder(tf.float32,shape=[None,prior_size],name="z_prior")
    label_placeholder = tf.placeholder(tf.int32,shape=[None,],name="label")
    is_training_placeholder = tf.placeholder_with_default(False,shape=(),name="is_training")

    mnist_GANer=SGAN_mnist.SGAN_mnist(smooth,is_training_placeholder,num_class,batch_size)

    generated_data = mnist_GANer.generator(z_prior_placeholder,label_placeholder)
    print("generated_data:",generated_data)
    _generated_data = tf.identity(generated_data,name="generated_output")

    d_sigmoid_real, q_cat_logit_real = \
         mnist_GANer.discriminator(real_data_placeholder,label_placeholder)
    d_sigmoid_fake, q_cat_logit_fake = \
        mnist_GANer.discriminator(generated_data,label_placeholder,reuse=True)

    g_vars, d_vars = mnist_GANer.get_vars()
    g_loss, d_loss = mnist_GANer.loss(d_sigmoid_real, d_sigmoid_fake,label_placeholder,
                                      q_cat_logit_real,q_cat_logit_fake)

    global_step = tf.Variable(0, trainable=False)
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        #生成器和判别器各自更新自己的参数
        g_train_opt = tf.train.AdamOptimizer(d_learning_rate,beta1=0.5, beta2=0.99).\
                          minimize(g_loss,global_step=global_step, var_list = g_vars)
        d_train_opt = tf.train.AdamOptimizer(g_learning_rate,beta1=0.5, beta2=0.99).\
                          minimize(d_loss, var_list = d_vars)

    # 为了为了能够接着之前的训练继续训练,所以这里最好不要只保存生成器的参数
    saver = tf.train.Saver()
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

        for epoch_num in range(epochs):
            is_break=False
            for _ in range(mnist.train.num_examples//batch_size):
                batch = mnist.train.next_batch(batch_size)

                batch_images = batch[0].reshape((batch_size,image_height,image_width,channels))
                batch_labels = np.array(batch[1])

                batch_images = batch_images*2-1
                batch_z_prior = np.random.normal(0,1,size=(batch_size,prior_size))

                #更新判别器
                feed_dict={real_data_placeholder:batch_images,
                            z_prior_placeholder:batch_z_prior,
                            label_placeholder:batch_labels,
                            is_training_placeholder:True}

                global_step_,train_loss_d,_ = sess.run([global_step,d_loss,d_train_opt],
                                          feed_dict=feed_dict)
                # 更新生成器
                train_loss_g, _ = sess.run([g_loss, g_train_opt],
                                           feed_dict=feed_dict)

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

                if global_step_ >= max_step:
                    is_break=True
                    break
            if is_break:
                break


if __name__ == '__main__':
    train()