import tensorflow as tf
import os
import cv2
from tensorflow.python.framework import graph_util
import matplotlib.pyplot as plt
from net import LapGAN_mnist_independent_noConv as LapGAN_mnist
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


class_num=10
image_height = 28
image_width = 28
image_channal = 1

# prior_scales=[100]
prior_scales=[10*10, 10*10, 100]

batch_size = 128
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
model_name="LapGAN_mnist_independent3"

def train():

    real_data_placeholder = tf.placeholder(tf.float32,shape=(None,image_height,image_width,
                                                             image_channal),name="real_data")
    #每个生成器的输入噪声
    z_prior_placeholders = []
    for ind,prior_size in enumerate(prior_scales):
        z_prior_placeholders.append(tf.placeholder(tf.float32,shape=(None,prior_size*image_channal),
                                                   name="z_prior_"+str(ind)))
    label_placeholder = tf.placeholder(tf.float32,shape=(None,class_num),name="label")
    keep_prob_placeholder = tf.placeholder_with_default(1.0,shape=(),name="keep_prob")
    is_training_placeholder = tf.placeholder_with_default(False, shape=(), name="is_training")

    mnist_GANer=LapGAN_mnist.LapGAN_mnist(real_data_placeholder,z_prior_placeholders,
                                          label_placeholder,keep_prob_placeholder,
                                          is_training_placeholder,prior_scales,smooth)

    mnist_GANer.build_LapGAN()

    #generate_Laplace_images:生成器生成的分辨率从低到高的图像
    generate_Laplace_images = mnist_GANer.generate_Laplace()
    generated_out_names=[]
    for ind ,size in enumerate(prior_scales):
        generated_out_names.append("generate_Laplace_out"+str(ind))
    #下面逆序保证"generate_Laplace_out0"是分辨率最高的图像(金字塔底端)
    for ind,generate_Laplace_image in enumerate(generate_Laplace_images[::-1]):
        _ = tf.identity(generate_Laplace_image,generated_out_names[ind])

    g_loss,d_loss = mnist_GANer.loss()

    g_vars, d_vars = mnist_GANer.get_vars()
    print(g_vars)

    d_opt=[]; g_opt=[]
    all_g_vars=[]
    for ind,prior_size in enumerate(prior_scales):
        g_opt.append(tf.train.AdamOptimizer(learning_rate,beta1=0.5,beta2=0.9).
                     minimize(g_loss[ind], var_list = g_vars[ind]))
        d_opt.append(tf.train.AdamOptimizer(learning_rate,beta1=0.5,beta2=0.9).
                     minimize(d_loss[ind], var_list = d_vars[ind]))
        all_g_vars.extend(g_vars[ind])

    mnist = input_data.read_data_sets("MNIST_data")
    saver = tf.train.Saver(var_list=all_g_vars)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch_num in range(epochs):
            for _ in range(mnist.train.num_examples//batch_size):
                batch = mnist.train.next_batch(batch_size)
                batch_images = batch[0].reshape((batch_size,image_height,image_width,image_channal))
                batch_images = batch_images * 2 - 1    #归一化到-1~1
                batch_labels = one_hot[np.array(batch[1])]  #注意要变成one-hot形式

                feed_dict_train={real_data_placeholder:batch_images,
                                 keep_prob_placeholder:0.5,
                                 is_training_placeholder:True,
                                 label_placeholder:batch_labels}

                for ind,prior_size in enumerate(prior_scales):
                    z_prior = np.random.normal(0, 1,size=(batch_size, prior_size*image_channal))
                    feed_dict_train[z_prior_placeholders[ind]] = z_prior

                g_loss_eval=[]
                d_loss_eval = []
                for ind, prior_size in enumerate(prior_scales):
                    cur_d_loss, _ = sess.run([d_loss[ind],d_opt[ind]],feed_dict=feed_dict_train)
                    cur_g_loss, _ = sess.run([g_loss[ind], g_opt[ind]],feed_dict=feed_dict_train)

                    d_loss_eval.append(cur_d_loss)
                    g_loss_eval.append(cur_g_loss)

            # print("d_loss:",d_loss_eval)
            # print("g_loss:",g_loss_eval)

                # image_show_resize,g_pyramid,corase_pyramid,l_pyramid = \
                #     sess.run([mnist_GANer.real_data,mnist_GANer.g_pyramid[0], \
                #               mnist_GANer.corase_pyramid[0],mnist_GANer.l_pyramid[0]],feed_dict=feed_dict_train)
                #
                # image_show = (batch_images[0] * 255).astype(np.uint8)
                # image_show_resize = (image_show_resize[0]* 255).astype(np.uint8)
                # g_pyramid = (g_pyramid[0] * 255).astype(np.uint8)
                # corase_pyramid = (corase_pyramid[0] * 255).astype(np.uint8)
                # l_pyramid = (l_pyramid[0] * 255).astype(np.uint8)
                #
                # # cv2.imshow("mnist",image_show)
                # cv2.imshow("image_show_resize", image_show_resize)
                # cv2.imshow("g_pyramid", g_pyramid)
                # cv2.imshow("corase_pyramid", corase_pyramid)
                # cv2.imshow("l_pyramid", l_pyramid)
                # cv2.waitKey()

                # # 将像素值转到-1~1范围,这很有必要,不然后期生成效果很差
                # batch_images = batch_images*2-1
                # batch_z_prior = np.random.normal(0,1,size=(batch_size,prior_size))

                #更新判别器
            #     discriminator_dict={real_data_placeholder:batch_images,
            #                         z_prior_placeholder:batch_z_prior,
            #                         label_placeholder:batch_labels}
            #
            #     train_loss_d,_ = sess.run([d_loss,d_train_opt],
            #                               feed_dict=discriminator_dict)
            #     # 更新生成器
            #     batch_z_prior = np.random.normal(0, 1, (batch_size, prior_size))
            #     # 随机赋予标签,要保证每个标签都被充分地用过
            #     # 本人做过实验,如果有一个标签一直没有出现过,那么最终该标签下的数据生成效果极差
            #     # 甚至如果每次这里都赋予一个固定的标签,如每次都赋值0标签,那么就不会生成其他标签的数据
            #     # 可自行尝试cGAN_mnist_noLabel-300.pb
            #     batch_labels = np.random.randint(0, 1, (batch_size, 1))
            #     batch_labels =one_hot[batch_labels].squeeze()
            #
            #     generator_dict={z_prior_placeholder:batch_z_prior,
            #                     label_placeholder: batch_labels}
            #     train_loss_g, _ = sess.run([g_loss, g_train_opt],
            #                                feed_dict=generator_dict)


            print("Epoch:{}/{} \nD_Loss: {} \nG_Loss: {}".format(
                epoch_num, epochs, d_loss_eval, g_loss_eval))

            if epoch_num % snapshot == 0:
                # 保存PB
                constant_graph = graph_util.convert_variables_to_constants(sess,
                                                sess.graph_def, generated_out_names)
                save_model_name = model_name + "-" + str(epoch_num) + ".pb"
                with tf.gfile.FastGFile(pb_path + save_model_name, mode="wb") as fw:
                    fw.write(constant_graph.SerializeToString())
                # 保存CKPT
                saver.save(sess, ckpt_path + model_name + ".ckpt", global_step=epoch_num)
                print("Successfully saved model {}".format(save_model_name))

if __name__ == '__main__':
    train()