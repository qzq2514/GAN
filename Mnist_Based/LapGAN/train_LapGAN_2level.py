import tensorflow as tf
import os
import cv2
import time
from tensorflow.python.framework import graph_util
import matplotlib.pyplot as plt
from net import LapGAN_mnist
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

class_num=10
image_height = 28
image_width = 28
image_channal = 1

prior_scales=[28*28, 100]

batch_size = 64
epochs = 2000
learning_rate = 0.001
discriminator_train_intervel=1
smooth=0.1
snapshot = 100
one_hot=np.eye(10)


model_path="models_mnist/"
pb_path=os.path.join(model_path,"pb/")
ckpt_path=os.path.join(model_path,"ckpt/")
result_dir="result_mnist"
save_train_img_root=result_dir+"/train_independent"
save_test_img_root=result_dir+"/test_dependent"
if not os.path.exists(save_train_img_root):
    os.makedirs(save_train_img_root)
if not os.path.exists(save_test_img_root):
    os.makedirs(save_test_img_root)
if not os.path.exists(pb_path):
    os.makedirs(pb_path)
if not os.path.exists(ckpt_path):
    os.makedirs(ckpt_path)
image_path = "../../Data/MNIST_data"
model_name="LapGAN_mnist_2Level"

def train():


    real_data_placeholder = tf.placeholder(tf.float32,shape=(batch_size,image_height,image_width,
                                                             image_channal),name="real_data")
    #每个生成器的输入噪声
    z_prior_placeholders = []
    for ind,prior_size in enumerate(prior_scales):
        z_prior_placeholders.append(tf.placeholder(tf.float32,shape=(batch_size,prior_size*image_channal),
                                                   name="z_prior_"+str(ind)))
    label_placeholder = tf.placeholder(tf.float32,shape=(batch_size,class_num),name="label")
    keep_prob_placeholder = tf.placeholder_with_default(1.0,shape=(),name="keep_prob")
    is_training_placeholder = tf.placeholder_with_default(False, shape=(), name="is_training")

    mnist_GANer=LapGAN_mnist.LapGAN_mnist(real_data_placeholder,z_prior_placeholders,
                                          label_placeholder,keep_prob_placeholder,
                                          is_training_placeholder,prior_scales,smooth)

    mnist_GANer.build_LapGAN()

    # generate_Laplace_images:生成器生成的分辨率从低到高的图像
    # 是串联更高层的生成器生成的结果,最终可以在eval程序中查看,效果并不好
    # 我觉得是因为本来mnist是28x28的图像,高层的原始图像是非常模糊的,所以真实图像就不好,生成的图像的效果也不会很好
    generate_Laplace_images = mnist_GANer.generate_Laplace()[::-1]
    generated_out_names=[]
    for ind ,size in enumerate(prior_scales):
        generated_out_names.append("generate_Laplace_out"+str(ind))
    #下面逆序保证"generate_Laplace_out0"是分辨率最高的图像(金字塔底端)
    for ind,generate_Laplace_image in enumerate(generate_Laplace_images):
        _ = tf.identity(generate_Laplace_image,generated_out_names[ind])

    g_loss,d_loss = mnist_GANer.loss()

    g_vars, d_vars = mnist_GANer.get_vars()

    d_opt=[]; g_opt=[]
    all_g_vars=[]
    global_step = tf.Variable(0, trainable=False)
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        for ind,prior_size in enumerate(prior_scales):
            g_opt.append(tf.train.AdamOptimizer(learning_rate,beta1=0.5,beta2=0.9).
                         minimize(g_loss[ind],global_step=global_step, var_list = g_vars[ind]))
            d_opt.append(tf.train.AdamOptimizer(learning_rate,beta1=0.5,beta2=0.9).
                         minimize(d_loss[ind],global_step=global_step, var_list = d_vars[ind]))
            all_g_vars.extend(g_vars[ind])

    mnist = input_data.read_data_sets(image_path)
    saver = tf.train.Saver()
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
                    global_step_ ,cur_d_loss, _ = sess.run([global_step,d_loss[ind],d_opt[ind]],feed_dict=feed_dict_train)
                    cur_g_loss, _ = sess.run([g_loss[ind], g_opt[ind]],feed_dict=feed_dict_train)

                    d_loss_eval.append(cur_d_loss)
                    g_loss_eval.append(cur_g_loss)

                step_pyramid = int((global_step_+1)/(len(prior_scales)*2))-1
                epoche_num = step_pyramid // step_per_epoch
                step = step_pyramid % step_per_epoch
                print("Epoch:{},Step:[{}/{}] time:{}s\nD_Loss: {} \nG_Loss: {}".format(
                    epoche_num, step,step_per_epoch,time.time()-start_time,d_loss_eval, g_loss_eval))

                # 使用batch_norm就要查看其中norm的均值或方差有没有稳定
                one_moving_meaning_show = "No mean or variance"
                if len(bn_moving_vars) > 0:
                    one_moving_meaning = sess.graph.get_tensor_by_name(bn_moving_vars[0].name)
                    one_moving_meaning_show = np.mean(one_moving_meaning.eval())
                print("one_moving_meaning:", one_moving_meaning_show)

                if step_pyramid % snapshot == 0:
                    # batch_labels = np.random.randint(0,10,size=[batch_size,1]).squeeze(axis=1)
                    # batch_labels_one_hot = one_hot[batch_labels]
                    # feed_dict_show = {real_data_placeholder: batch_images,
                    #                    label_placeholder: batch_labels_one_hot}
                    #
                    # for ind, prior_size in enumerate(prior_scales):
                    #     z_prior = np.random.normal(0, 1, size=(batch_size, prior_size * image_channal))
                    #     feed_dict_show[z_prior_placeholders[ind]] = z_prior

                    for scale_index in range(len(prior_scales)):
                        # 保存从高级到低级生成器串联的生成结果
                        save_dir = os.path.join(save_test_img_root, str(scale_index))
                        if not os.path.exists(save_dir):
                            os.makedirs(save_dir)
                        cur_generated_out = sess.run(generate_Laplace_images[scale_index],
                                                     feed_dict=feed_dict_train)  #feed_dict_train feed_dict_show
                        cur_generated_out = ((cur_generated_out + 1) / 2) * 255.0
                        # cur_generated_out = cur_generated_out.astype(np.uint8)

                        for index in range(10):
                            cv2.imwrite("{}/generated_img_{}_{}.jpg".
                                        format(save_dir, batch[1][index], step_pyramid), cur_generated_out[index])

                        # 保存各个生成器独立生成的图
                        save_dir = os.path.join(save_train_img_root, str(scale_index))
                        if not os.path.exists(save_dir):
                            os.makedirs(save_dir)

                        g_pyramid_place=mnist_GANer.g_pyramid[scale_index]
                        generted_place=mnist_GANer.generators[scale_index]
                        corase_pyramid= mnist_GANer.corase_pyramid[scale_index] if \
                            scale_index!=len(prior_scales)-1 else generted_place

                        real_image,generated_out,corase_img_out = sess.run(
                            [g_pyramid_place,generted_place,corase_pyramid],
                              feed_dict=feed_dict_train)


                        if scale_index!=len(prior_scales):
                            fine_sample = ((generated_out + corase_img_out + 1) / 2) * 255.0
                        else:
                            fine_sample = ((generated_out + 1) / 2) * 255.0

                        # fine_sample = fine_sample.astype(np.uint8)
                        corase_img_out = ((corase_img_out + 1) / 2) * 255.0
                        # corase_img_out = corase_img_out.astype(np.uint8)
                        real_image = ((real_image + 1) / 2) * 255.0
                        # real_image = real_image.astype(np.uint8)

                        for index in range(10):
                            cv2.imwrite("{}/corase_img_{}_{}_{}.jpg".
                                        format(save_dir,batch[1][index], step_pyramid,index), corase_img_out[index])
                            cv2.imwrite("{}/sr_sample_{}_{}_{}.jpg".
                                        format(save_dir,batch[1][index], step_pyramid,index), fine_sample[index])
                            cv2.imwrite("{}/real_sample_{}_{}_{}.jpg".
                                        format(save_dir,batch[1][index], step_pyramid,index), real_image[index])

                    # 保存PB
                    constant_graph = graph_util.convert_variables_to_constants(sess,
                                                    sess.graph_def, generated_out_names) #
                    save_model_name = model_name + "-" + str(step_pyramid) + ".pb"
                    with tf.gfile.FastGFile(pb_path + save_model_name, mode="wb") as fw:
                        fw.write(constant_graph.SerializeToString())
                    # 保存CKPT
                    saver.save(sess, ckpt_path + model_name + ".ckpt", global_step=step_pyramid)
                    print("Successfully saved model {}".format(save_model_name))

                step_pyramid += 1

if __name__ == '__main__':
    train()