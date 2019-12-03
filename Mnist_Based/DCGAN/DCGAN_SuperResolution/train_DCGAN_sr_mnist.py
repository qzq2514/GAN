from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.framework import graph_util
from net import DCGAN_sr_mnist as DCGAN_mnist
import tensorflow as tf
import numpy as np
import time
import cv2
import os


image_height = 28
image_width = 28
noise_channal=1

class_num=10
channels=1
batch_size = 64
epochs = 2000
learning_rate = 0.001
discriminator_train_intervel=1
smooth=0.1
snapshot = 100

model_path="models_fashion/"
pb_path=os.path.join(model_path,"pb/")
ckpt_path=os.path.join(model_path,"ckpt/")

save_img_dir = "result_fashion/"
image_path = "../../Data/MNIST_fashion"
if not os.path.exists(pb_path):
    os.makedirs(pb_path)
if not os.path.exists(ckpt_path):
    os.makedirs(ckpt_path)
if not os.path.exists(save_img_dir):
    os.makedirs(save_img_dir)
model_name="DCGAN_fashion_sr"

def get_corase_image(org_images):

    corase_images = []
    for org_image in org_images:
        down_sample = cv2.resize(org_image,(image_width//2,image_height//2))
        up_sample = cv2.resize(down_sample,(image_width,image_height))
        up_sample = up_sample.reshape((image_width,image_height,channels))
        corase_images.append(up_sample)

        # print("org_image:",org_image.shape,up_sample.shape)
    return np.array(corase_images)

def train():

    real_data_placeholder = tf.placeholder(tf.float32,shape=(None,image_height,image_width,channels),name="real_data")
    corase_data_placeholder = tf.placeholder(tf.float32, shape=(None, image_height, image_width, channels),name="corase_data")
    z_prior_placeholder = tf.placeholder(tf.float32,shape=(None,image_height,image_width,noise_channal),name="z_prior")
    is_training_placeholder = tf.placeholder_with_default(False,shape=(),name="is_training")

    mnist_GANer=DCGAN_mnist.DCGAN_mnist(smooth,is_training_placeholder)

    generated_data = mnist_GANer.generator(corase_data_placeholder,z_prior_placeholder)
    _generated_data = tf.identity(generated_data,name="generated_output")

    added_img = generated_data + corase_data_placeholder
    real_prop,real_logist = mnist_GANer.discriminator(real_data_placeholder)
    generated_prop,generated_logits = mnist_GANer.discriminator(added_img,reuse=True)

    g_vars, d_vars = mnist_GANer.get_vars()
    g_loss, d_loss = mnist_GANer.loss(real_prop, generated_prop)
    # g_loss, d_loss = mnist_GANer.loss_tf(real_logist,generated_logits)

    global_step = tf.Variable(0, trainable=False)
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        #生成器和判别器各自更新自己的参数
        g_train_opt = tf.train.AdamOptimizer(learning_rate,beta1=0.5).\
                          minimize(g_loss,global_step=global_step, var_list = g_vars)
        d_train_opt = tf.train.AdamOptimizer(learning_rate,beta1=0.5).\
                          minimize(d_loss, var_list = d_vars)

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

        for _ in range(epochs):
            for _ in range(mnist.train.num_examples//batch_size):
                batch = mnist.train.next_batch(batch_size)
                batch_images = batch[0].reshape((batch_size,image_height,image_width,channels))

                # 将像素值转到-1~1范围,这很有必要,不然后期生成效果很差
                batch_images = batch_images*2-1
                corase_images = get_corase_image(batch_images)

                batch_z_prior = np.random.uniform(-1,1,size=(batch_size,image_height,
                                                           image_width,noise_channal))
                # print(batch_images.shape,corase_images.shape)
                #更新判别器
                feed_dict={real_data_placeholder:batch_images,
                           corase_data_placeholder:corase_images,
                            z_prior_placeholder:batch_z_prior,
                            is_training_placeholder:True}

                global_step_,train_loss_d,_ = sess.run([global_step,d_loss,d_train_opt],
                                                       feed_dict=feed_dict)
                generated_out,train_loss_g, _ = \
                    sess.run([_generated_data,g_loss, g_train_opt],
                             feed_dict={corase_data_placeholder:corase_images,
                                        z_prior_placeholder:batch_z_prior,
                                        is_training_placeholder:True})

                epoch_num = global_step_ // step_per_epoch
                step = global_step_ % step_per_epoch

                print("Epoch:[{}] ,Step:[{}/{}] ,time:{} s ,D_Loss: {:.4f} ,G_Loss: {:.4f}".format(
                    epoch_num, step, step_per_epoch, time.time() - start_time, train_loss_d, train_loss_g))

                #使用batch_norm就要查看其中norm的均值或方差有没有稳定
                one_moving_meaning_show = "No mean or variance"
                if len(bn_moving_vars) > 0:
                    one_moving_meaning = sess.graph.get_tensor_by_name(bn_moving_vars[0].name)
                    one_moving_meaning_show = np.mean(one_moving_meaning.eval())
                print("one_moving_meaning:", one_moving_meaning_show)

                # print(generated_out[0])
                # print(np.min(generated_out[0]),np.max(generated_out[0]))
                if global_step_ % snapshot == 0 :

                    #保存图像
                    fine_sample = ((generated_out+corase_images +1)/2)*255.0
                    # fine_sample = fine_sample.astype(np.uint8)

                    corase_images = ((corase_images+1)/2)*255.0
                    corase_images = corase_images.astype(np.uint8)

                    batch_images = ((batch_images+1)/2)*255.0
                    batch_images = batch_images.astype(np.uint8)

                    for index in range(10):
                        cv2.imwrite(save_img_dir+"/{}_{}_org_img.jpg".
                                    format( global_step_, index), batch_images[index])
                        cv2.imwrite(save_img_dir+"/{}_{}_corase_img.jpg".
                                    format(global_step_,index), corase_images[index])
                        cv2.imwrite(save_img_dir+"/{}_{}_sr_sample.jpg".
                                    format(global_step_,index),fine_sample[index])

                    # 保存PB
                    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def,
                                                                               ["generated_output"])
                    save_model_name = model_name + "-" + str(global_step_) + ".pb"
                    with tf.gfile.FastGFile(pb_path + save_model_name, mode="wb") as fw:
                        fw.write(constant_graph.SerializeToString())
                    # 保存CKPT
                    saver.save(sess, ckpt_path + model_name + ".ckpt", global_step=global_step_)
                    print("Successfully saved model {}".format(save_model_name))

if __name__ == '__main__':
    train()