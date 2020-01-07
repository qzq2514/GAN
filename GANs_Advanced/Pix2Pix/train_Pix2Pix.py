from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.framework import graph_util
import tensorflow.contrib.slim as slim
from DataLoader import Pix2Pix_loader
from net.Pix2Pix import Pix2Pix
import tensorflow as tf
import numpy as np
import scipy.misc
import os

os.environ['CUDA_VISIBLE_DEVICES']='1'

image_height = 64
image_width = 64
batch_size = 1
sample_num = 10
Train_Step = 600005
learning_rate  = 0.0002
lambda_reconst = 100.0

#读取未分开的成对数据
image_dir = "/media/cgim/data/GAN/data/edges2shoes/"
model_name = "Pix2Pix_shoe2edge"
model_path="/media/cgim/dataset/models/"+model_name

pb_path=os.path.join(model_path,"pb/")
ckpt_path=os.path.join(model_path,"ckpt/")
result_dir=model_path+"/result"

if not os.path.exists(result_dir):
    os.makedirs(result_dir)
if not os.path.exists(pb_path):
    os.makedirs(pb_path)
if not os.path.exists(ckpt_path):
    os.makedirs(ckpt_path)


def train():
    input_A_place = tf.placeholder(tf.float32,shape=[None,image_height,image_width, 3],name="input_A")
    input_B_place = tf.placeholder(tf.float32, shape=[None, image_height,image_width, 3], name="input_B")
    is_training_place = tf.placeholder(tf.bool, shape=(),name="is_training")
    keep_prob_place = tf.placeholder_with_default(1.0, shape=(),name="keep_prob")

    pix2pix = Pix2Pix(is_training_place,keep_prob_place,lambda_reconst)

    G_loss,D_loss = pix2pix.build_Pix2Pix(input_A_place,input_B_place)

    g_vars,d_vars = pix2pix.get_vars()
    global_step = tf.Variable(-1, trainable=False,name="global_step")
    global_step_increase = tf.assign(global_step, tf.add(global_step, 1))
    train_op_D = tf.train.AdamOptimizer(learning_rate, beta1=0.5, ).minimize(D_loss, var_list=d_vars)
    train_op_G = tf.train.AdamOptimizer(learning_rate, beta1=0.5, ).minimize(G_loss, var_list=g_vars)

    A2B_out = pix2pix.sample_generate(input_A_place)
    A2B_output = tf.identity(A2B_out, name="A2B_output")

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        ckpt = tf.train.get_checkpoint_state(ckpt_path)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join(ckpt_path, ckpt_name))

        _global_step = sess.run(global_step_increase)
        dataLoader = Pix2Pix_loader(image_dir, image_height, image_width,batch_size=batch_size,global_step=_global_step)
        while _global_step<Train_Step:
            images_A , images_B = dataLoader.next_batch()      #0~255

            feed_dict = {input_A_place:images_A,input_B_place:images_B,
                         is_training_place:True,keep_prob_place:0.5}

            sess.run(train_op_D, feed_dict=feed_dict)
            sess.run(train_op_G, feed_dict=feed_dict)

            _D_loss,_G_loss = sess.run([D_loss,G_loss],feed_dict=feed_dict)
            if _global_step%10==0:
                print("Step:{},D_loss:{},G_loss:{}".format(_global_step, _D_loss, _G_loss,))
            if _global_step%100==0:
                test_images_A, test_images_B = dataLoader.random_next_test_batch()

                #save result form A to B
                _A2B_output = sess.run(A2B_output,feed_dict={input_A_place:test_images_A,
                                     is_training_place:False,keep_prob_place:1.0})
                _A2B_output = (_A2B_output + 1) / 2 * 255.0
                for ind,trg_image in enumerate(_A2B_output[:sample_num]):
                    scipy.misc.imsave(result_dir + "/{}_{}_A.jpg".format(_global_step,ind),test_images_A[ind])
                    scipy.misc.imsave(result_dir + "/{}_{}_A2B.jpg".format(_global_step,ind), _A2B_output[ind])

            if _global_step%100000==0:
                # 保存PB
                constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def,["A2B_output"])
                save_model_name = model_name + "-" + str(_global_step) + ".pb"
                with tf.gfile.FastGFile(pb_path + save_model_name, mode="wb") as fw:
                    fw.write(constant_graph.SerializeToString())
                # 保存CKPT
                saver.save(sess, ckpt_path + model_name + ".ckpt", global_step=_global_step)
                print("Successfully saved model {}".format(save_model_name))
                # return
            _global_step = sess.run(global_step_increase)

if __name__ == '__main__':
    train()