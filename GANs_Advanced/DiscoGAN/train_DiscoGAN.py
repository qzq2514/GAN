from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.framework import graph_util
import tensorflow.contrib.slim as slim
from DataLoader import Pix2Pix_loader
from net.DiscoGAN import DiscoGAN
import tensorflow as tf
import numpy as np
import scipy.misc
import dbread as db
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'

image_height = 64
image_width = 64
batch_size = 64
sample_num = 10
Train_Step = 30005
starting_rate = 0.01
change_rate = 0.5
learning_rate  = 0.0002

#读取已经分开的成对数据
db_dir_A = "/media/cgim/data/GAN/data/edges2shoes_split/train/A"
db_dir_B = "/media/cgim/data/GAN/data/edges2shoes_split/train/B"

model_name = "DiscoGAN_1227_new"
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
    is_training_place = tf.placeholder_with_default(False, shape=(),name="is_training")
    reconst_rate_place = tf.placeholder(tf.float32, shape=(),name="reconst_rate")
    discoGan = DiscoGAN(is_training_place,reconst_rate_place)

    G_loss,D_loss = discoGan.build_DiscoGAN(input_A_place,input_B_place)
    g_vars,d_vars = discoGan.get_vars()

    global_step = tf.Variable(-1, trainable=False,name="global_step")
    global_step_increase = tf.assign(global_step, tf.add(global_step, 1))
    #不要使用with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS))来更新batchnorm的参数
    #因为其tf.GraphKeys.UPDATE_OPS包含了生成器和判别器所有的batchnorm的参数
    train_op_D = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(D_loss, var_list=d_vars)
    train_op_G = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(G_loss, var_list=g_vars)

    A2B_out,ABA_out = discoGan.sample_generate(input_A_place, "A2B")
    A2B_output = tf.identity(A2B_out, name="A2B_output")
    B2A_out,BAB_out = discoGan.sample_generate(input_B_place, "B2A")
    B2A_output = tf.identity(B2A_out, name="B2A_output")

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        ckpt = tf.train.get_checkpoint_state(ckpt_path)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join(ckpt_path, ckpt_name))

        _global_step = sess.run(global_step_increase)
        database_A = db.DBreader(db_dir_A, batch_size=batch_size, labeled=False, resize=[image_height, image_width])
        db_for_vis_A = db.DBreader(db_dir_A, batch_size=batch_size, labeled=False, resize=[image_height, image_width])
        database_B = db.DBreader(db_dir_B, batch_size=batch_size, labeled=False, resize=[image_height, image_width])
        db_for_vis_B = db.DBreader(db_dir_B, batch_size=batch_size, labeled=False, resize=[image_height, image_width])
        
        while _global_step<Train_Step:
            if _global_step<10000:
                reconst_rate = starting_rate
            else:
                reconst_rate = change_rate

            images_A = database_A.next_batch()
            images_B = database_B.next_batch()

            feed_dict = {input_A_place:images_A,input_B_place:images_B,
                         is_training_place:True,reconst_rate_place:reconst_rate}

            if _global_step%2==0:
                sess.run(train_op_D,feed_dict=feed_dict)
            sess.run(train_op_G, feed_dict=feed_dict)
            _global_step,_D_loss,_G_loss = sess.run([global_step,D_loss,G_loss],feed_dict=feed_dict)

            if _global_step%50==0:
                print("Step:{},Reconst_rate:{},D_loss:{},G_loss:{}".format(_global_step,reconst_rate, _D_loss, _G_loss,))

            if _global_step%100==0:
                test_images_A = db_for_vis_A.next_batch()
                test_images_B = db_for_vis_B.next_batch()
                
                #save result form A to B
                _A2B_output,_ABA_out = sess.run([A2B_output,ABA_out],feed_dict={input_A_place:test_images_A})
                _A2B_output = (_A2B_output + 1) / 2 * 255.0
                _ABA_out = (_ABA_out + 1) / 2 * 255.0
                for ind,trg_image in enumerate(_A2B_output[:sample_num]):
                    scipy.misc.imsave(result_dir + "/{}_{}_A.jpg".format(_global_step,ind),test_images_A[ind])
                    scipy.misc.imsave(result_dir + "/{}_{}_A2B.jpg".format(_global_step,ind), _A2B_output[ind])
                    scipy.misc.imsave(result_dir + "/{}_{}_ABA.jpg".format(_global_step, ind), _ABA_out[ind])

                # save result form B to A
                _B2A_output,_BAB_out = sess.run([B2A_output,BAB_out], feed_dict={input_B_place: test_images_B})
                _B2A_output = (_B2A_output + 1) / 2 * 255.0
                _BAB_out = (_BAB_out + 1) / 2 * 255.0
                for ind,trg_image in enumerate(_B2A_output[:sample_num]):
                    scipy.misc.imsave(result_dir + "/{}_{}_B.jpg".format(_global_step,ind),test_images_B[ind])
                    scipy.misc.imsave(result_dir + "/{}_{}_B2A.jpg".format(_global_step,ind), _B2A_output[ind])
                    scipy.misc.imsave(result_dir + "/{}_{}_BAB.jpg".format(_global_step, ind), _BAB_out[ind])

            if _global_step%800==0:
                # 保存PB
                constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def,
                                                                           ["A2B_output","B2A_output"])
                save_model_name = model_name + "-" + str(_global_step) + ".pb"
                with tf.gfile.FastGFile(pb_path + save_model_name, mode="wb") as fw:
                    fw.write(constant_graph.SerializeToString())
                # 保存CKPT
                saver.save(sess, ckpt_path + model_name + ".ckpt", global_step=_global_step)
                print("Successfully saved model {}".format(save_model_name))

            _global_step = sess.run(global_step_increase)












if __name__ == '__main__':
    train()
