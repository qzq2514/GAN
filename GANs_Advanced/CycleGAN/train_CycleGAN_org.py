from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.framework import graph_util
import tensorflow.contrib.slim as slim
from DataLoader import Pix2Pix_loader
from net.CycleGAN import CycleGAN
from net.ImagePool import ImagePool
import tensorflow as tf
import numpy as np
import scipy.misc
import dbread as db
import os

os.environ['CUDA_VISIBLE_DEVICES']='0'

image_height = 64
image_width = 64
batch_size = 1
sample_num = 10
pool_size=50
Train_Step=600005

end_learning_rate = 0.0
start_decay_step = 100000
decay_steps = 100000
starter_learning_rate  = 2e-4

lambda_reconst = 10.0
decay=0.9
image_dir = "D:/forTensorflow/forGAN/edges2shoes/"
model_name = "CycleGAN_new"
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
    fake_pool_A_place = tf.placeholder(tf.float32, shape=[None, image_height,image_width, 3], name="fake_pool_A")
    fake_pool_B_place = tf.placeholder(tf.float32, shape=[None, image_height, image_width, 3], name="fake_pool_B")
    is_training_place = tf.placeholder(tf.bool, shape=(),name="is_training")

    cycleGAN = CycleGAN(is_training_place,lambda_reconst)

    Gen_AB_loss, Gen_BA_loss, Dis_A_loss, Dis_B_loss,fake_A,fake_B= \
        cycleGAN.build_CycleGAN(input_A_place,input_B_place,
                                fake_pool_A_place,fake_pool_B_place)

    gen_A2B_vars, gen_B2A_vars, dis_A_vars, dis_B_vars = cycleGAN.get_vars()
    global_step = tf.Variable(-1, trainable=False,name="global_step")
    global_step_increase = tf.assign(global_step, tf.add(global_step, 1))

    learning_rate = (tf.where(tf.greater_equal(global_step, start_decay_step),
                    tf.train.polynomial_decay(starter_learning_rate, global_step-start_decay_step,decay_steps,
                    end_learning_rate,power=1.0),starter_learning_rate))
    # 不要使用with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS))来更新batchnorm的参数
    # 因为其tf.GraphKeys.UPDATE_OPS包含了生成器和判别器所有的batchnorm的参数
    train_op_G = tf.train.AdamOptimizer(learning_rate, beta1=0.5, ). \
        minimize(Gen_AB_loss+Gen_BA_loss, var_list=gen_A2B_vars+gen_B2A_vars)
    train_op_D = tf.train.AdamOptimizer(learning_rate, beta1=0.5, ). \
        minimize(Dis_A_loss+Dis_B_loss, var_list=dis_A_vars+dis_B_vars)

    A2B_out,ABA_out = cycleGAN.sample_generate(input_A_place, "A2B")
    A2B_output = tf.identity(A2B_out, name="A2B_output")
    B2A_out,BAB_out = cycleGAN.sample_generate(input_B_place, "B2A")
    B2A_output = tf.identity(B2A_out, name="B2A_output")

    fake_A_pool = ImagePool(pool_size)
    fake_B_pool = ImagePool(pool_size)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        ckpt = tf.train.get_checkpoint_state(ckpt_path)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join(ckpt_path, ckpt_name))

        _global_step = sess.run(global_step_increase)
        dataLoader = Pix2Pix_loader(image_dir, image_height, image_width, batch_size=batch_size,
                                    global_step=_global_step)
        while _global_step<Train_Step:
            images_A,images_B = dataLoader.next_batch()

            feed_dict_pool = {input_A_place:images_A,input_B_place:images_B,
                               is_training_place:True}

            fake_A_vals, fake_B_vals = sess.run([fake_A,fake_B],feed_dict=feed_dict_pool)

            feed_dict_train = {input_A_place: images_A, input_B_place: images_B,
                              is_training_place: True,
                               fake_pool_A_place: fake_A_pool.query(fake_A_vals),
                               fake_pool_B_place: fake_B_pool.query(fake_B_vals)}
                               
            sess.run(train_op_D, feed_dict=feed_dict_train)
            sess.run(train_op_G, feed_dict=feed_dict_train)
            sess.run(train_op_G, feed_dict=feed_dict_train)


            _Gen_AB_loss, _Gen_BA_loss,_Dis_A_loss, _Dis_B_loss\
                = sess.run([Gen_AB_loss, Gen_BA_loss, Dis_A_loss, Dis_B_loss],feed_dict=feed_dict_train)

            # if _global_step%50==0:
            print("Step:{},Gen_AB_loss:{},Gen_BA_loss:{},Dis_A_loss:{},Dis_B_loss:{}".format(_global_step,
                        _Gen_AB_loss, _Gen_BA_loss,_Dis_A_loss, _Dis_B_loss,))

            if _global_step%100==0:
                test_images_A, test_images_B = dataLoader.random_next_train_batch()

                #save result form A to B
                _A2B_output,_ABA_out = sess.run([A2B_output,ABA_out],feed_dict={input_A_place:test_images_A,
                                     is_training_place:False})
                _A2B_output = (_A2B_output + 1) / 2 * 255.0
                _ABA_out = (_ABA_out + 1) / 2 * 255.0
                for ind,trg_image in enumerate(_A2B_output[:sample_num]):
                    scipy.misc.imsave(result_dir + "/{}_{}_A.jpg".format(_global_step,ind),test_images_A[ind])
                    scipy.misc.imsave(result_dir + "/{}_{}_A2B.jpg".format(_global_step,ind), _A2B_output[ind])
                    scipy.misc.imsave(result_dir + "/{}_{}_ABA.jpg".format(_global_step, ind), _ABA_out[ind])

                # save result form B to A
                _B2A_output,_BAB_out = sess.run([B2A_output,BAB_out], feed_dict={input_B_place: test_images_B,
                                     is_training_place: False})
                _B2A_output = (_B2A_output + 1) / 2 * 255.0
                _BAB_out = (_BAB_out + 1) / 2 * 255.0
                for ind,trg_image in enumerate(_B2A_output[:sample_num]):
                    scipy.misc.imsave(result_dir + "/{}_{}_B.jpg".format(_global_step,ind),test_images_B[ind])
                    scipy.misc.imsave(result_dir + "/{}_{}_B2A.jpg".format(_global_step,ind), _B2A_output[ind])
                    scipy.misc.imsave(result_dir + "/{}_{}_BAB.jpg".format(_global_step, ind), _BAB_out[ind])

            if _global_step%100000==0:
                # 保存PB
                constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def,
                                                                           ["A2B_output","B2A_output"])
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