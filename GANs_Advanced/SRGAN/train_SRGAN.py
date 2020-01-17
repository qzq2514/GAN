from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.framework import graph_util
import tensorflow.contrib.slim as slim
from DataLoader import CartoonGAN_Loader
from net.SRGAN_AllLoss import SRGAN
import tensorflow as tf
import numpy as np
import scipy.misc
import os

os.environ['CUDA_VISIBLE_DEVICES']='0'

image_height = 256
image_width = 256
batch_size = 8
sample_num = 10
pool_size=50

pre_train_iter = 20000
gan_train_iter = 100005
learning_rate = 0.0001

vgg_weight = 1e3
image_dir = "/media/cgim/data/GAN/data/cartoonGAN_dataset2/Cartoon"
model_name = "SRGAN_AllLoss"
model_path="/media/cgim/dataset/models/"+model_name

pb_path=os.path.join(model_path,"pb/")
ckpt_path=os.path.join(model_path,"ckpt/")
result_dir=model_path+"/result/"
content_result=result_dir+"content"
HR_gen_result=result_dir+"HR_gen"
if not os.path.exists(content_result):
    os.makedirs(content_result)
if not os.path.exists(HR_gen_result):
    os.makedirs(HR_gen_result)
if not os.path.exists(pb_path):
    os.makedirs(pb_path)
if not os.path.exists(ckpt_path):
    os.makedirs(ckpt_path)


def train():

    HR_input = tf.placeholder(tf.float32,shape=[None,image_height,image_width, 3],name="HR_input")
    is_training_place = tf.placeholder(tf.bool, shape=(),name="is_training")

    srgan = SRGAN(is_training_place,vgg_weight)
    LR = srgan.sample(HR_input)

    gen_loss, dis_loss, content_loss,psnr  = srgan.build_CartoonGAN(LR,HR_input)

    gen_vars, dis_vars = srgan.get_vars()
    global_step = tf.Variable(-1, trainable=False,name="global_step")
    global_step_increase = tf.assign(global_step, tf.add(global_step, 1))
    
    train_op_init = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999). \
                                            minimize(content_loss, var_list=gen_vars)
    train_op_G = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999). \
                                            minimize(gen_loss, var_list=gen_vars)
    train_op_D = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999). \
                                            minimize(dis_loss, var_list=dis_vars)
    
    generate_out = srgan.sample_generate(LR)
    generate_out = tf.identity(generate_out, name="generate_out")
    
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
    
        ckpt = tf.train.get_checkpoint_state(ckpt_path)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join(ckpt_path, ckpt_name))
    
        _global_step = sess.run(global_step_increase)
        dataLoader = CartoonGAN_Loader(image_dir, image_height, image_width,
                                    batch_size=batch_size,global_step=_global_step)
    
        while _global_step<pre_train_iter+gan_train_iter:
            images = dataLoader.next_batch()
            feed_dict = {HR_input: images, is_training_place: True}
            if _global_step<=pre_train_iter:
    
                sess.run(train_op_init, feed_dict=feed_dict)
                _vgg_loss,_psnr = sess.run([content_loss,psnr], feed_dict=feed_dict)
                if _global_step%50==0:
                    print("Step:{},Vgg_loss:{},PSNR:{}".format(_global_step,_vgg_loss,_psnr))
    
                if _global_step%100==0:
                    images = dataLoader.random_next_batch()
    
                    #save result form LR to HR(check init of content loss)
                    LR_images,_generate_out = sess.run([LR,generate_out],feed_dict={HR_input: images, is_training_place: True})
                    _generate_out = (_generate_out + 1) / 2 * 255.0
                    for ind,trg_image in enumerate(_generate_out[:sample_num]):
                        scipy.misc.imsave(content_result + "/{}_{}_real_HR.jpg".format(_global_step,ind), images[ind])
                        scipy.misc.imsave(content_result + "/{}_{}_real_LR.jpg".format(_global_step,ind), LR_images[ind])
                        scipy.misc.imsave(content_result + "/{}_{}_fake_HR.jpg".format(_global_step,ind), _generate_out[ind])
            else:
                sess.run(train_op_D, feed_dict=feed_dict)
                sess.run(train_op_G, feed_dict=feed_dict)
                
                _gen_loss,_dis_loss,_psnr = sess.run([gen_loss,dis_loss,psnr], feed_dict=feed_dict)
                if _global_step%50==0:
                    print("Step:{},Gen_loss:{},Dis_loss:{},PSNR:{}".format(_global_step, _gen_loss, _dis_loss,_psnr))
    
                if _global_step%500==0:
                    images = dataLoader.next_batch()
    
                    #save result form reality to cartoon
                    LR_images, _generate_out = sess.run([LR, generate_out], feed_dict={HR_input: images, is_training_place: True})
                    _generate_out = (_generate_out + 1) / 2 * 255.0
                    for ind,trg_image in enumerate(_generate_out[:sample_num]):
                        scipy.misc.imsave(HR_gen_result + "/{}_{}_real_HR.jpg".format(_global_step, ind), images[ind])
                        scipy.misc.imsave(HR_gen_result + "/{}_{}_real_LR.jpg".format(_global_step, ind), LR_images[ind])
                        scipy.misc.imsave(HR_gen_result + "/{}_{}_fake_HR.jpg".format(_global_step, ind), _generate_out[ind])
    
            if _global_step%20000==0:
                # save PB
                constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def,
                                                                           ["generate_out"])
                save_model_name = model_name + "-" + str(_global_step) + ".pb"
                with tf.gfile.FastGFile(pb_path + save_model_name, mode="wb") as fw:
                    fw.write(constant_graph.SerializeToString())
                # save CKPT
                saver.save(sess, ckpt_path + model_name + ".ckpt", global_step=_global_step)
                print("Successfully saved model {}".format(save_model_name))
            _global_step = sess.run(global_step_increase)

if __name__ == '__main__':
    train()