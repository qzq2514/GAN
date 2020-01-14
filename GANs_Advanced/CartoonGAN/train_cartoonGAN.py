from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.framework import graph_util
import tensorflow.contrib.slim as slim
from DataLoader import CartoonGAN_Loader
from net.CartoonGAN import CartoonGAN
import tensorflow as tf
import numpy as np
import scipy.misc
import os

os.environ['CUDA_VISIBLE_DEVICES']='1'

image_height = 96
image_width = 96
batch_size = 16
sample_num = 10
pool_size=50

pre_train_iter = 20000
gan_train_iter = 100005
learning_rate = 1e-4

vgg_weight = 5e3
decay=0.9
image_dir = "/media/cgim/data/GAN/data/cartoonGAN_dataset2/"
model_name = "CartoonGAN"
model_path="/media/cgim/dataset/models/"+model_name

pb_path=os.path.join(model_path,"pb/")
ckpt_path=os.path.join(model_path,"ckpt/")
result_dir=model_path+"/result/"
content_result=result_dir+"content"
cartoon_gen_result=result_dir+"cartoon_gen"
if not os.path.exists(content_result):
    os.makedirs(content_result)
if not os.path.exists(cartoon_gen_result):
    os.makedirs(cartoon_gen_result)
if not os.path.exists(pb_path):
    os.makedirs(pb_path)
if not os.path.exists(ckpt_path):
    os.makedirs(ckpt_path)


def train():

    reality_place = tf.placeholder(tf.float32,shape=[None,image_height,image_width, 3],name="input_reality")
    cartoon_place = tf.placeholder(tf.float32, shape=[None, image_height,image_width, 3], name="input_cartoon")
    smooth_cartoon_place = tf.placeholder(tf.float32, shape=[None, image_height, image_width, 3], name="input_smooth_cartoon")
    is_training_place = tf.placeholder(tf.bool, shape=(),name="is_training")

    cartoonGAN = CartoonGAN(is_training_place,vgg_weight)

    gen_loss, dis_loss, vgg_loss  = cartoonGAN.build_CartoonGAN(reality_place,cartoon_place,smooth_cartoon_place)

    gen_vars, dis_vars = cartoonGAN.get_vars()
    global_step = tf.Variable(-1, trainable=False,name="global_step")
    global_step_increase = tf.assign(global_step, tf.add(global_step, 1))

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op_init = tf.train.AdamOptimizer(learning_rate, beta1=0., beta2=0.9). \
            minimize(vgg_loss, var_list=gen_vars,colocate_gradients_with_ops=True)
        train_op_G = tf.train.AdamOptimizer(learning_rate*2, beta1=0., beta2=0.9). \
            minimize(gen_loss, var_list=gen_vars,colocate_gradients_with_ops=True)
        train_op_D = tf.train.AdamOptimizer(learning_rate, beta1=0., beta2=0.9). \
            minimize(dis_loss, var_list=dis_vars,colocate_gradients_with_ops=True)

    generate_out = cartoonGAN.sample_generate(reality_place)
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
            images_cartoon, images_smooth_cartoon, images_reality=dataLoader.next_batch()
            if _global_step<=pre_train_iter:
                feed_dict_init = {reality_place:images_reality,is_training_place:True,
                                  smooth_cartoon_place:images_smooth_cartoon,cartoon_place:images_cartoon}
                sess.run(train_op_init, feed_dict=feed_dict_init)
                _vgg_loss = sess.run(vgg_loss, feed_dict=feed_dict_init)
                if _global_step%50==0:
                    print("Step:{},Vgg_loss:{}".format(_global_step,_vgg_loss))

                if _global_step%100==0:
                    images_cartoon, images_smooth_cartoon, images_reality = dataLoader.random_next_batch()
                    feed_dict_init_test = {reality_place:images_reality,is_training_place:True}

                    #save result form reality to cartoon(check init of content loss)
                    _generate_out = sess.run(generate_out,feed_dict=feed_dict_init_test)
                    _generate_out = (_generate_out + 1) / 2 * 255.0
                    for ind,trg_image in enumerate(_generate_out[:sample_num]):
                        scipy.misc.imsave(content_result + "/{}_{}_reality.jpg".format(_global_step,ind), images_reality[ind])
                        scipy.misc.imsave(content_result + "/{}_{}_reality_content.jpg".format(_global_step,ind), _generate_out[ind])
            else:
                feed_dict_GAN = {reality_place:images_reality,cartoon_place:images_cartoon,
                              smooth_cartoon_place:images_smooth_cartoon, is_training_place:True}

                sess.run(train_op_G, feed_dict=feed_dict_GAN) 
                sess.run(train_op_D, feed_dict=feed_dict_GAN)
                
                _gen_loss,_dis_loss = sess.run([gen_loss,dis_loss], feed_dict=feed_dict_GAN)
                if _global_step%50==0:
                    print("Step:{},Gen_loss:{},Dis_loss:{}".format(_global_step, _gen_loss, _dis_loss))

                if _global_step%500==0:
                    images_cartoon, images_smooth_cartoon, images_reality = dataLoader.random_next_batch()

                    #save result form reality to cartoon
                    _generate_out = sess.run(generate_out,feed_dict={reality_place:images_reality,
                                                                      is_training_place:True})
                    _generate_out = (_generate_out + 1) / 2 * 255.0
                    for ind,trg_image in enumerate(_generate_out[:sample_num]):
                        scipy.misc.imsave(cartoon_gen_result + "/{}_{}_reality.jpg".format(_global_step,ind), images_reality[ind])
                        scipy.misc.imsave(cartoon_gen_result + "/{}_{}_cartoon.jpg".format(_global_step,ind), _generate_out[ind])

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