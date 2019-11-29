import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.framework import graph_util
import tensorflow.contrib.slim as slim
from net import BEGAN_mnist
import numpy as np
import time
import cv2
import os



#data para
image_height = 28
image_width = 28
prior_size=62
num_class=10
channels=1


#train para
epochs = 2000
max_step = 2000000
batch_size = 64
d_learning_rate = 0.0002
g_learning_rate = 0.0002*5
discriminator_train_intervel=1
snapshot = 100
beta1=0.5

#BEGAN para
#gamma代表生成图像的重构误差和真实图像重构误差的比例,同时其越高,
#在优化损失中生成图像的损失占比就越高,故其也可以代表生成图像的多样性比例
gamma = 0.75
lamda = 0.001


image_path = "../Data/MNIST_fashion"
model_path="models_fashion/"
pb_path=os.path.join(model_path,"pb/")
ckpt_path=os.path.join(model_path,"ckpt/")
result_dir = "fashion_result"
if not os.path.exists(pb_path):
    os.makedirs(pb_path)
if not os.path.exists(ckpt_path):
    os.makedirs(ckpt_path)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
model_name="BEGAN_fashion"

def train():

    real_data_placeholder = tf.placeholder(tf.float32,shape=[None,image_height,image_width,channels],name="real_data")
    z_prior_placeholder = tf.placeholder(tf.float32,shape=[None,prior_size],name="z_prior")
    is_training_placeholder = tf.placeholder(tf.bool,shape=(),name="is_training")

    mnist_GANer=BEGAN_mnist.BEGAN_mnist(is_training_placeholder,num_class,batch_size,image_height,
                                        image_width,channels,gamma,lamda)

    generated_data = mnist_GANer.generator(z_prior_placeholder)
    print("generated_data:",generated_data)
    _generated_data = tf.identity(generated_data,name="generated_output")

    real_recon_out, real_recon_error, real_code = mnist_GANer.discriminator(real_data_placeholder)
    fake_recon_out, fake_recon_error, fake_code = mnist_GANer.discriminator(generated_data,reuse=True)

    g_loss, d_loss ,metric,update_k= mnist_GANer.loss(real_recon_error,
                                        fake_recon_error)
    g_vars, d_vars = mnist_GANer.get_vars()

    global_step = tf.Variable(0, trainable=False)
    # optimizers
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        d_train_opt = tf.train.AdamOptimizer(d_learning_rate, beta1=beta1) \
            .minimize(d_loss,var_list=d_vars)
        g_train_opt = tf.train.AdamOptimizer(g_learning_rate, beta1=beta1) \
            .minimize(g_loss, global_step=global_step,var_list=g_vars)

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

        #show model
        slim.model_analyzer.analyze_vars(var_list, print_info=True)

        step_per_epoch = mnist.train.num_examples//batch_size
        start_time = time.time()
        for _ in range(epochs):
            is_break=False
            for _ in range(step_per_epoch):

                batch = mnist.train.next_batch(batch_size)
                batch_images = batch[0].reshape((batch_size,image_height,image_width,channels))

                #图像像素值范围0~1,所以后面生成器和判别器的输出层的激活函数要用sigmod
                batch_images = batch_images
                # print(np.min(batch_images),"~",np.max(batch_images))
                batch_z_prior = np.random.uniform(-1, 1, [batch_size, prior_size]).astype(np.float32)

                full_dict={real_data_placeholder: batch_images,
                           z_prior_placeholder: batch_z_prior,
                           is_training_placeholder: True}
                #更新判别器
                for _ in range(discriminator_train_intervel):
                    train_loss_d,_ = sess.run([d_loss,d_train_opt],
                                      feed_dict=full_dict)
                # 更新生成器
                global_step_,train_loss_g, _ = sess.run([global_step,g_loss, g_train_opt],
                                       feed_dict={z_prior_placeholder:batch_z_prior,
                                                  is_training_placeholder: True})

                #更新K值并计算收敛指标:
                _, _metric, _k_value = sess.run([update_k,metric,mnist_GANer.k],feed_dict=full_dict)

                epoche_num =  global_step_//step_per_epoch
                step = global_step_%step_per_epoch
                print("Epoch:[{}] ,Step:[{}/{}] ,time:{} s ,D_Loss: {:.4f} ,G_Loss: {:.4f},"
                      "Metric:{},K: {}".format(epoche_num,step ,step_per_epoch,
                    time.time()-start_time,train_loss_d, train_loss_g,_metric,_k_value))

                #使用batch_norm就要查看其中norm的均值或方差有没有稳定
                one_moving_meaning_show = "No mean or variance"
                if len(bn_moving_vars) > 0:
                    one_moving_meaning = sess.graph.get_tensor_by_name(bn_moving_vars[0].name)
                    one_moving_meaning_show = np.mean(one_moving_meaning.eval())
                print("one_moving_meaning:", one_moving_meaning_show)

                if global_step_ % snapshot == 0:
                    #显示中间结果
                    samples = sess.run(generated_data, feed_dict={z_prior_placeholder: batch_z_prior,
                                                                  is_training_placeholder: False})
                    for ind in range(len(samples))[:20]:
                        image_reshape_org = samples[ind].reshape((image_height, image_width))
                        image_reshape = image_reshape_org * 255.0
                        cv2.imwrite(result_dir+"/epoch{}_{}_{}.jpg".format(epoche_num,step, ind), image_reshape)

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