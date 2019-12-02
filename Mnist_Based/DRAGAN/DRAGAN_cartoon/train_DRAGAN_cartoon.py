from tensorflow.python.framework import graph_util
import tensorflow.contrib.slim as slim
from net import DRAGAN_cartoon
import tensorflow as tf
import numpy as np
import DataLoader
import time
import cv2
import os


image_height = 64
image_width = 64
prior_size=200

class_num=10
channels=3
batch_size = 64
epochs = 10
learning_rate = 0.0002
discriminator_train_intervel=1
snapshot = 100
beta1 = 0.5

#DRAGAN 参数
lambd = 0.25    #lambd:梯度惩罚在总体损失中的权重(该lambd和WGAN-GP中的意义一致)

model_path="models/"
image_dir="D:/forTensorflow/cartoon"
pb_path=os.path.join(model_path,"pb/")
ckpt_path=os.path.join(model_path,"ckpt/")
result_dir = "result"
if not os.path.exists(pb_path):
    os.makedirs(pb_path)
if not os.path.exists(ckpt_path):
    os.makedirs(ckpt_path)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
model_name="DRAGAN_cartoon"

def train():
    dataLoader = DataLoader.Cartoo_loader(image_dir, batch_size, image_height, image_width, channels)
    real_data_placeholder = tf.placeholder(tf.float32,shape=[None,image_height,
                                            image_width,channels],name="real_data")
    z_prior_placeholder = tf.placeholder(tf.float32,shape=[None,prior_size],name="z_prior")
    is_training_placeholder = tf.placeholder(tf.bool,shape=(),name="is_training")

    GANer=DRAGAN_cartoon.DRAGAN_cartoon(is_training_placeholder,batch_size,
                                   image_height,image_width,channels,lambd)

    generated_data = GANer.generator(z_prior_placeholder)
    _generated_data = tf.identity(generated_data,name="generated_output")

    real_prop,real_logist = GANer.discriminator(real_data_placeholder)
    generated_prop,generated_logits = GANer.discriminator(generated_data,reuse=True)


    g_loss, d_loss = GANer.loss(real_prop, generated_prop, real_data_placeholder)
    g_vars, d_vars = GANer.get_vars()
    global_step = tf.Variable(0, trainable=False)

    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        g_train_opt = tf.train.AdamOptimizer(learning_rate,beta1=beta1).\
                          minimize(g_loss,global_step=global_step, var_list = g_vars)
        d_train_opt = tf.train.AdamOptimizer(learning_rate*10,beta1=beta1).\
                          minimize(d_loss, var_list = d_vars)


    #保存所有参数,以便后面进行断点重新训练
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        ckpt = tf.train.get_checkpoint_state(ckpt_path)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join(ckpt_path, ckpt_name))
            print("Successfully reload model:",ckpt_name )

        var_list = tf.trainable_variables()
        g_list = tf.global_variables()  # 从全局变量中获得batch norm的缩放和偏差
        bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
        bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
        var_list += bn_moving_vars

        slim.model_analyzer.analyze_vars(var_list, print_info=True)

        step_per_epoch = dataLoader.sample_num // batch_size
        start_time = time.time()
        for _ in range(epochs):
            for _ in range(step_per_epoch):
                batch_images = dataLoader.next_batch(True, None)

                batch_images = (batch_images / 255.0) * 2 - 1
                batch_z_prior = np.random.uniform(-1, 1,[batch_size, prior_size]).astype(np.float32)

                #更新判别器
                feed_dict={real_data_placeholder:batch_images,
                           z_prior_placeholder:batch_z_prior,
                           is_training_placeholder:True}

                for _ in range(discriminator_train_intervel):
                    _,global_step_,train_loss_d = sess.run([d_train_opt,global_step,d_loss],
                                                        feed_dict=feed_dict)

                batch_z_prior = np.random.uniform(-1, 1, [batch_size, prior_size]).astype(np.float32)
                train_loss_g, _ = sess.run([g_loss, g_train_opt],feed_dict={
                                                    z_prior_placeholder: batch_z_prior,
                                                    is_training_placeholder: True })
                epoche_num = global_step_ // step_per_epoch
                step = global_step_ % step_per_epoch
                print("Epoch:[{}] ,Step:[{}/{}] ,time:{} s ,D_Loss: {:.4f} ,G_Loss: {:.4f}".format(
                    epoche_num, step, step_per_epoch,
                    time.time() - start_time, train_loss_d, train_loss_g))

                #使用batch_norm就要查看其中norm的均值或方差有没有稳定
                one_moving_meaning_show = "No mean or variance"
                if len(bn_moving_vars) > 0:
                    one_moving_meaning = sess.graph.get_tensor_by_name(bn_moving_vars[0].name)
                    one_moving_meaning_show = np.mean(one_moving_meaning.eval())
                print("one_moving_meaning:", one_moving_meaning_show)

                if global_step_ % snapshot == 0:
                    # 显示中间结果
                    samples = sess.run(generated_data, feed_dict={z_prior_placeholder: batch_z_prior,
                                                                  is_training_placeholder: False})
                    for ind in range(len(samples))[:20]:
                        image_reshape_org = samples[ind].reshape((image_height, image_width,channels))
                        image_reshape = ((image_reshape_org + 1) / 2) * 255.0
                        cv2.imwrite(result_dir + "/epoch{}_{}_{}.jpg".format(
                            epoche_num, step, ind), image_reshape)

                    # 保存PB
                    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ["generated_output"])
                    save_model_name = model_name + "-" + str(global_step_) + ".pb"
                    with tf.gfile.FastGFile(pb_path + save_model_name, mode="wb") as fw:
                        fw.write(constant_graph.SerializeToString())
                    # 保存CKPT
                    saver.save(sess, ckpt_path + model_name + ".ckpt", global_step=global_step_)
                    print("Successfully saved model {}".format(save_model_name))

if __name__ == '__main__':
    train()