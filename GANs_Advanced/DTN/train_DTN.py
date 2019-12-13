from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.framework import graph_util
import tensorflow.contrib.slim as slim
from DataLoader import SVHN_loader
import tensorflow as tf
from net.DTN import DTN
import numpy as np
import cv2
import os



#data para
image_height = 32
image_width = 32
batch_size_pre = 128
batch_size = 64
src_channals = 3
trg_channals = 1
snapshot = 100

SVHN_data_dir="D:/forTensorflow/forGAN/SVHN"
MNIST_data_dir ="D:/forTensorflow/forGAN/MNIST_data"

model_path="models/"
pb_path=os.path.join(model_path,"pb/")
ckpt_path=os.path.join(model_path,"ckpt/")
result_dir="result"
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
if not os.path.exists(pb_path):
    os.makedirs(pb_path)
if not os.path.exists(ckpt_path):
    os.makedirs(ckpt_path)
model_name = "DTN"

def resize_mnist(images,height,width):
    res_images = np.zeros([images.shape[0],height,width,trg_channals])
    for ind,image in enumerate(images):
        image = cv2.resize(image,(width,height))
        image = image[:,:,np.newaxis]
        res_images[ind] = image
    return res_images

def mnist_next_batch(data,batch_size):
    batch_images, batch_labels = data.next_batch(batch_size)
    batch_images = batch_images.reshape((batch_size, 28, 28, trg_channals))
    batch_images = resize_mnist(batch_images, image_height, image_width)
    batch_images = batch_images*2-1   #归一化到-1~1
    return batch_images,batch_labels

def train():
    source_image_place = tf.placeholder(tf.float32,shape=[None,image_height,
                                    image_width,src_channals],name="svhn_image")
    source_label_place = tf.placeholder(tf.int64,shape=[None],name="svhn_label")
    target_image_place = tf.placeholder(tf.float32, shape=[None, image_height,
                                    image_width, trg_channals], name="mnist_image")
    is_training_place = tf.placeholder_with_default(False, shape=(), name="is_training")

    
    mnist = input_data.read_data_sets(MNIST_data_dir)

    dtn = DTN(is_training = is_training_place)

    preTrain_op,preTrain_loss,preTrain_accu = dtn.build_model_preTrain(
        source_image_place,source_label_place)

    dtn.build_model_train(source_image_place,target_image_place)

    generated_output = dtn.generateor(source_image_place)
    _generated_output = tf.identity(generated_output, name="generated_output")

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        ckpt = tf.train.get_checkpoint_state(ckpt_path)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join(ckpt_path, ckpt_name))

        svhn_loader_pre = SVHN_loader(SVHN_data_dir,batch_size_pre)
        while True:
            batch_images,batch_labels = svhn_loader_pre.next_batch()      #0~255
            # batch_images,batch_labels = mnist_next_batch(mnist.train,batch_size_pre)     #-1~1
            sess.run(preTrain_op, feed_dict={source_image_place: batch_images,
                                            source_label_place: batch_labels,
                                            is_training_place: True})

            batch_images_test, batch_labels_test = svhn_loader_pre.random_next_test_batch()
            # batch_images_test,batch_labels_test = mnist_next_batch(mnist.test,batch_size_pre)

            _loss, _accu = sess.run([preTrain_loss, preTrain_accu], feed_dict=
                                                {source_image_place: batch_images_test,
                                                 source_label_place: batch_labels_test,
                                                 is_training_place: False})
            print("PreTraining----Loss:{},Accu:{}".format(_loss, _accu))
            if _accu>0.95:
                break

        svhn_loader = SVHN_loader(SVHN_data_dir,batch_size)
        train_step = 0
        while True:
            batch_images_svhn, _ = svhn_loader.next_batch()
            batch_images_mnist, _ = mnist_next_batch(mnist.train,batch_size)
            feed_dict_src = {source_image_place: batch_images_svhn,
                                is_training_place: True}

            sess.run(dtn.d_train_op_src, feed_dict=feed_dict_src)
            for _ in range(5):
                sess.run(dtn.g_train_op_src, feed_dict=feed_dict_src)

            if train_step%15==0:
                sess.run(dtn.f_train_op_src, feed_dict=feed_dict_src)

            _d_loss_src, _g_loss_src, _f_loss_src = \
                sess.run([dtn.d_loss_src,dtn.g_loss_src,dtn.f_loss_src],
                         feed_dict=feed_dict_src)

            print("Training[src]----Step:{},D_loss:{},G_loss:{},F_loss:{}".format(
                train_step,_d_loss_src,_g_loss_src,_f_loss_src))

            feed_dict_all = {source_image_place: batch_images_svhn,
                             target_image_place:batch_images_mnist,
                             is_training_place: True}
            sess.run(dtn.d_train_op_trg, feed_dict=feed_dict_all)
            sess.run(dtn.d_train_op_trg, feed_dict=feed_dict_all)
            for _ in range(4):
                sess.run(dtn.g_train_op_trg, feed_dict=feed_dict_all)

            _d_loss_trg, _g_loss_trg = sess.run([dtn.d_loss_trg, dtn.g_loss_trg],
                                        feed_dict=feed_dict_all)

            print("Training[trg]----Step:{},D_loss:{},G_loss:{}".format(
                train_step, _d_loss_trg, _g_loss_trg))

            if train_step%snapshot==0:
                gengrated_images = sess.run(_generated_output,feed_dict={
                                        source_image_place:batch_images_svhn})
                results = (gengrated_images + 1) / 2 * 255.0

                for ind,trg_image in enumerate(results[:20]):
                    cv2.imwrite(result_dir + "/{}_{}_src.jpg".format(train_step,ind),batch_images_svhn[ind])
                    cv2.imwrite(result_dir + "/{}_{}_trg.jpg".format(train_step, ind), results[ind])

                # 保存PB
                constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ["generated_output"])
                save_model_name = model_name + "-" + str(train_step) + ".pb"
                with tf.gfile.FastGFile(pb_path + save_model_name, mode="wb") as fw:
                    fw.write(constant_graph.SerializeToString())
                # 保存CKPT
                saver.save(sess, ckpt_path + model_name + ".ckpt", global_step=train_step)
                print("Successfully saved model {}".format(save_model_name))

            train_step += 1

            


if __name__ == '__main__':
    train()
