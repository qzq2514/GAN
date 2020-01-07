import numpy as np
import tensorflow as tf
from DataLoader import Pix2Pix_loader
import matplotlib.pyplot as plt
import cv2
import os

os.environ['CUDA_VISIBLE_DEVICES']='0'

model_path = '/media/cgim/dataset/models/Pix2Pix/pb/CycleGAN_frozen_model.pb'
image_dir = "/media/cgim/data/GAN/data/edges2shoes/"
batch_size = 1
image_height = 64
image_width = 64

def eval():
    sess = tf.Session()
    with tf.gfile.FastGFile(model_path, "rb") as fr:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(fr.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name="")

    sess.run(tf.global_variables_initializer())

    input_A_place = sess.graph.get_tensor_by_name('input_A:0')

    A2B_output = sess.graph.get_tensor_by_name('A2B_output:0')
    keep_prob_place = sess.graph.get_tensor_by_name('keep_prob:0')
    is_training = sess.graph.get_tensor_by_name('is_training:0')

    dataLoader = Pix2Pix_loader(image_dir, image_height, image_width, batch_size=batch_size)

    while True:
        images_A, images_B = dataLoader.random_next_test_batch()

        _A2B_output = sess.run(A2B_output, feed_dict={input_A_place: images_A,
                                        is_training:False,keep_prob_place:1.0})
        print(_A2B_output)
        _A2B_output = (_A2B_output + 1) / 2 * 255.0
        _A2B_output = _A2B_output.astype(np.uint8)

		fig =plt.figure()
        ax1 = fig.add_subplot(131)
        ax1.imshow(np.uint8(images_A[0]))
        ax2 = fig.add_subplot(132)
        ax2.imshow(np.uint8(images_B[0]))
        ax3 = fig.add_subplot(133)
        ax3.imshow(np.uint8(_A2B_output[0]))

        plt.show()

if __name__ == '__main__':
    eval()