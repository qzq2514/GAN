import numpy as np
import tensorflow as tf
from DataLoader import Pix2Pix_loader
import matplotlib.pyplot as plt
import cv2
import os

os.environ['CUDA_VISIBLE_DEVICES']='0'
model_path = '/media/cgim/dataset/models/Pix2Pix/ckpt/Pix2Pix.ckpt-100000'
image_dir = "/media/cgim/data/GAN/data/edges2shoes/"
batch_size = 1
image_height = 64
image_width = 64

def eval():
    with tf.Session() as sess:
        ckpt_path = model_path
        saver = tf.train.import_meta_graph(ckpt_path + '.meta')
        saver.restore(sess, ckpt_path)

        input_A_place = tf.get_default_graph().get_tensor_by_name('input_A:0')
        keep_prob_place = tf.get_default_graph().get_tensor_by_name('keep_prob:0')
        is_training = tf.get_default_graph().get_tensor_by_name('is_training:0')

        A2B_output = tf.get_default_graph().get_tensor_by_name('A2B_output:0')

        dataLoader = Pix2Pix_loader(image_dir, image_height, image_width, batch_size=batch_size)

        while True:
            images_A, images_B = dataLoader.random_next_test_batch()

            _A2B_output = sess.run(A2B_output, feed_dict={input_A_place: images_A,
                                            is_training:False,keep_prob_place:1.0})
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