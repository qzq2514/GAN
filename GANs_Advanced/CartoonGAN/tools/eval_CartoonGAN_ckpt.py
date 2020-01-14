import numpy as np
from scipy import misc
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import os

os.environ['CUDA_VISIBLE_DEVICES']='0'
model_path = '/media/cgim/dataset/models/CartoonGAN_mine_dis_loss/ckpt/CartoonGAN_mine_dis_loss.ckpt-100000'
image_dir = "/media/cgim/QZQUSB64G/qzq/CartoonGAN/CartoonGAN/test2/"
#/media/cgim/QZQUSB64G/qzq/CartoonGAN/CartoonGAN/test2
#/media/cgim/data/GAN/data/cartoonGAN_dataset2/Reality/
batch_size = 1
image_height = 96
image_width = 96
print("model_path:",model_path)
def eval():
    with tf.Session() as sess:
        ckpt_path = model_path
        saver = tf.train.import_meta_graph(ckpt_path + '.meta')
        saver.restore(sess, ckpt_path)

        input_reality_place = tf.get_default_graph().get_tensor_by_name('input_reality:0')
        is_training = tf.get_default_graph().get_tensor_by_name('is_training:0')

        Cartoon_output = tf.get_default_graph().get_tensor_by_name('generate_out:0')

        for image_name in os.listdir(image_dir):
            image_path = os.path.join(image_dir,image_name)
            reality = misc.imread(image_path)
            reality = misc.imresize(reality,[image_height,image_width])
            reality = reality[np.newaxis,:]
            

            cartoon_output = sess.run(Cartoon_output, feed_dict={input_reality_place: reality,is_training:False})
            cartoon_output = (cartoon_output + 1) / 2 * 255.0
            cartoon_output[cartoon_output>255]=255 
            cartoon_output[cartoon_output<0]=0 
            # cartoon_output = cartoon_output.astype(np.uint8)

            fig =plt.figure()
            ax1 = fig.add_subplot(121)
            ax1.imshow(np.uint8(reality[0]))
            misc.imsave("reality.jpg",reality[0])
            ax2 = fig.add_subplot(122)
            ax2.imshow(np.uint8(cartoon_output[0]))
            misc.imsave("cartoon_output.jpg",cartoon_output[0])
            plt.show()

if __name__ == '__main__':
    eval()
