import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import os
import time

"models/pb/InfoGAN_mnist_38noise_1-2000.pb"
'models_fashion/pb/InfoGAN_fashion_38noise_-2000.pb'
model_path = 'models/pb/InfoGAN_mnist_38noise_1-2000.pb'

image_height = 28
image_width = 28
prior_size=38
sample_num = 10
latent_code_size = 2
one_hot=np.eye(10)
tabel_height = sample_num*image_height
tabel_width = sample_num*image_width
image_channal = 1
def eval():
    sess = tf.Session()
    with tf.gfile.FastGFile(model_path, "rb") as fr:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(fr.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name="")

    sess.run(tf.global_variables_initializer())

    prior_input = sess.graph.get_tensor_by_name('z_prior:0')
    latent_code_input = sess.graph.get_tensor_by_name('latent_code:0')
    generated_output = sess.graph.get_tensor_by_name('generated_output:0')
    label_place = sess.graph.get_tensor_by_name('label:0')

    imgShow = np.zeros(shape=[tabel_height, tabel_width,image_channal])
    ind=1
    while True:
        #指定标签
        # label_org = np.array([[3]])
        #随机标签
        label_org = np.random.randint(0, 10, size=[1, ])
        z_prior = np.random.uniform(-1, 1, size=(1, prior_size))

        for x,laten1 in enumerate(np.linspace(-1.0, 1.0, sample_num)):
            for y,laten2 in enumerate(np.linspace(-1.0, 1.0, sample_num)):
                latent_code = np.array([laten1,laten2])
                latent_code = latent_code[np.newaxis,:]
                image_output = sess.run(generated_output,feed_dict={
                                        prior_input:z_prior,label_place:label_org,
                                        latent_code_input:latent_code})

                image_reshape_org = image_output[0].reshape((image_height,image_width,image_channal))
                image_reshape = ((image_reshape_org + 1) / 2) * 255.0

                x_start = x * image_width
                x_end = (x + 1) * image_width
                y_start = y * image_height
                y_end = (y + 1) * image_height
                imgShow[y_start:y_end, x_start:x_end, :] = image_reshape
                print("----------",x,"  ",y)
        imgShow = np.array(imgShow, np.uint8)
        cv2.imwrite("tabel_mnist/imageTable{}.jpg".format(ind),imgShow)
        cv2.imshow("imageTable",imgShow)
        cv2.waitKey(0)
        ind+=1

if __name__ == '__main__':
    eval()