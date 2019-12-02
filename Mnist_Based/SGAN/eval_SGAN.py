import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import time

#SGAN_mnist-2000:判别器带有类别信息,其实就是就相当于DCGAN加入了类别损失,
#和improvedGAN思想也是一样的,就这种网络,都训练好几次了.[无语]
model_path = 'models_fashion/pb/SGAN_fashion-2000.pb'

image_height = 28
image_width = 28
prior_size=100
one_hot=np.eye(10)

def eval():
    sess = tf.Session()
    with tf.gfile.FastGFile(model_path, "rb") as fr:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(fr.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name="")

    sess.run(tf.global_variables_initializer())

    prior_input = sess.graph.get_tensor_by_name('z_prior:0')
    generated_output = sess.graph.get_tensor_by_name('generated_output:0')
    is_training = sess.graph.get_tensor_by_name('is_training:0')
    label_place = sess.graph.get_tensor_by_name('label:0')
    ind=0
    while True:
        #指定标签
        label = np.array([8])
        #随机标签
        # label = np.random.randint(0,10,size=[1,])

        # label_org = np.random.randint(0, 10, size=[1, 1])
        # label = one_hot[label_org].reshape([1, 1, 1, 10])

        z_prior = np.random.normal(0, 1, size=(1,prior_size))

        image_output = sess.run(generated_output,feed_dict={
                                prior_input:z_prior,label_place:label,
                                is_training:False})
        image_reshape_org = image_output[0].reshape((image_height,image_width))


        image_reshape = ((image_reshape_org+1)/2)*255.0
        image_show = image_reshape.astype(np.uint8)

        print("label:",label[0])
        cv2.imwrite("result_fashion/{}_{}.jpg".format(label[0], ind), image_show)
        image_show=cv2.resize(image_show,(image_height*2,image_width*2))
        cv2.imshow("image_fine", image_show)
        cv2.waitKey(0)
        ind+=1



if __name__ == '__main__':
    eval()