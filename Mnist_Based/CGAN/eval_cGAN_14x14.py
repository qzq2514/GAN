import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import time

model_path = 'models/pb/cGAN_mnist_14x14-600.pb'

image_height = 14
image_width = 14
image_flatten_size = image_height*image_width
prior_size=100
sample_num=64
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
    label_place = sess.graph.get_tensor_by_name('label:0')
    while True:
        #指定标签
        # label_org = np.array([[3]])
        #随机标签
        label_org = np.random.randint(0,10,size=[sample_num,1])

        label = one_hot[label_org].squeeze(axis=1)
        z_prior = np.random.normal(0, 1, size=(sample_num, prior_size))
        image_output = sess.run(generated_output,feed_dict={
                                prior_input:z_prior,label_place:label})

        image_reshape_org = image_output[0].reshape((image_height, image_width))

        #因为输出的元素值范围在-1~1之,所以正确的应该是先加1除以2再乘以255
        #上面直接乘以255虽然也能达到效果,但是不准确
        image_reshape = ((image_reshape_org + 1) / 2) * 255.0
        image_show = image_reshape.astype(np.uint8)

        print("label:", label_org[0, 0])
        image_show = cv2.resize(image_show, (image_height , image_width ))
        cv2.imshow("image_fine", image_show)
        cv2.waitKey(0)



if __name__ == '__main__':
    eval()