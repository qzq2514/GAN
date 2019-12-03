import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import time

#LapGAN_mnist_oneLevel-1700.pb:金字塔只有一层,prior_scales=[100],这是就相当于是一般的cGAN(向量噪声)
#LapGAN_mnist_independent3-1000.pb:各个生成器独立训练,独立生成,其实就是多个独立的cGAN(向量噪声)
#LapGAN_mnist_conv_3level-5600.pb:三级生成器各个独立训练,联合测试(无dropout)
#LapGAN_mnist_conv-5700.pb::三级生成器各个独立训练,联合测试(有dropout,效果提高点)
#LapGAN_mnist_conv_2_level-5600.pb:两级生成器各个独立训练,联合测试(无dropout,效果有点差,可自行使用带dropout进行训练)

#models_mnist/pb/LapGAN_mnist_3level-11400.pb
#models_fashion/pb/LapGAN_fashion_3level-16300.pb
model_path = 'models_fashion/pb/LapGAN_fashion_3level-16300.pb'

image_height = 28
image_width = 28
image_channal = 1
batch_size = 64
# prior_scales=[10*10, 10*10, 100] #LapGAN_mnist_independent3
prior_scales=[28*28, 14*14, 100] #LapGAN_mnist_conv_3_level
# prior_scales=[28*28, 100]   #LapGAN_mnist_conv_2_level-5600
# prior_scales=[100]        #LapGAN_mnist_oneLevel

one_hot=np.eye(10)

def eval():
    sess = tf.Session()
    with tf.gfile.FastGFile(model_path, "rb") as fr:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(fr.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name="")

    sess.run(tf.global_variables_initializer())

    z_prior_placeholders=[sess.graph.get_tensor_by_name(
                        'z_prior_{}:0'.format(ind)) for ind in range(len(prior_scales))]

    generated_output = sess.graph.get_tensor_by_name('generate_Laplace_out0:0')
    label_place = sess.graph.get_tensor_by_name('label:0')
    indx = 200
    while True:
        #指定标签
        # label_org = np.array([[5]])
        #随机标签
        label_org = np.random.randint(0,10,size=[batch_size,1])
        label = one_hot[label_org].squeeze(axis=1)
        feed_dict_train = {label_place: label}

        for ind, prior_size in enumerate(prior_scales):
            z_prior = np.random.normal(0, 1, size=(batch_size, prior_size * image_channal))
            feed_dict_train[z_prior_placeholders[ind]] = z_prior

        image_outputs = sess.run(generated_output,feed_dict=feed_dict_train)
        for ind,image_reshape in  enumerate(image_outputs):
            image_reshape = image_reshape.reshape((28,28))

            image_reshape = (image_reshape+1)/2*255
            image_reshape[image_reshape < 0] = 0
            image_reshape[image_reshape > 255] = 255
            image_show = image_reshape.astype(np.uint8)

            cur_label = label_org[ind,0]
            print("label:",cur_label,indx)
            # if label_org[0,0]==5:
            cv2.imwrite("result_fashion/{}_{}.jpg".format(cur_label,indx),image_show)
            image_show=cv2.resize(image_show,(image_height*2,image_width*2))
            cv2.imshow("image_show",image_show)
            cv2.waitKey(0)
            indx += 1



if __name__ == '__main__':
    eval()