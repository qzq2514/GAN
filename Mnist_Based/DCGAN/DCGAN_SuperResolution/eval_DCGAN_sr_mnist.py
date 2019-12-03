import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import cv2

#models_fashion/pb/DCGAN_fashion_sr-2000.pb
#models_new/pb/DCGAN_mnist_sr-2600.pb
model_path = 'models_fashion/pb/DCGAN_fashion_sr-2000.pb'

image_height = 28
image_width = 28
channels=1
one_hot=np.eye(10)
image_path = "../../Data/MNIST_fashion"
def get_corase_image(org_image):

    down_sample = cv2.resize(org_image,(image_width//2,image_height//2))
    up_sample = cv2.resize(down_sample,(image_width,image_height))
    up_sample = up_sample.reshape((image_width,image_height,channels))

    return up_sample

def eval():
    mnist = input_data.read_data_sets(image_path, one_hot=True)
    real_datas,real_labels = mnist.test.next_batch(200)

    sess = tf.Session()
    with tf.gfile.FastGFile(model_path, "rb") as fr:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(fr.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name="")

    sess.run(tf.global_variables_initializer())

    prior_input = sess.graph.get_tensor_by_name('z_prior:0')
    generated_output = sess.graph.get_tensor_by_name('generated_output:0')
    corase_data_place = sess.graph.get_tensor_by_name('corase_data:0')
    ind = 0
    for real_data in real_datas:
        real_data = real_data.reshape([image_height,image_width,channels])
        z_prior = np.random.uniform(1, 1, size=(1,image_height, image_width,1))

        corase_real_data =get_corase_image(real_data)
        corase_real_data = corase_real_data[np.newaxis,:]
        corase_real_data = corase_real_data * 2 - 1

        residual_output = sess.run(generated_output,
                        feed_dict={corase_data_place:corase_real_data,
                                   prior_input:z_prior})

        fine_sample = ((residual_output + corase_real_data + 1 ) / 2) * 255.0
        fine_sample[fine_sample < 0] = 0
        fine_sample[fine_sample > 255] = 255
        fine_sample = fine_sample.astype(np.uint8)
        print(fine_sample.shape)

        corase_image = ((corase_real_data + 1) / 2) * 255.0
        corase_image = corase_image.astype(np.uint8)

        real_data *= 255.0
        real_data = real_data.astype(np.uint8)

        image_save = np.zeros(shape=[image_height, image_width * 3, channels])
        image_save[0:image_height, 0:image_width, ...] = corase_image
        image_save[0:image_height, image_width:image_width * 2, ...] = fine_sample
        image_save[0:image_height, image_width * 2:image_width * 3, ...] = real_data
        cv2.imwrite("compare_res_fashion/{}.jpg".format(ind), image_save)


        image_org = cv2.resize(real_data, (image_height * 2, image_width * 2))
        corase_image = cv2.resize(corase_image[0], (image_height * 2, image_width * 2))
        fine_sample=cv2.resize(fine_sample[0],(image_height*2,image_width*2))

        cv2.imshow("image_org",image_org)
        cv2.imshow("corase_image_show", corase_image)
        cv2.imshow("fine_sample", fine_sample)

        cv2.waitKey(0)
        ind+=1

if __name__ == '__main__':
    eval()