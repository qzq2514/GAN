from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os


class CartoonGAN_Loader:
    def __init__(self, image_dir, image_height, image_width, batch_size, global_step=0):

        self.fileList = self.get_filePath(image_dir)
        self.shuffle_dataset()

        self.image_height = image_height
        self.image_width = image_width
        self.batch_size = batch_size

        self.sample_num = len(self.fileList)
        self.step_per_epoch = self.sample_num // self.batch_size
        self.batch_id = global_step % self.step_per_epoch

    def get_filePath(self, dir):
        file_paths = [os.path.join(dir, file_name) for file_name in os.listdir(dir)]
        return np.array(file_paths)

    def shuffle_dataset(self):
        shuffle_ind = np.arange(0, len(self.fileList))
        np.random.shuffle(shuffle_ind)
        self.fileList = self.fileList[shuffle_ind]

    def next_batch_core(self, samples_id_range):
        images = []

        for id in samples_id_range:
            image = misc.imread(self.fileList[id])
            images.append(misc.imresize(image, [self.image_height, self.image_width]))

        images = np.array(images)

        return images

    def random_next_batch(self):
        indices = np.random.choice(self.sample_num, self.batch_size)
        return self.next_batch_core(indices)

    def next_batch(self):
        if self.batch_id >= self.step_per_epoch:
            self.batch_id = 0
            self.shuffle_dataset()
        bacth_ind_start = self.batch_id * self.batch_size
        bacth_ind_end = (self.batch_id + 1) * self.batch_size
        batch_id_range = np.arange(bacth_ind_start,bacth_ind_end)
        self.batch_id += 1
        return self.next_batch_core(batch_id_range)

if __name__ == '__main__':
    image_dir="D:/forTensorflow/forGAN/CartoonPictures/AllCartoon_crop/"
    dataLoader = CartoonGAN_Loader(image_dir, 512, 512,batch_size=16,global_step=0)
    while True:
        print("----------")
        images = dataLoader.next_batch()
        plt.imshow(images[0])
        plt.show()
