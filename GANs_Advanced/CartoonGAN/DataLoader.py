from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os


class CartoonGAN_Loader:
    def __init__(self, root, image_height, image_width, batch_size, global_step=0):
        print("CartoonGAN_Loader")
        self.cartoon_fileList = self.get_filePath(os.path.join(root, "Cartoon"))
        self.reality_fileList = self.get_filePath(os.path.join(root, "Reality"))
        self.shuffle_dataset()

        self.image_height = image_height
        self.image_width = image_width
        self.batch_size = batch_size

        self.sample_num = min(len(self.cartoon_fileList),len(self.reality_fileList))
        self.step_per_epoch = self.sample_num // self.batch_size
        self.batch_id = global_step % self.step_per_epoch

    def get_filePath(self, dir):
        file_paths = [os.path.join(dir, file_name) for file_name in os.listdir(dir)]
        return np.array(file_paths)

    def shuffle_dataset(self):
        shuffle_ind = np.arange(0, len(self.cartoon_fileList))
        np.random.shuffle(shuffle_ind)
        self.cartoon_fileList = self.cartoon_fileList[shuffle_ind]

        shuffle_ind = np.arange(0, len(self.reality_fileList))
        np.random.shuffle(shuffle_ind)
        self.reality_fileList = self.reality_fileList[shuffle_ind]

    def next_batch_core(self, samples_id_range):
        cartoon_images = []
        smooth_cartoon_images = []
        reality_images = []

        for id in samples_id_range:
            cartoon_image = misc.imread(self.cartoon_fileList[id])
            smooth_cartoon_image = cv2.GaussianBlur(cartoon_image, (5, 5), 0)
            reality_image = misc.imread(self.reality_fileList[id])

            cartoon_images.append(misc.imresize(cartoon_image, [self.image_height, self.image_width]))
            smooth_cartoon_images.append(misc.imresize(smooth_cartoon_image, [self.image_height, self.image_width]))
            reality_images.append(misc.imresize(reality_image, [self.image_height, self.image_width]))

        cartoon_images = np.array(cartoon_images)
        smooth_cartoon_images = np.array(smooth_cartoon_images)
        reality_images = np.array(reality_images)

        return cartoon_images, smooth_cartoon_images,reality_images

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
    image_dir="/media/cgim/data/GAN/data/cartoonGAN_dataset2/"
    dataLoader = CartoonGAN_Loader(image_dir, 256, 256,batch_size=16,global_step=0)
    while True:
        print("----------")
        cartoon_images, smooth_cartoon_images, reality_images= dataLoader.next_batch()
        fig = plt.figure()
        ax1 = fig.add_subplot(131)
        ax1.imshow(cartoon_images[0])
        ax2 = fig.add_subplot(132)
        ax2.imshow(smooth_cartoon_images[0])
        ax3 = fig.add_subplot(133)
        ax3.imshow(reality_images[0])
        plt.show()
