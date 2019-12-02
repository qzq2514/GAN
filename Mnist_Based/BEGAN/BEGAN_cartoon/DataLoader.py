import pickle
import numpy as np
import cv2
import os

class Cartoo_loader:
    def __init__(self,image_dir,batch_size,image_height,image_width,image_channal):
        self.image_dir = image_dir
        self.support_image_extensions=[".jpg",".png",".jpeg",".bmp"]
        self.image_paths = self.get_images_path()
        print(len(self.image_paths))
        self.sample_num = len(self.image_paths)
        self.batch_size = batch_size
        self.image_height = image_height
        self.image_width = image_width
        self.image_channal = image_channal

    def get_images_path(self):
        image_paths = []

        for file_name in os.listdir(self.image_dir):
            extension = os.path.splitext(file_name)[1].lower()
            if extension not in self.support_image_extensions:
                continue
            image_path = os.path.join(self.image_dir, file_name)
            image_paths.append(image_path)

        return np.array(image_paths)

    def shuffle(self):
        shuffle_ind = np.arange(0, self.sample_num)
        np.random.shuffle(shuffle_ind)
        self.train_data = self.train_data[shuffle_ind]
        self.train_label = self.train_label[shuffle_ind]

    def next_batch(self,is_random_sample, indices):
        if is_random_sample:
            # np.random.choice(最大值,N) :从[0,最大值)中选择N个数,默认放回
            # np.random.choice(数组,N，replace=false) :从数组中选择N个数,replace=False不放回
            indices = np.random.choice(len(self.image_paths), self.batch_size)
        elif indices == None:
            print("Please assign indices in the mode of random sampling!")
            return None, None
        try:
            batch_image_paths = self.image_paths[indices]
        except Exception as e:
            print("list index out of range while next_batch!")
            return None, None

        image_id = 0
        batch_images_data = []
        for image_file_path in batch_image_paths:
            channal_flag = cv2.IMREAD_GRAYSCALE if self.image_channal == 1 else cv2.IMREAD_COLOR

            image = cv2.imread(image_file_path, channal_flag)
            if image is None:
                continue
            image_id += 1
            image_resized = cv2.resize(image, (self.image_width, self.image_height))

            image_np_resized = np.resize(image_resized,
                                         (self.image_height, self.image_width, self.image_channal))
            batch_images_data.append(image_np_resized)

        batch_images_data = np.array(batch_images_data)
        return batch_images_data

if __name__ == '__main__':
    cifar=Cartoo_loader("D:/forTensorflow/cartoon",5001,96,96,3)
    while True:
        batch_data=cifar.next_batch(True,None)
        for image in batch_data:
            cv2.imshow("image",image)
            cv2.waitKey(0)