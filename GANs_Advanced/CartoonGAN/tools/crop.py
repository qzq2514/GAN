from scipy import misc
import matplotlib.pyplot as plt
import random
import cv2
import os

images_dir = "P:/WorkSpace/PyCharm/GANs_Advanced/CartoonGAN/pictures/AllReality"
min_size = 256
def main():
    file_names = os.listdir(images_dir)
    save_dir = images_dir+"_crop"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for file_name in file_names:
        file_path = os.path.join(images_dir,file_name)
        print(file_path)
        try:
            image = cv2.imread(file_path)
            h,w = image.shape[:2]
        except:
            continue

        file_name_no = os.path.splitext(file_name)[0]
        center_size = min(h,w)
        min_size_half = min_size // 2
        center_x = w//2
        center_y = h//2
        start_x = center_x - center_size//2
        end_x = center_x + center_size//2
        start_y = center_y - center_size//2
        end_y = center_y + center_size//2

        center_image = image[start_y:end_y,start_x:end_x,...]

        save_path = os.path.join(save_dir,"{}_center.jpg".format(file_name_no))
        cv2.imwrite(save_path,center_image)

        # cv2.imshow("org_image",image)
        # cv2.imshow("center_image", center_image)

        # print(h,w)
        for x_grid in range((w-min_size)//min_size_half):
            for y_grid in range((h-min_size) // (min_size // 2)):
                # print("range:",min_size_half*x_grid,(x_grid+1)*min_size_half)
                random_center_x = random.randint(min_size_half*x_grid,(x_grid+1)*min_size_half)+min_size_half
                random_center_y = random.randint(min_size_half*y_grid,(y_grid+1)*min_size_half)+min_size_half

                start_x = random_center_x - min_size_half
                end_x = random_center_x + min_size_half
                start_y = random_center_y - min_size_half
                end_y = random_center_y + min_size_half
                random_crop_image = image[start_y:end_y,start_x:end_x,...]
                # cv2.imshow("{}_{}".format(random_center_x,random_center_y), random_crop_image)
                # print("grid:",random_center_x,random_center_y)
                print("({},{})--->({},{})".format(start_y,start_x,end_y,end_x))
                save_name = file_name_no+"{}_{}_{}_{}.jpg".format(start_y,start_x,end_y,end_x)
                cv2.imwrite(os.path.join(save_dir,save_name), random_crop_image)
        # cv2.waitKey()

        # fig = plt.figure()
        # axis1 = fig.add_subplot(121)
        # axis1.imshow(image)
        # axis2 = fig.add_subplot(122)
        # axis2.imshow(center_image)
        # plt.show()





if __name__ == '__main__':
    main()


