import os
import cv2
import numpy as np

for image_name in os.listdir("result"):
    if image_name.endswith("LR.jpg"):
        image_path = os.path.join("result",image_name)
        image_resize_path = image_path.replace("LR.jpg","4x.jpg")
        image = cv2.imread(image_path)
        image_4x = cv2.resize(image,(0,0),fx=4,fy=4)
        # cv2.imshow("image_4x",image_4x)
        # cv2.imshow("image",image)
        cv2.imwrite(image_resize_path,image_4x)
        print(image_path)