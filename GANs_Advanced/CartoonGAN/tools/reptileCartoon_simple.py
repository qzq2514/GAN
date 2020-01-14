from bs4 import BeautifulSoup
import urllib
import json
import requests
import re
import os

root_url_format="http://www.win4000.com/wallpaper_detail_{}.html"
save_dir = "pictures/cartoon_wallpaper"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
def main():
    start_id=78078
    num=1
    for id in range(start_id,start_id+num):
        if id==start_id:
            pad = start_id
        else:
            pad = "{}_{}".format(start_id,id-start_id+1)
        root_url = root_url_format.format(pad)
        print(root_url)
        response = requests.get(root_url)
        response.encoding = "GB18030"
        soup = BeautifulSoup(response.text, 'html.parser')
        image_name = soup.select(".pic-large")[0]["src"]

        picNum = len(os.listdir(save_dir))
        save_path = "{}/wallpaper_{}.jpg".format(save_dir, picNum + 1)
        print(image_name)
        urllib.request.urlretrieve(image_name, save_path)



if __name__ == '__main__':
    main()







