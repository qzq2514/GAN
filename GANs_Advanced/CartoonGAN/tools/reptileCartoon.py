from bs4 import BeautifulSoup
import urllib
import json
import requests
import re
import os

root_url_format="https://image.baidu.com/search/acjson?tn=resultjson_com&ipn=rj&ct=201326592&is=&fp=result&queryWord=%E9%A3%8E%E6%99%AF+%E6%97%85%E8%A1%8C&cl=2&lm=-1&ie=utf-8&oe=utf-8&adpicid=&st=-1&z=&ic=&hd=&latest=&copyright=&word=%E9%A3%8E%E6%99%AF+%E6%97%85%E8%A1%8C&s=&se=&tab=&width=&height=&face=0&istype=2&qc=&nc=1&fr=&expermode=&force=&pn={}&rn=30&gsm=&{}="
save_dir = "pictures/reality1"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
def main():

    for id in range(30,5000,30):
        root_url = root_url_format.format(id,1578554012393+id)
        print(root_url)
        response = requests.get(root_url)
        response.encoding = "GB18030"
        try:
            info = json.loads(response.text)
            for data in info["data"]:
                if "middleURL" not in data.keys():
                    continue
                picNum = len(os.listdir(save_dir))
                save_path = "{}/reality1_{}.jpg".format(save_dir,picNum+1)
                urllib.request.urlretrieve(data["middleURL"],save_path)
                print(data["middleURL"])
        except:
            print("continue")
            continue


if __name__ == '__main__':
    main()







