import os
import sys
if "/opt/ros/kinetic/lib/python2.7/dist-packages" in sys.path:
    sys.path.remove("/opt/ros/kinetic/lib/python2.7/dist-packages")
import cv2
import ujson
import numpy as np

target_dir = "sample/"

lightstates = ["red","yellow","green","left_arrow"]
mapping_dict = {
    "1000":"red",
    "1001":"redleft",
    "0100":"yellow",
    "0010":"green",
    "0011":"greenleft"}

visit = dict()

path = "/media/sang/20/20만장 납품_2021.02.16/light_object/image/2"

imgnames = os.listdir(path)
for imgname in imgnames:
    imgpath = path+'/'+imgname
    jsonpath=imgpath.replace("/image","/json").replace(".jpg",".json").replace(".png",".json")
    img = None
    with open(jsonpath,"r") as json:
        json_data = ujson.load(json)
    index = 0
    for d in json_data["annotation"]:
        if d["class"] == "traffic_light" and d["type"] == "car" and d["direction"]=="horizontal":
            if d['attribute'][0]["others_arrow"] == "on" or d['attribute'][0]["x_light"] == "on":
                continue
            if img is None:
                img = cv2.imread(path+"/"+imgname)
            x0,y0,x1,y1 = d["box"]
            light = ""
            for lightstate in lightstates:
                light += str(int(d['attribute'][0][lightstate] == "on"))

            if light=="0000" or light=="0001":
                continue
            light = mapping_dict[light]
            cropped_img = img[y0:y1,x0:x1]
            if cropped_img.size == 0:
                continue

            dstpath = target_dir+light+"/"+ imgname.replace(".jpg","").replace(".png","") + " " + str(index)+".png"

            if not light in visit:
                visit[light]=1
                print(dstpath)
                cv2.imwrite(dstpath,cropped_img)
            index += 1


    
