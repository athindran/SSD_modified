'''
For each image do

count number of objects (n) in the image
if n == 1:
    resize to 150x150
elif n==2:
    resize to 2x75x150 to make grid
elif n==3
    make 2x2 grid and fill one with zeros
elif n==4
    make 2x2 grid and fill (net res = 300x300)
elif n==5
    make 2x3 grid and do so, shorter size is 300px
elif n==6
    make 2x3 grid and fill it.

while making each of the grids :
1. expand cropped area to (1+f)x BBOX in both dim, f is a fraction, default 0.2
2. perform resize operation on this
3. add this to grid
4. transform bounding boxes accordingly
5. save image and annotation accordingly

need to use:
XML parser (lxml or xml.Etree)
cv2 for resizing
file handling

@Author : Giridhur S.

'''

import os
import sys
import cv2
import xmltodict
import glob
import progressbar
import numpy as np

def makeMosaic(img,slices,objects,borders):
    """
    first we resize all objects to 75x75 and try to make a mosaic of these.
    mosaic will have zeros wherever
    """
    n_objects = len(objects)
    objects_out = objects.copy()
    img_temp = []
    for i in range(len(objects)):
        img_temp.append(cv2.resize(slices[i],(150,150),interpolation=cv2.INTER_AREA))
        bndbox = objects[i]["bndbox"]
        (xmin,xmax,ymin,ymax) = (int(bndbox["xmin"]),int(bndbox["xmax"]),int(bndbox["ymin"]),int(bndbox["ymax"]))
        (x_min,x_max,y_min,y_max) = (borders[i]["xmin"],borders[i]["xmax"],borders[i]["ymin"],borders[i]["ymax"]) #borders
        objects_out[i]["bndbox"] = {"xmin":int(150*(xmin-x_min)/(x_max-x_min)),"xmax":int(150*(xmin-x_min)/(x_max-x_min)),"ymin":int(150*(ymin-y_min)/(y_max-y_min)),"ymax":int(150*(ymin-y_min)/(y_max-y_min))}
    #all bounding boxes have been transformed to 150x150 so now we move on to tiling them
    if n_objects==1:
        return (img_temp[0],objects_out)
    elif n_objects==2:
        #putting them side by side
        img_out = np.concatenate((img_temp[0],img_temp[1]),axis=1)
        (xmin,xmax,ymin,ymax) = (objects_out[1]["bndbox"]["xmin"],objects_out[1]["bndbox"]["xmax"],objects_out[1]["bndbox"]["ymin"],objects_out[1]["bndbox"]["ymax"])
        objects_out[1]["bndbox"] = {"xmin":xmin,"xmax":xmax,"ymin":(ymin+150),"ymax":(ymax+150)}
        return (img_out,objects_out)
    elif n_objects==3:
        #putting first two side by side.
        img_out = np.concatenate((img_temp[0],img_temp[1]),axis =1)
        (xmin,xmax,ymin,ymax) = (objects_out[1]["bndbox"]["xmin"],objects_out[1]["bndbox"]["xmax"],objects_out[1]["bndbox"]["ymin"],objects_out[1]["bndbox"]["ymax"])
        objects_out[1]["bndbox"] = {"xmin":xmin,"xmax":xmax,"ymin":(ymin+150),"ymax":(ymax+150)}
        img__ = np.zeros(img_temp[2])
        img__ = np.concatenate((img_temp[2],img__),axis = 1)
        img_out = np.concatenate((img_out,img__),axis = 0) #on top of the other
        (xmin,xmax,ymin,ymax) = (objects_out[2]["bndbox"]["xmin"],objects_out[2]["bndbox"]["xmax"],objects_out[2]["bndbox"]["ymin"],objects_out[2]["bndbox"]["ymax"])
        objects_out[2]["bndbox"] = {"xmin":(xmin+150),"xmax":(xmax+150),"ymin":ymin,"ymax":ymax}
        return (img_out,objects_out)
    elif n_objects==4:
        img_out = np.concatenate((img_temp[0],img_temp[1]),axis =1)
        (xmin,xmax,ymin,ymax) = (objects_out[1]["bndbox"]["xmin"],objects_out[1]["bndbox"]["xmax"],objects_out[1]["bndbox"]["ymin"],objects_out[1]["bndbox"]["ymax"])
        objects_out[1]["bndbox"] = {"xmin":xmin,"xmax":xmax,"ymin":(ymin+150),"ymax":(ymax+150)}
        img__ = np.concatenate((img_temp[2],img_temp[3]),axis = 1)
        img_out = np.concatenate((img_out,img__),axis = 0) #on top of the other
        (xmin,xmax,ymin,ymax) = (objects_out[2]["bndbox"]["xmin"],objects_out[2]["bndbox"]["xmax"],objects_out[2]["bndbox"]["ymin"],objects_out[2]["bndbox"]["ymax"])
        objects_out[2]["bndbox"] = {"xmin":(xmin+150),"xmax":(xmax+150),"ymin":ymin,"ymax":ymax}
        (xmin,xmax,ymin,ymax) = (objects_out[3]["bndbox"]["xmin"],objects_out[3]["bndbox"]["xmax"],objects_out[3]["bndbox"]["ymin"],objects_out[3]["bndbox"]["ymax"])
        objects_out[3]["bndbox"] = {"xmin":(xmin+150),"xmax":(xmax+150),"ymin":(ymin+150),"ymax":(150+ymax)}
        return (img_out,objects_out)
    elif n_objects==5:













def modify_image(image_file,annotation_file,f=0.2):
    """
    Takes in an image, locates corresponding annotation and does the ^ said actions
    modified image is 100000+orig image id
    """
    with open(annotation_file,mode='r') as xml_file:
        xml_ = xml_file.read()
    xml = xmltodict.parse(xml_)
    xml_out = xml.copy()
    n_objects = len(xml["annotation"]["object"])
    objects = xml["annotation"]["object"]


    (_,filename) = os.path.split(annotation_file)
    (filename,_) = filename.split(".")
    out_filename = str(int(filename)+100000)

    xml_out["annotation"]["filename"] = out_filename = str(int(filename)+10000)+".jpg"
    slices = [];
    borders = [];
    img = cv2.imread(image_file)
    (H,W,D) = img.shape
    for i in range(n_objects):
        bndbox = objects[i]["bndbox"]
        (xmin,xmax,ymin,ymax) = (int(bndbox["xmin"]),int(bndbox["xmax"]),int(bndbox["ymin"]),int(bndbox["ymax"]))
        (xavg,yavg) = (int(0.5*(xmin+xmax)),int(0.5*(ymin+ymax)))
        (xmin,xmax) = (max(0,int((1+f)*xmin-f*xavg)),min(W,int((1+f)*xmax-f*xavg)))
        (ymin,ymax) = (max(0,int((1+f)*ymin-f*yavg)),min(W,int((1+f)*ymax-f*yavg)))
        slices.append(img[ymin:ymax,xmin:xmax,:])
        borders.append({"xmin":xmin,"xmax":xmax,"ymin":ymin,"ymax":ymax})
    (img_out,objects_out) = makeMosaic(img,slices,objects,borders)





    return 0

base_folder = "path/to/VOCdevkit/"
VOC_yr = "VOC2007/"
image_folder = "JPEGImages/"
annotation_folder = "Annotations/"
file_list = glob.glob(base_folder+VOC_yr+image_folder+"*.jpg") #list all jpg file
pbar = progressbar.ProgressBar()

for file_path in pbar(file_list):
    (_,filename) = os.path.split(file_path)
    (filename,_) = filename.split(".")
    #all files are .jpg so no worries
    image_file = file_path
    annotation_file = base_folder+VOC_yr+annotation_folder+filename+".xml"
    if os.path.isfile(annotation_file) == False:
        continue #if there's no annotation file for an image throw it away.
    modify_image(image_file)
