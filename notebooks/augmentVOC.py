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
    make 2x2 grid and fill
elif n==5
    make 2x3 grid and do so
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

@Authors : Giridhur S. and Athindran R.

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
    nobjects = len(objects)
    objects_out = objects.copy()
    img_temp = []

    if nobjects==1:
      nrows = 1
      rowsize = 150
      ncols = 1
      colsize = 150
    elif nobjects==2:
      nrows = 1
      rowsize = 150
      ncols = 2
      colsize = 75
    elif nobjects==3 or nobjects==4:
      nrows = 2
      rowsize = 75
      ncols = 2
      colsize = 75
    elif nobjects==5 or nobjects==6:
      nrows = 2
      rowsize = 75
      ncols = 3
      colsize = 50 

    #print(nobjects)
    for i in range(len(objects)):
        #print(slices[i].shape,rowsize,colsize)
        img_temp.append(cv2.resize(slices[i],(colsize,rowsize),interpolation=cv2.INTER_AREA))
        bndbox = objects[i]["bndbox"]
        (xmin,xmax,ymin,ymax) = (int(bndbox["xmin"]),int(bndbox["xmax"]),int(bndbox["ymin"]),int(bndbox["ymax"]))
        (x_min,x_max,y_min,y_max) = (borders[i]["xmin"],borders[i]["xmax"],borders[i]["ymin"],borders[i]["ymax"]) #borders
        objects_out[i]["bndbox"] = {"xmin":int(colsize*(xmin-x_min)/(x_max-x_min)),"xmax":int(colsize*(xmax-x_min)/(x_max-x_min)),"ymin":int(rowsize*(ymin-y_min)/(y_max-y_min)),"ymax":int(rowsize*(ymax-y_min)/(y_max-y_min))}
    #all bounding boxes have been transformed to 150x150 so now we move on to tiling them
    if nobjects==1:
        return (img_temp[0],objects_out)
    elif nobjects==2:
        #putting them side by side
        img_out = np.concatenate((img_temp[0],img_temp[1]),axis=1)
        (xmin,xmax,ymin,ymax) = (objects_out[1]["bndbox"]["xmin"],objects_out[1]["bndbox"]["xmax"],objects_out[1]["bndbox"]["ymin"],objects_out[1]["bndbox"]["ymax"])
        objects_out[1]["bndbox"] = {"xmin":xmin+colsize,"xmax":xmax+colsize,"ymin":ymin,"ymax":ymax}
        return (img_out,objects_out)
    elif nobjects==3:
        #putting first two side by side.
        img_out = np.concatenate((img_temp[0],img_temp[1]),axis =1)
        (xmin,xmax,ymin,ymax) = (objects_out[1]["bndbox"]["xmin"],objects_out[1]["bndbox"]["xmax"],objects_out[1]["bndbox"]["ymin"],objects_out[1]["bndbox"]["ymax"])
        objects_out[1]["bndbox"] = {"xmin":xmin+colsize,"xmax":xmax+colsize,"ymin":ymin,"ymax":ymax}
        img__ = np.zeros((img_temp[2].shape))
        img__ = np.concatenate((img_temp[2],img__),axis = 1)
        img_out = np.concatenate((img_out,img__),axis = 0) #on top of the other
        (xmin,xmax,ymin,ymax) = (objects_out[2]["bndbox"]["xmin"],objects_out[2]["bndbox"]["xmax"],objects_out[2]["bndbox"]["ymin"],objects_out[2]["bndbox"]["ymax"])
        objects_out[2]["bndbox"] = {"xmin":xmin,"xmax":xmax,"ymin":ymin+rowsize,"ymax":ymax+rowsize}
        return (img_out,objects_out)
    elif nobjects==4:
        img_out = np.concatenate((img_temp[0],img_temp[1]),axis =1)
        (xmin,xmax,ymin,ymax) = (objects_out[1]["bndbox"]["xmin"],objects_out[1]["bndbox"]["xmax"],objects_out[1]["bndbox"]["ymin"],objects_out[1]["bndbox"]["ymax"])
        objects_out[1]["bndbox"] = {"xmin":xmin+colsize,"xmax":xmax+colsize,"ymin":ymin,"ymax":ymax}
        img__ = np.concatenate((img_temp[2],img_temp[3]),axis = 1)
        img_out = np.concatenate((img_out,img__),axis = 0) #on top of the other
        (xmin,xmax,ymin,ymax) = (objects_out[2]["bndbox"]["xmin"],objects_out[2]["bndbox"]["xmax"],objects_out[2]["bndbox"]["ymin"],objects_out[2]["bndbox"]["ymax"])
        objects_out[2]["bndbox"] = {"xmin":xmin,"xmax":xmax,"ymin":ymin+rowsize,"ymax":ymax+rowsize}
        (xmin,xmax,ymin,ymax) = (objects_out[3]["bndbox"]["xmin"],objects_out[3]["bndbox"]["xmax"],objects_out[3]["bndbox"]["ymin"],objects_out[3]["bndbox"]["ymax"])
        objects_out[3]["bndbox"] = {"xmin":(xmin+colsize),"xmax":(xmax+colsize),"ymin":(ymin+rowsize),"ymax":(ymax+rowsize)}
        return (img_out,objects_out)
    elif nobjects==5:
        img_out = np.concatenate((img_temp[0],img_temp[1],img_temp[2]),axis =1)
        (xmin,xmax,ymin,ymax) = (objects_out[1]["bndbox"]["xmin"],objects_out[1]["bndbox"]["xmax"],objects_out[1]["bndbox"]["ymin"],objects_out[1]["bndbox"]["ymax"])
        objects_out[1]["bndbox"] = {"xmin":xmin+colsize,"xmax":xmax+colsize,"ymin":ymin,"ymax":ymax}
        (xmin,xmax,ymin,ymax) = (objects_out[2]["bndbox"]["xmin"],objects_out[2]["bndbox"]["xmax"],objects_out[2]["bndbox"]["ymin"],objects_out[2]["bndbox"]["ymax"])
        objects_out[2]["bndbox"] = {"xmin":xmin+2*colsize,"xmax":xmax+2*colsize,"ymin":ymin,"ymax":ymax}
        img__ = np.concatenate((img_temp[3],img_temp[4],np.zeros((img_temp[3].shape))),axis = 1)
        img_out = np.concatenate((img_out,img__),axis = 0) #on top of the other
        (xmin,xmax,ymin,ymax) = (objects_out[3]["bndbox"]["xmin"],objects_out[3]["bndbox"]["xmax"],objects_out[3]["bndbox"]["ymin"],objects_out[3]["bndbox"]["ymax"])
        objects_out[3]["bndbox"] = {"xmin":xmin,"xmax":xmax,"ymin":ymin+rowsize,"ymax":ymax+rowsize}
        (xmin,xmax,ymin,ymax) = (objects_out[4]["bndbox"]["xmin"],objects_out[4]["bndbox"]["xmax"],objects_out[4]["bndbox"]["ymin"],objects_out[4]["bndbox"]["ymax"])
        objects_out[4]["bndbox"] = {"xmin":xmin+colsize,"xmax":xmax+colsize,"ymin":ymin+rowsize,"ymax":ymax+rowsize}
        return (img_out,objects_out)
    elif nobjects==6:
        img_out = np.concatenate((img_temp[0],img_temp[1],img_temp[2]),axis =1)
        (xmin,xmax,ymin,ymax) = (objects_out[1]["bndbox"]["xmin"],objects_out[1]["bndbox"]["xmax"],objects_out[1]["bndbox"]["ymin"],objects_out[1]["bndbox"]["ymax"])
        objects_out[1]["bndbox"] = {"xmin":xmin+colsize,"xmax":xmax+colsize,"ymin":ymin,"ymax":ymax}
        (xmin,xmax,ymin,ymax) = (objects_out[2]["bndbox"]["xmin"],objects_out[2]["bndbox"]["xmax"],objects_out[2]["bndbox"]["ymin"],objects_out[2]["bndbox"]["ymax"])
        objects_out[2]["bndbox"] = {"xmin":xmin+2*colsize,"xmax":xmax+2*colsize,"ymin":ymin,"ymax":ymax}
        img__ = np.concatenate((img_temp[3],img_temp[4],img_temp[5]),axis = 1)
        img_out = np.concatenate((img_out,img__),axis = 0) #on top of the other
        (xmin,xmax,ymin,ymax) = (objects_out[3]["bndbox"]["xmin"],objects_out[3]["bndbox"]["xmax"],objects_out[3]["bndbox"]["ymin"],objects_out[3]["bndbox"]["ymax"])
        objects_out[3]["bndbox"] = {"xmin":xmin,"xmax":xmax,"ymin":ymin+rowsize,"ymax":ymax+rowsize}
        (xmin,xmax,ymin,ymax) = (objects_out[4]["bndbox"]["xmin"],objects_out[4]["bndbox"]["xmax"],objects_out[4]["bndbox"]["ymin"],objects_out[4]["bndbox"]["ymax"])
        objects_out[4]["bndbox"] = {"xmin":xmin+colsize,"xmax":xmax+colsize,"ymin":ymin+rowsize,"ymax":ymax+rowsize}
        (xmin,xmax,ymin,ymax) = (objects_out[5]["bndbox"]["xmin"],objects_out[5]["bndbox"]["xmax"],objects_out[5]["bndbox"]["ymin"],objects_out[5]["bndbox"]["ymax"])
        objects_out[5]["bndbox"] = {"xmin":(xmin+2*colsize),"xmax":(xmax+2*colsize),"ymin":(ymin+rowsize),"ymax":(ymax+rowsize)}
        return (img_out,objects_out) 


def modify_image(image_file,annotation_file,set_file,f=0.2):
    """
    Takes in an image, locates corresponding annotation and does the ^ said actions
    modified image is 100000+orig image id
    """
    with open(annotation_file,mode='r') as xml_file:
        xml_ = xml_file.read()
    xml = xmltodict.parse(xml_)
    xml_out = xml.copy()
    objects = xml["annotation"]["object"]
    if not isinstance(objects,list):
      objects = list([objects])
    n_objects = len(objects)

    (annoPath,filename) = os.path.split(annotation_file)
    (imgPath,filename_) = os.path.split(image_file)
    (filename,_) = filename.split(".")
    out_filename = str(int(filename)+100000)
    set_file.write(filename+"\n") 
    xml_out["annotation"]["filename"] = str(int(filename)+100000)+".jpg"
    slices = [];
    borders = [];
    img = cv2.imread(image_file)
    (H,W,D) = img.shape
    print("Number of objects: ",n_objects)
    #print("Objects: ",objects)
    if n_objects>0 and n_objects<=6:
      for i in range(n_objects):
        bndbox = objects[i]["bndbox"]
        (xmin,xmax,ymin,ymax) = (int(bndbox["xmin"]),int(bndbox["xmax"]),int(bndbox["ymin"]),int(bndbox["ymax"]))
        #print(xmin,xmax,ymin,ymax)
        (xavg,yavg) = (int(0.5*(xmin+xmax)),int(0.5*(ymin+ymax)))
        (xmin,xmax) = (max(0,int((1+f)*xmin-f*xavg)),min(W,int((1+f)*xmax-f*xavg)))
        (ymin,ymax) = (max(0,int((1+f)*ymin-f*yavg)),min(H,int((1+f)*ymax-f*yavg)))
        #print(xmin,xmax,ymin,ymax)
        slices.append(img[ymin:ymax,xmin:xmax,:])
        borders.append({"xmin":xmin,"xmax":xmax,"ymin":ymin,"ymax":ymax})
      (img_out,objects_out) = makeMosaic(img,slices,objects,borders) 
      img_out = np.concatenate((img_out,np.zeros((150,150,3))),axis=1)
      img_out = np.concatenate((img_out,np.zeros((150,300,3))),axis=0)
      out_imgpath = imgPath + "/" + out_filename + ".jpg"
      out_annopath= annoPath + "/" +  out_filename + ".xml"
      cv2.imwrite(out_imgpath, img_out)
      xml_out["annotation"]["object"] = objects_out
      xml_out["annotation"]["size"]["width"]=300
      xml_out["annotation"]["size"]["height"]=300
      str_xmlout = xmltodict.unparse(xml_out,pretty=True)
      with open(out_annopath,"w") as f:
        f.write(str_xmlout)
      set_file.write(str(int(filename)+100000)+"\n")
    #exit(1)
    return 0

base_folder = "/scratch/scratch1/athindran/augmentedVOC/VOC2007train/VOCdevkit/"
VOC_yr = "VOC2007/"
image_folder = "JPEGImages/"
imageset_folder = "ImageSets/Main/"
annotation_folder = "Annotations/"
file_list = glob.glob(base_folder+VOC_yr+image_folder+"*.jpg") #list all jpg file
#pbar = progressbar.ProgressBar()
set_file = open(base_folder+VOC_yr+imageset_folder+"trainval_augment.txt","w")
#print(file_list)
for fileno,file_path in enumerate(file_list):
    print(fileno)
    (_,filename) = os.path.split(file_path)
    (filename,_) = filename.split(".")
    #all files are .jpg so no worries
    image_file = file_path
    annotation_file = base_folder+VOC_yr+annotation_folder+filename+".xml"
    if os.path.isfile(annotation_file) == False:
        continue #if there's no annotation file for an image throw it away.
    modify_image(image_file,annotation_file,set_file)
set_file.close()
