
# coding: utf-8

# In[2]:

import os
import math
import random

import numpy as np
import tensorflow as tf
import cv2
import time
from collections import namedtuple
import pickle
import xml.etree.ElementTree as ET

slim = tf.contrib.slim


# In[3]:

#get_ipython().magic(u'matplotlib inline')
#get_ipython().magic(u"config InlineBackend.figure_format = 'jpg'")
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.image as mpimg


# In[4]:

import sys
sys.path.append('../')


# In[5]:

from nets import ssd_vgg_300, ssd_common, np_methods
import nets

from preprocessing import ssd_vgg_preprocessing
import visualization

relevantids = ["n02691156","n02834778","n01503061","n02924116",
               "n02958343","n02402425","n02084071","n02121808",
               "n02374451","n03790512","n02411705","n04468005"]

pascalclasses = [1,2,3,6,7,10,12,8,13,14,17,19]

# In[6]:

# TensorFlow session: grow memory when needed. TF, DO NOT USE ALL MY GPU MEMORY!!!
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
isess = tf.InteractiveSession(config=config)


# ## SSD 300 Model
# 
# The SSD 300 network takes 300x300 image inputs. In order to feed any image, the latter is resize to this input shape (i.e.`Resize.WARP_RESIZE`). Note that even though it may change the ratio width / height, the SSD model performs well on resized images (and it is the default behaviour in the original Caffe implementation).
# 
# SSD anchors correspond to the default bounding boxes encoded in the network. The SSD net output provides offset on the coordinates and dimensions of these anchors.

# In[7]:

SSDParams = namedtuple('SSDParameters', ['img_shape',
                                         'num_classes',
                                         'no_annotation_label',
                                         'feat_layers',
                                         'feat_shapes',
                                         'anchor_size_bounds',
                                         'anchor_sizes',
                                         'anchor_ratios',
                                         'anchor_steps',
                                         'anchor_offset',
                                         'normalizations',
                                         'prior_scaling'
                                         ])


# Input placeholder.
net_shape = (300,300)
default_params = SSDParams(
        img_shape=(300,300),
        num_classes=21,
        no_annotation_label=21,
        feat_layers=['block4', 'block7', 'block8', 'block9', 'block10', 'block11'],
        feat_shapes=[(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)],
        anchor_size_bounds=[0.15, 0.90],
       # anchor_size_bounds=[0.20, 0.90],
        anchor_sizes=[(21., 45.),
                      (45., 99.),
                      (99., 153.),
                      (153., 207.),
                      (207., 261.),
                      (261., 315.)],
        # anchor_sizes=[(30., 60.),
        #               (60., 111.),
        #               (111., 162.),
        #               (162., 213.),
        #               (213., 264.),
        #               (264., 315.)],
        anchor_ratios=[[2, .5],
                       [2, .5, 3, 1./3],
                       [2, .5, 3, 1./3],
                       [2, .5, 3, 1./3],
                       [2, .5],
                       [2, .5]],
        anchor_steps=[8, 16, 32, 64, 100, 300],
        anchor_offset=0.5,
        normalizations=[20, -1, -1, -1, -1, -1],
        prior_scaling=[0.1, 0.1, 0.2, 0.2]
        )

medium_params = SSDParams(
        img_shape=(150,150),
        num_classes=21,
        no_annotation_label=21,
        feat_layers=['block4', 'block7', 'block8', 'block9', 'block10'],
        feat_shapes=[(19, 19), (10, 10), (5, 5), (3, 3), (1, 1)],
        anchor_size_bounds=[0.15, 0.90],
       # anchor_size_bounds=[0.20, 0.90],
        anchor_sizes=[(21., 45.),
                      (45., 99.),
                      (99., 153.),
                      (153., 207.),
                      (207., 261.)],
        # anchor_sizes=[(30., 60.),
        #               (60., 111.),
        #               (111., 162.),
        #               (162., 213.),
        #               (213., 264.),
        #               (264., 315.)],
        anchor_ratios=[[2, .5],
                       [2, .5, 3, 1./3],
                       [2, .5, 3, 1./3],
                       [2, .5, 3, 1./3],
                       [2, .5],
                       [2, .5]],
        anchor_steps=[8, 16, 32, 64, 100],
        anchor_offset=0.5,
        normalizations=[20, -1, -1, -1, -1, -1],
        prior_scaling=[0.1, 0.1, 0.2, 0.2]
        )



fovea_params = SSDParams(
        img_shape=(75,75),
        num_classes=21,
        no_annotation_label=21,
        feat_layers=['block4', 'block7', 'block8', 'block9'],
        feat_shapes=[ (10, 10), (5, 5), (3, 3), (1, 1)],
        anchor_size_bounds=[0.15, 0.90],
        # anchor_size_bounds=[0.20, 0.90],
        anchor_sizes=[(21., 45.),
                      (45., 99.),
                      (99., 153.),
                      (153., 207.)],
        # anchor_sizes=[(30., 60.),
        #               (60., 111.),
        #               (111., 162.),
        #               (162., 213.),
        #               (213., 264.),
        #               (264., 315.)],
        anchor_ratios=[[2, .5],
                       [2, .5, 3, 1./3],
                       [2, .5, 3, 1./3],
                       [2, .5, 3, 1./3],
                       [2, .5],
                       [2, .5]],
        anchor_steps=[8, 16, 32, 64],
        anchor_offset=0.5,
        normalizations=[20, -1, -1, -1, -1, -1],
        prior_scaling=[0.1, 0.1, 0.2, 0.2]
        )


def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        #obj_struct['pose'] = obj.find('pose').text
        #obj_struct['truncated'] = int(obj.find('truncated').text)
        #obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        objects.append(obj_struct)

    return objects


data_format = 'NHWC'
img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
large = tf.placeholder(tf.bool)
medium = tf.placeholder(tf.bool)

# Evaluation pre-processing: resize to SSD net shape.
image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(img_input, None, None, (300,300), data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE,large=large,medium=medium)
image_4d = tf.expand_dims(image_pre, 0)

# Define the SSD model.
reuse = True if 'ssd_net' in locals() else None
#reuse =True
ssd_net = ssd_vgg_300.SSDNet(default_params)
with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
    predictions, localisations, _, _ = ssd_net.net(image_4d, is_training=False, reuse=tf.AUTO_REUSE,large=large,medium=medium)

# Restore SSD model.
#ckpt_filename = '../checkpoints/VGG_ILSVRC2016_SSD_300x300_iter_440000.ckpt'
ckpt_filename = '../checkpoints/ssd_300_vgg.ckpt'
isess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(isess, ckpt_filename)

# SSD default anchor boxes.
ssd_anchors1 = ssd_net.anchors((300,300))
ssd_anchors2 = ssd_vgg_300.ssd_anchors_all_layers((75,75),fovea_params.feat_shapes,fovea_params.anchor_sizes,fovea_params.anchor_ratios,fovea_params.anchor_steps,offset=0.5,dtype=np.float32)
ssd_anchors3 = ssd_vgg_300.ssd_anchors_all_layers((150,150),medium_params.feat_shapes,medium_params.anchor_sizes,medium_params.anchor_ratios,medium_params.anchor_steps,offset=0.5,dtype=np.float32)


# ## Post-processing pipeline
# 
# The SSD outputs need to be post-processed to provide proper detections. Namely, we follow these common steps:
# 
# * Select boxes above a classification threshold;
# * Clip boxes to the image shape;
# * Apply the Non-Maximum-Selection algorithm: fuse together boxes whose Jaccard score > threshold;
# * If necessary, resize bounding boxes to original image shape.

# In[8]:

# Main image processing routine.
def process_image(img, select_threshold=0.5, nms_threshold=.45, net_shape=(300, 300), islarge=True, ismedium= True):
    # Run SSD network.
    starttime = time.time()
    
    rimg, rpredictions, rlocalisations, rbbox_img = isess.run([image_4d, predictions, localisations, bbox_img],
                                                              feed_dict={img_input: img, large:islarge, medium:ismedium})
    if not islarge and not ismedium:
      rpredictions = rpredictions[0:4]
      rlocalisations = rlocalisations[0:4]
      ssd_anchors = ssd_anchors2
    elif ismedium:
      rpredictions = rpredictions[0:5]
      rlocalisations = rlocalisations[0:5]
      ssd_anchors = ssd_anchors3
    else:
      ssd_anchors = ssd_anchors1
    networktime = time.time()-starttime
    # Get classes and bboxes from the net outputs.
    rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
            rpredictions, rlocalisations, ssd_anchors,
            select_threshold=select_threshold, img_shape=net_shape, num_classes=21, decode=True)
    
    rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
    rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
    rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
    # Resize bboxes to original image shape. Note: useless for Resize.WARP!
    rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
    return rclasses, rscores, rbboxes,networktime

def process_fovea_lowres(img,foveaimg,previousfoveapositions,ndetectionsinvideo,select_threshold=0.5, nms_threshold=.45, net_shape=(300, 300), islarge=True, ismedium= True):
  #rimg, rpredictions, rlocalisations, rbbox_img = isess.run([image_4d, predictions, localisations, bbox_img],
  #                                                            feed_dict={img_input: img, large:islarge, medium:ismedium})
  fimg, fpredictions, flocalisations, fbbox_img = isess.run([image_4d, predictions, localisations, bbox_img],
                                                              feed_dict={img_input: foveaimg, large:islarge, medium:ismedium})
  if not islarge and not ismedium:
    fpredictions = fpredictions[0:4]
    flocalisations = flocalisations[0:4]
    ssd_anchors = ssd_anchors2
  elif ismedium:
    fpredictions = fpredictions[0:5]
    flocalisations = flocalisations[0:5]
    ssd_anchors = ssd_anchors3
  else:
    ssd_anchors = ssd_anchors1
   # Get classes and bboxes from the net outputs.
  fclasses, fscores, fbboxes = np_methods.ssd_bboxes_select(
            fpredictions, flocalisations, ssd_anchors,
            select_threshold=select_threshold, img_shape=net_shape, num_classes=21, decode=True)
    
  fbboxes = np_methods.bboxes_clip(fbbox_img, fbboxes)
  fclasses, fscores, fbboxes = np_methods.bboxes_sort(fclasses, fscores, fbboxes, top_k=400)
  fclasses, fscores, fbboxes = np_methods.bboxes_nms(fclasses, fscores, fbboxes, nms_threshold=nms_threshold)
  mfbboxes = np.array(fbboxes)
  if ndetectionsinvideo==1:
     nrows = 1
     ncolumns = 1
  elif ndetectionsinvideo==2:
     nrows = 1
     ncolumns = 2
  elif ndetectionsinvideo==3 or ndetectionsinvideo==4:
     nrows = 2
     ncolumns = 2
  elif ndetectionsinvideo == 5 or ndetectionsinvideo==6:
     nrows = 2
     ncolumns = 3
  
  rowindices = np.arange(0,1.01,1.0/nrows)
  columnindices = np.arange(0,1.01,1.0/ncolumns)
  finalclasses = np.array([],dtype=np.int64)
  finalscores = np.array([])
  finalbboxes = np.array([])
  #print('MFclasses',fclasses)
  #print('MFscores',fscores)
  #print('MFbboxes',fbboxes)
  #print('Rowindices',rowindices)
  #print('Columnindices',columnindices) 
  frameid = 0
  #print(ndetectionsinvideo)
  #print(fbboxes)
  #print(previousfoveapositions)
  for row in range(nrows):
    for column in range(ncolumns):
      if(frameid<ndetectionsinvideo): 
        frameindices = np.argwhere(np.logical_and(np.logical_and(np.logical_and(fbboxes[:,0]>=rowindices[row],fbboxes[:,2]<=rowindices[row+1]),fbboxes[:,1]>=columnindices[column]),fbboxes[:,3]<=columnindices[column+1])).ravel()
        #print(columnindices[column])
        #print(fbboxes[:,1])
        #print("Condition",fbboxes[:,1]>=columnindices[column])
        #print("Row",row)
        #print("Column",column)
        #print("Frameindices",frameindices)
        #print(maxscore)
        #print(validbboxes)
        if(frameindices.size>0):
          #maxscore = np.argmax(fscores[frameindices]) 
          #validbboxes = np.array(fbboxes[frameindices[maxscore],:])
          validbboxes = np.array(fbboxes[frameindices,:])
          validbboxes[:,0] = previousfoveapositions[frameid,0]+(fbboxes[frameindices,0]-rowindices[row])*(previousfoveapositions[frameid,2]-previousfoveapositions[frameid,0])*nrows
          validbboxes[:,1] = previousfoveapositions[frameid,1]+(fbboxes[frameindices,1]-columnindices[column])*(previousfoveapositions[frameid,3]-previousfoveapositions[frameid,1])*ncolumns
          validbboxes[:,2] = previousfoveapositions[frameid,0]+(fbboxes[frameindices,2]-rowindices[row])*(previousfoveapositions[frameid,2]-previousfoveapositions[frameid,0])*nrows
          validbboxes[:,3] = previousfoveapositions[frameid,1]+(fbboxes[frameindices,3]-columnindices[column])*(previousfoveapositions[frameid,3]-previousfoveapositions[frameid,1])*ncolumns
          finalclasses = np.concatenate((finalclasses,fclasses[frameindices]),axis=0)
          finalscores = np.concatenate((finalscores,fscores[frameindices]),axis=0) 
          if(finalbboxes.size==0):
            #finalbboxes = np.array([validbboxes])
            finalbboxes = validbboxes
          else:
            #finalbboxes = np.vstack((finalbboxes,validbboxes)) 
            finalbboxes = np.append(finalbboxes,validbboxes, axis=0)
      frameid=frameid+1 

  #print("finalbboxes",finalbboxes)
  #print("finalclasses",finalclasses)
  #print("finalscores",finalscores)
  finalclasses, finalscores, finalbboxes = np_methods.bboxes_sort(finalclasses, finalscores, finalbboxes, top_k=400)
  finalclasses, finalscores, finalbboxes = np_methods.bboxes_nms(finalclasses, finalscores, finalbboxes, nms_threshold=nms_threshold)
  #print("finalbboxes",finalbboxes)
  #print("finalclasses",finalclasses)
  #print("finalscores",finalscores)

  #print('classes',finalclasses)
  #print('scores',finalscores)
  #print('bboxes',finalbboxes)
 
  return finalclasses,finalscores,finalbboxes,0,fclasses,fscores,mfbboxes

def formMultiObjectFovea(img,rbboxes,ndetectionsinvideo,foveasize):
  if ndetectionsinvideo==1:
     rowsize = foveasize
     columnsize = foveasize;
  elif ndetectionsinvideo==2:
     rowsize = foveasize
     columnsize = foveasize/2
  elif ndetectionsinvideo==3 or ndetectionsinvideo==4:
     rowsize = foveasize/2
     columnsize = foveasize/2
  elif ndetectionsinvideo == 5 or ndetectionsinvideo==6:
     rowsize = foveasize/2
     columnsize = foveasize/3

  foveaimg = np.zeros((foveasize,foveasize,3),dtype=np.uint8)
  rowindex = 0
  columnindex = 0
  previousfoveapositions = np.zeros((ndetectionsinvideo,4))
  for findex in range(ndetectionsinvideo):
     #print(columnsize)
     xmin = int(0.9*rbboxes[findex,0]*img.shape[0])
     xmax = min(int(1.1*rbboxes[findex,2]*img.shape[0]),img.shape[0])
     ymin = int(0.9*rbboxes[findex,1]*img.shape[1])
     ymax = min(int(1.1*rbboxes[findex,3]*img.shape[1]),img.shape[1])
     foveaimg[rowsize*rowindex:(rowindex+1)*rowsize,columnsize*columnindex:(columnindex+1)*columnsize,:]=cv2.resize(img[xmin:xmax,ymin:ymax,:], (columnsize, rowsize))
     if(columnindex+1==int(foveasize/columnsize)):
       columnindex = 0
       rowindex = rowindex+1
     else:
       columnindex = columnindex+1  
     previousfoveapositions[findex,0] = float(xmin)/img.shape[0]
     previousfoveapositions[findex,1] = float(ymin)/img.shape[1]
     previousfoveapositions[findex,2] = float(xmax)/img.shape[0]
     previousfoveapositions[findex,3] = float(ymax)/img.shape[1]
  return foveaimg,previousfoveapositions

# In[11]:

# Test on some demo image and visualize output.
path = '../../../../../../ILSVRC_full/Data/VID/'
annpath = '../../../../../../ILSVRC_full/Annotations/VID/'
relevantvideos = pickle.load(open('relevantvideos.p','rb'))
print(len(relevantvideos))
#exit()

outputpath = '../outputimages/'
#path = '../demo/'
fid =[[]]*12
for classes in range(12):
  fid[classes] = open('./detections_fovea/'+relevantids[classes],'w+')
start_time = time.time()
totimages = 0
total_process_time = 0
for videoindex,video in enumerate(relevantvideos[2:]):
  print(videoindex)
  image_names = os.listdir(path+video)
  image_names.sort(key=lambda f: int(filter(str.isdigit, f)))
  #print(image_names)
  #process_time = np.zeros((len(image_names),1))
  #network_time = np.zeros((len(image_names),1))
  #process_medium_time = np.zeros((len(image_names),1))
  #process_fovea_time = np.zeros((len(image_names),1))
  ndetectionsinvideo = 0
  for lvar in range(len(image_names)):
  #for lvar in range(5):
    img = mpimg.imread(path + video + image_names[lvar])
    totimages = totimages+1
    #print(img.shape)
    #start_time = time.time(
    #process_start_time = time.time()
    if(lvar%10!=0 and ndetectionsinvideo<=6 and areaunderdetections<=0.7):
      if ndetectionsinvideo!=0:
        foveaimg,previousfoveapositions = formMultiObjectFovea(img,rbboxes,ndetectionsinvideo,150)
        allclasses,allscores,allbboxes,_,mfclasses,mfscores,mfbboxes = process_fovea_lowres(img,foveaimg,previousfoveapositions,ndetectionsinvideo,0.01,0.45,(150,150),False,True)
        #visualization.bboxes_draw_on_img(img, rclasses, rscores, rbboxes, visualization.colors_tableau)
        #visualization.plt_bboxes(img, rclasses, rscores, rbboxes, outputpath+"fovea_"+image_names[lvar]) 
        #visualization.bboxes_draw_on_img(foveaimg, mfclasses, mfscores, mfbboxes, visualization.colors_tableau)
        #visualization.plt_bboxes(foveaimg, mfclasses, mfscores, mfbboxes, outputpath+"mfovea_"+image_names[lvar])    
        scoreinds = np.argwhere(allscores>0.5).ravel()        
        rscores = allscores[scoreinds]
        rclasses = allclasses[scoreinds]
        rbboxes = allbboxes[scoreinds,:]
        if(rbboxes.size==0):
          ndetectionsinvideo = 0
          areaunderdetections = 0
        else:
          ndetectionsinvideo = rbboxes.shape[0]
          areaunderdetections = np.sum((rbboxes[:,2]-rbboxes[:,0])*(rbboxes[:,3]-rbboxes[:,1]))
    else:
      allclasses, allscores, allbboxes,_ =  process_image(img,0.01,0.45,(300,300),True,False)
      #visualization.bboxes_draw_on_img(img, rclasses, rscores, rbboxes, visualization.colors_tableau)
      #visualization.plt_bboxes(img, rclasses, rscores, rbboxes, outputpath+"fovea_"+image_names[lvar]) 

      scoreinds = np.argwhere(allscores>0.5).ravel()        
      rscores = allscores[scoreinds]
      rclasses = allclasses[scoreinds]
      rbboxes = allbboxes[scoreinds,:]

      if(rbboxes.size==0):
        ndetectionsinvideo = 0
        areaunderdetections = 0
      else:
        ndetectionsinvideo = rbboxes.shape[0]
        areaunderdetections = np.sum((rbboxes[:,2]-rbboxes[:,0])*(rbboxes[:,3]-rbboxes[:,1]))
    #print(ndetectionsinvideo)
    #process_end_time = time.time()
    #total_process_time = total_process_time+process_end_time-process_start_time
   
    #gobjects = parse_rec(annpath+video+image_names[lvar][0:-5]+'.xml')
    for index,allclass in enumerate(allclasses):
      fileindex = np.argwhere(pascalclasses==allclass)
      if(fileindex.size>0):
         #print(fileindex[0,0])
         fid[fileindex[0,0]].write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(video+image_names[lvar], allscores[index],
                                       allbboxes[index, 0]*img.shape[0] + 1, allbboxes[index, 1]*img.shape[1] + 1,
                                       allbboxes[index, 2]*img.shape[0] + 1, allbboxes[index, 3]*img.shape[1] + 1))
end_time = time.time()

#print("Average time taken: ", total_process_time/totimages)
print("Total time taken: ", end_time-start_time)
    #print(rbboxes) 
    #plt.imshow(img)
    #plt.show()
    #end_time = time.time()
    #process_time[lvar] = end_time-start_time
    #visualization.bboxes_draw_on_img(img, rclasses, rscores, rbboxes, visualization.colors_plasma)
    #visualization.plt_bboxes(img, rclasses, rscores, rbboxes)
    #rimg = np.array(img) 
    #visualization.bboxes_draw_on_img(rimg, rclasses, rscores, rbboxes, visualization.colors_tableau)
    #visualization.plt_bboxes(rimg, rclasses, rscores, rbboxes, outputpath+"complete_"+image_names[lvar])
    #print(rbboxes)
    #print(rclasses)
    #print(rscores)
    #rbboxes =np.array([[0,1,0,1],[0.46,0.14,0.74,0.56]])
    #if rbboxes.shape[0]>=1:
      #start_time = time.time()
      #y1 = int(rbboxes[-1,0]*0.8*img.shape[0])
      #x1 = int(rbboxes[-1,1]*0.8*img.shape[1])
      #y2 = min(int(rbboxes[-1,2]*1.2*img.shape[0]),img.shape[0])
      #x2 = min(int(rbboxes[-1,3]*1.2*img.shape[1]),img.shape[1])
      #print(y2-y1,x2-x1)
      #medium_img = np.array(img[y1:y2,x1:x2,:])
      #medium_img = np.array(img)
      #plt.imshow(medium_img)
      #plt.show() 
      #r2classes, r2scores, r2bboxes, network_time[lvar] =  process_image(medium_img,0.4,0.45,(150,150),False,True) 
      #end_time = time.time()
      #process_medium_time[lvar] = end_time-start_time
      #plt.figure()
      #print(rbboxes)
      #print(r2classes)
      #print(r2scores)
      #visualization.bboxes_draw_on_img(medium_img, r2classes, r2scores, r2bboxes, visualization.colors_tableau)
      #visualization.plt_bboxes(medium_img, r2classes, r2scores, r2bboxes,outputpath+"fovea150_"+image_names[lvar])
 
      #start_time = time.time()
      #y1 = int(rbboxes[-1,0]*0.8*img.shape[0])
      #x1 = int(rbboxes[-1,1]*0.8*img.shape[1])
      #y2 = min(int(rbboxes[-1,2]*1.2*img.shape[0]),img.shape[0])
      #x2 = min(int(rbboxes[-1,3]*1.2*img.shape[1]),img.shape[1])

      #fovea = np.array(img[y1:y2,x1:x2,:])
      #r2classes, r2scores, r2bboxes, network_time[lvar] =  process_image(fovea,0.4,0.45,(75,75),False,False) 
      #end_time = time.time()
      #process_fovea_time[lvar] = end_time-start_time
      #visualization.bboxes_draw_on_img(fovea, r2classes, r2scores, r2bboxes, visualization.colors_tableau)
      #visualization.plt_bboxes(fovea, r2classes, r2scores, r2bboxes,outputpath+"fovea75_"+image_names[lvar])
    #plt.imshow(fovea)
    #plt.show()
 
  
  #print(process_time)
  #print(np.mean(process_time[1:]))
  #print(process_medium_time)
  #print(np.mean(process_medium_time[1:]))
  #print(process_fovea_time)
  #print(np.mean(process_fovea_time[1:]))
  #break;
 

#print(np.mean(network_time[1:]))


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:





      
