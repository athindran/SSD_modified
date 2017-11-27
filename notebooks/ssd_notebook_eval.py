
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
            select_threshold=select_threshold, img_shape=net_shape, num_classes=201, decode=True)
    
    rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
    rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
    rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
    # Resize bboxes to original image shape. Note: useless for Resize.WARP!
    rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
    return rclasses, rscores, rbboxes,networktime


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
  fid[classes] = open('./detections/'+relevantids[classes],'w+')
start_time = time.time()
totimages = 0;
total_process_time = 0;
for video in relevantvideos[2:10]:
  image_names = os.listdir(path+video)
  image_names.sort(key=lambda f: int(filter(str.isdigit, f)))
  #print(image_names)
  #process_time = np.zeros((len(image_names),1))
  #network_time = np.zeros((len(image_names),1))
  #process_medium_time = np.zeros((len(image_names),1))
  #process_fovea_time = np.zeros((len(image_names),1))
  for lvar in range(len(image_names)):
    img = mpimg.imread(path + video + image_names[lvar])
    process_start_time = time.time()
    rclasses, rscores, rbboxes,_ =  process_image(img,0.5,0.45,(300,300),True,False)
    process_end_time = time.time()
    total_process_time = total_process_time+process_end_time-process_start_time
    totimages = totimages+1 
    #gobects = parse_rec(annpath+video+image_names[lvar][0:-5]+'.xml')
    #for index,rclass in enumerate(rclasses):
    #  fileindex = np.argwhere(pascalclasses==rclass)
    #  if(fileindex.size>0):
         #print(fileindex[0,0])
    #     fid[fileindex[0,0]].write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
    #                            format(video+image_names[lvar], rscores[index],
    #                                   rbboxes[index, 0]*img.shape[0] + 1, rbboxes[index, 1]*img.shape[1] + 1,
    #                                   rbboxes[index, 2]*img.shape[0] + 1, rbboxes[index, 3]*img.shape[1] + 1))


end_time = time.time()
print("Total time taken: ", end_time-start_time)
print("Average Process Time: ", total_process_time/totimages)
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





      
