
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
slim = tf.contrib.slim


# In[3]:

#get_ipython().magic(u'matplotlib inline')
#get_ipython().magic(u"config InlineBackend.figure_format = 'jpg'")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# In[4]:

import sys
sys.path.append('../')


# In[5]:

from nets import ssd_vgg_300, ssd_common, np_methods
import nets
from preprocessing import ssd_vgg_preprocessing
import visualization


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
        img_shape=net_shape,
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

#default_params = SSDParams(
#        img_shape=net_shape,
#        num_classes=21,
#        no_annotation_label=21,
#        feat_layers=['block4', 'block7', 'block8', 'block9'],
#        feat_shapes=[ (10, 10), (5, 5), (3, 3), (1, 1)],
#        anchor_size_bounds=[0.15, 0.90],
#        # anchor_size_bounds=[0.20, 0.90],
#        anchor_sizes=[(21., 45.),
#                      (45., 99.),
#                      (99., 153.),
#                      (153., 207.),
#                      (207., 261.),
#                      (261., 315.)],
        # anchor_sizes=[(30., 60.),
        #               (60., 111.),
        #               (111., 162.),
        #               (162., 213.),
        #               (213., 264.),
        #               (264., 315.)],
#        anchor_ratios=[[2, .5],
#                       [2, .5, 3, 1./3],
#                       [2, .5, 3, 1./3],
#                       [2, .5, 3, 1./3],
#                       [2, .5],
#                       [2, .5]],
#        anchor_steps=[8, 16, 32, 64, 100, 300],
#        anchor_offset=0.5,
#        normalizations=[20, -1, -1, -1, -1, -1],
#        prior_scaling=[0.1, 0.1, 0.2, 0.2]
#        )



data_format = 'NHWC'
img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
large = tf.placeholder(tf.bool)

# Evaluation pre-processing: resize to SSD net shape.
image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(img_input, None, None, (300,300), data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE,large=large)
image_4d = tf.expand_dims(image_pre, 0)

# Define the SSD model.
reuse = True if 'ssd_net' in locals() else None
#reuse =True
ssd_net = ssd_vgg_300.SSDNet(default_params)
with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
    predictions, localisations, _, _ = ssd_net.net(image_4d, is_training=False, reuse=tf.AUTO_REUSE,large=large)

# Restore SSD model.
ckpt_filename = '../checkpoints/ssd_300_vgg.ckpt'
# ckpt_filename = '../checkpoints/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt'
isess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(isess, ckpt_filename)

# SSD default anchor boxes.
ssd_anchors = ssd_net.anchors(net_shape)


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
def process_image(img, select_threshold=0.5, nms_threshold=.45, net_shape=(300, 300), islarge=True):
    # Run SSD network.
    starttime = time.time()
    
    rimg, rpredictions, rlocalisations, rbbox_img = isess.run([image_4d, predictions, localisations, bbox_img],
                                                              feed_dict={img_input: img, large:islarge})
    print("Finished session")
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


# In[11]:

# Test on some demo image and visualize output.
path = '../demo/'
image_names = sorted(os.listdir(path))

process_time = np.zeros((13,1))
network_time = np.zeros((13,1))
for lvar in range(1):
  img = mpimg.imread(path + image_names[lvar])
  start_time = time.time()
  rclasses, rscores, rbboxes, network_time[lvar] =  process_image(img,0.5,0.45,(300,300),True)

    
  #plt.imshow(img)
  #plt.show()
  end_time = time.time()
  process_time[lvar] = end_time-start_time
  #visualization.bboxes_draw_on_img(img, rclasses, rscores, rbboxes, visualization.colors_plasma)
  #visualization.plt_bboxes(img, rclasses, rscores, rbboxes)
  visualization.bboxes_draw_on_img(img, rclasses, rscores, rbboxes, visualization.colors_tableau)
  visualization.plt_bboxes(img, rclasses, rscores, rbboxes)
  #print(rbboxes)
  #rbboxes =np.array([[0,1,0,1],[0.46,0.14,0.74,0.56]])
  y1 = int(rbboxes[-1,0]*0.75*img.shape[0])
  x1 = int(rbboxes[-1,1]*0.75*img.shape[1])
  y2 = min(int(rbboxes[-1,2]*1.25*img.shape[0]),img.shape[0])
  x2 = min(int(rbboxes[-1,3]*1.25*img.shape[1]),img.shape[1])

  fovea = img[y1:y2,x1:x2,:]
  #rclasses, rscores, rbboxes, network_time[lvar] =  process_image(fovea,0.5,0.45,(75,75),False) 
  #visualization.bboxes_draw_on_img(fovea, rclasses, rscores, rbboxes, visualization.colors_tableau)
  #visualization.plt_bboxes(fovea, rclasses, rscores, rbboxes)

  #plt.imshow(fovea)
  #plt.show()
 
  
#print(process_time)
#print(np.mean(process_time[1:]))
#print(np.mean(network_time[1:]))


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:





      
