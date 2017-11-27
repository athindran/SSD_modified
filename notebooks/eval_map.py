import pickle
import numpy as np
import os
import xml.etree.ElementTree as ET

relevantids = ["n02691156","n02834778","n01503061","n02924116",
               "n02958343","n02402425","n02084071","n02121808",
               "n02374451","n03790512","n02411705","n04468005"]

annpath = '../../../../../../ILSVRC_full/Annotations/VID/'
relevantvideos = pickle.load(open('relevantvideos.p','rb'))

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

def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


fid =[[]]*12
for classes in range(12):
  fid[classes] = open('./detections_fovea/'+relevantids[classes],'r')

ap = np.zeros((12,));
gobjects = {}
imagenames = []
for video in relevantvideos[2:]:
  xml_names = os.listdir(annpath+video)
  xml_names.sort(key=lambda f: int(filter(str.isdigit, f)))
  for lvar in range(len(xml_names)):
    gobjects[video+xml_names[lvar][0:-3]+'JPEG'] = parse_rec(annpath+video+xml_names[lvar])
    imagenames.append(video+xml_names[lvar][0:-3]+'JPEG')

for classindex,classname in enumerate(relevantids):
  class_recs = {}
  npos = 0
  for imagename in imagenames:
    R = [obj for obj in gobjects[imagename] if obj['name'] == classname]
    bbox = np.array([x['bbox'] for x in R])
    #difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
    #det = [False] * len(R)
    npos = npos + len(R)
    class_recs[imagename] = {'bbox': bbox}

  lines = fid[classindex].readlines()
  
  splitlines = [x.strip().split(' ') for x in lines]
  image_ids = [x[0] for x in splitlines]
  confidence = np.array([float(x[1]) for x in splitlines])
  BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

  # sort by confidence
  sorted_ind = np.argsort(-confidence)
  sorted_scores = np.sort(-confidence)
  #print("sorted_scores",sorted_scores)
  #firstindex = np.argwhere(sorted_scores>-0.5)[0,0]
  #print("Firstindex",firstindex)
  BB = BB[sorted_ind, :]
  image_ids = [image_ids[x] for x in sorted_ind]
  # go down dets and mark TPs and FPs
  nd = len(image_ids)
  tp = np.zeros(nd)
  fp = np.zeros(nd)
  #for d in range(5):
  for d in range(nd):
    R = class_recs[image_ids[d]]
    #print("R",R)
    bb = BB[d, :].astype(float)
    ovmax = -np.inf
    BBGT = R['bbox'].astype(float)
    #print("bb",bb)
    #print("BBGT",BBGT)
    if BBGT.size > 0:
      # compute overlaps
      # intersection
      ixmin = np.maximum(BBGT[:, 0], bb[1])
      iymin = np.maximum(BBGT[:, 1], bb[0])
      ixmax = np.minimum(BBGT[:, 2], bb[3])
      iymax = np.minimum(BBGT[:, 3], bb[2])
      iw = np.maximum(ixmax - ixmin + 1., 0.)
      ih = np.maximum(iymax - iymin + 1., 0.)
      inters = iw * ih

      # union
      uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
             (BBGT[:, 2] - BBGT[:, 0] + 1.) *
             (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)
      #if(uni<1e-4):
      #  print(uni)
      #  print(d) 
      overlaps = inters / uni
      ovmax = np.max(overlaps)
      jmax = np.argmax(overlaps)

    if ovmax > 0.5:
     tp[d] = 1.
     #R['det'][jmax] = 1
    else:
      fp[d] = 1.
  print("TP",tp)
  print("FP",fp)
  # compute precision recall
  fp = np.cumsum(fp)
  tp = np.cumsum(tp)
  print("Cumsum TP",tp)
  print("Cumsum FP",fp)
  rec = tp / float(npos)
  print("Recall",rec)
  # avoid divide by zero in case the first detection matches a difficult
  # ground truth
  prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
  print("Precision",prec)
  ap[classindex] = voc_ap(rec, prec, True)
  print("Average precision:",ap[classindex])
print('mAP:',np.mean(ap))
