import pickle
import numpy as np
import os
import xml.etree.ElementTree as ET

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

ap = np.zeros((12,));
gobjects = {}
imagenames = []
for vindex,video in enumerate(relevantvideos):
  xml_names = os.listdir(annpath+video)
  xml_names.sort(key=lambda f: int(filter(str.isdigit, f)))
  gobjects = parse_rec(annpath+video+xml_names[0])
  if(len(gobjects)==4):
    print(vindex)
    #exit()
