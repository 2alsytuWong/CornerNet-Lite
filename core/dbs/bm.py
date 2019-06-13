from .detection import DETECTION
import numpy as np
import os
import json

class BM(DETECTION):
'''
class used to Blade Master detection
'''
def init(self,db_config,split=None, sys_config=None):
super().init(db_config)
self._mean = np.array([0.40789654, 0.44719302, 0.47026115], dtype=np.float32)
self._std = np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32)
self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571], dtype=np.float32)
self._eig_vec = np.array([[-0.58752847, -0.69563484, 0.41340352],
[-0.5832747, 0.00994535, -0.81221408],
[-0.56089297, 0.71832671, 0.41158938]], dtype=np.float32)
self._configs["categories"] = 1
if split is not None:
data_dir = os.path.join(sys_config.data_dir,split,'zhenjiang')
self._image_dir = os.path.join(data_dir, "images")
self.m_annotations = os.path.join(data_dir,"annotations")
self.load_detection()

def load_detection(self):
    self._detections = {}
    json_list = [x for x in os.listdir(self.m_annotations) if(os.path.splitext(x)[1] == '.json')]
    print("total json number is {}".format(len(json_list)))
    for each_json in json_list:
        each_json_file = os.path.join(self.m_annotations,each_json)
        with open(each_json_file,'r') as f:
            json_label = json.load(f)
            image_file = os.path.basename(json_label['image']['information']['path'])
            annotation = json_label['annotation']['persons']
            if len(annotation) == 0:
                continue
            numpy_anno = np.zeros((len(annotation),5),dtype=np.float32)
            for i, anno in enumerate(annotation):
                numpy_anno[i,0] = anno['box'][0]*320
                numpy_anno[i,1] = anno['box'][1]*288
                numpy_anno[i,2] = anno['box'][2]*320
                numpy_anno[i,3] = anno['box'][3]*288
                numpy_anno[i,4] = 1
            self._detections[image_file]=numpy_anno
    print("contain object json is {}".format(len(self._detections)))
    self._db_inds = np.arange(len(self._detections))
    self._image_ids = list(self._detections.keys())

def image_path(self, ind):
    file_name = self._image_ids[ind]
    return os.path.join(self._image_dir, file_name)

def detections(self, ind):
    file_name = self._image_ids[ind]
    return self._detections[file_name].copy()
def cls2name(self, cls):
    return "person"`
modify the init.py
from .coco import COCO from .bus import BUS datasets = { "COCO": COCO, "BUS":BUS }
modify the *config.json
` "system": {
"dataset": "BUS",
"batch_size": 10,
"sampling_function": "cornernet",

    "train_split": "train",
    "val_split": "val",

    "learning_rate": 0.00025,
    "decay_rate": 10,

    "val_iter": 100,

    "opt_algo": "adam",
    "prefetch_size": 5,

    "max_iter": 500000,
    "stepsize": 450000,
    "snapshot": 5000,

    "chunk_sizes": [10],

    "data_dir": "/opt/jl/vmware_share/object_detection"
},