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
        self._mean    = np.array([0.40789654, 0.44719302, 0.47026115], dtype=np.float32)
        self._std     = np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571], dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]], dtype=np.float32)
        self._configs["categories"] = 12 # 暴恐数据12个类别
        self._bm_cls_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        self._bm_cls_names = [
            'tibetan_flag',
            'gongzhang_logo',
            'BK_LOGO',
            'taiwan_bairiqi_flag',
            'isis_flag',
            'not_terror_card_text',
            'not_terror',
            'china_guoqi_flag',
            'knives',
            'guns',
            'card',
            'islamic_flag']
        if split is not None:
            data_dir = os.path.join(sys_config.data_dir,split,'bm2019')
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
                annotation = json_label['annotations']
                if len(annotation) == 0:
                    continue
                numpy_anno = np.zeros((len(annotation),5),dtype=np.float32)
                for i, anno in enumerate(annotation):
                    numpy_anno[i,0] = anno['box'][0]
                    numpy_anno[i,1] = anno['box'][1]
                    numpy_anno[i,2] = anno['box'][2]
                    numpy_anno[i,3] = anno['box'][3]
                    numpy_anno[i,4] = anno['category']
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
        return self._bm_cls_names[cls-1]