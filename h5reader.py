"""
This is the file that transform hdf5 file into a simpler form that can be access easier.

"""
import numpy as np
import h5py
import cv2
import random
import torch
from torch.utils.data import Dataset
import json
MAX_LENGTH = 50
class RoboTwinH5:
    def __init__(self,path:str):
        self.path = path
        self.file = h5py.File(path,"r")
        self.action = self.file["action"]
        self.qpos = self.file["observations"]["qpos"]
        self.cam_high = self.file["observations"]["images"]["cam_high"]
        self.cam_left_wrist = self.file["observations"]["images"]["cam_left_wrist"]
        self.cam_right_wrist = self.file["observations"]["images"]["cam_right_wrist"]
    
    def byte2img(self,bytes):
        recovered = np.frombuffer(bytes,np.uint8)
        img = cv2.imdecode(recovered,cv2.IMREAD_COLOR)
        return img

    def __len__(self):
        return self.action.shape[0]
    
    @property
    def size(self):
        return self.__len__()

    def fetch_pair(self,idx):
        ret = {}
        ret["action"] = self.action[idx]
        ret["qpos"] = self.qpos[idx]
        ret["cam_high"] = self.byte2img(self.cam_high[idx])
        ret["cam_left_wrist"] = self.byte2img(self.cam_left_wrist[idx])
        ret["cam_right_wrist"] = self.byte2img(self.cam_right_wrist[idx])
        return ret


class FrameDataset(Dataset):
    def __init__(self,step = 5,begin = 0):
        self.resolution = 256
        self.pairs = []
        self.step = step
        self.begin = begin
        with open("processed_data/instruction.json","r") as f:
            self.inst = json.loads(f.read())
        limits = 100
        print("[",end="")
        for i in range(0,limits):
            self.__loadpairs__(RoboTwinH5(f"processed_data/block_hammer_beat_D435_pkl_100/episode_{i}.hdf5")
                                ,self.inst["hammer_beat"])
            if i < 91:
                self.__loadpairs__(RoboTwinH5(f"processed_data/blocks_stack_easy_D435_pkl_95/episode_{i}.hdf5")
                                   ,self.inst["stack"])
            if i < 95:
                self.__loadpairs__(RoboTwinH5(f"processed_data/block_handover_D435_pkl_95/episode_{i}.hdf5")
                                   ,self.inst["handover"])
            print(".",end="")
        print("]")
        
    def __len__(self):
        return len(self.pairs)
    
    @property
    def length(self):
        return len(self.pairs)

    def __loadpairs__(self,imset:RoboTwinH5,inst):
        length = imset.size
        for i in range(self.begin,length,self.step):
            rand_inst = random.choice(inst)
            pair = {
                "instruction":rand_inst+(MAX_LENGTH-len(rand_inst))*" ",
                "src": cv2.resize(imset.fetch_pair(i)["cam_high"],(256,256)),
                "tgt": cv2.resize(imset.fetch_pair(min(i+50,length-1))["cam_high"],(256,256))
            }
            self.pairs.append(pair)
    def __getitem__(self, index):
        return self.pairs[index]


if __name__ == "__main__":
    path = "processed_data/block_handover_D435_pkl_95/episode_5.hdf5"
    # path = "processed_data/blocks_stack_easy_D435_pkl_95/episode_60.hdf5"
    # path = "processed_data/block_hammer_beat_D435_pkl_100/episode_90.hdf5"
    data = RoboTwinH5(path)
    datasets = FrameDataset(step=10)
    print(len(datasets))
    cv2.imwrite('src.png',datasets.pairs[3]['src'])
    cv2.imwrite('tgt.png',datasets.pairs[3]['tgt'])
    img = cv2.resize(data.fetch_pair(80)["cam_high"],(256,256))
    cv2.imwrite("tst.png",img)
    for i in range(0,data.size,1):
        pair = data.fetch_pair(i)
        cv2.imshow(f"cam [{i}]",pair["cam_high"])
        cv2.waitKey(10)
        cv2.destroyAllWindows()
    
    
    