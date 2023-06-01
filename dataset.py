import os
import torch
from torch.utils.data import Dataset
import pickle
import pandas as pd
import numpy as np
import hyperspy.api as hs
import torchvision.transforms as transforms


class CustomDataset(Dataset):
    def __init__(self, 
                 data_dir,
                 train = True, 
                 pore = True,
                 transform = None
                ): 
        
        dtype = 'train' if train else 'validation'
        pnp = 'pore' if pore else 'np'
    
        # csv that contains metadata
        csv_path = data_dir+'/'+dtype+'/'+dtype+'.csv'
        self.df = pd.read_csv(csv_path)
        
        # file that have .ser files
        self.image_dir = data_dir+'/'+dtype+'/org_img'
        self.image_list = os.listdir(self.image_dir)
        
        # file that have label's pickle file
        self.pickle_dir = data_dir+'/'+dtype+'/'+pnp+'/pickle'
        self.pickle_list = os.listdir(self.pickle_dir)
        
        # file that have label image file 
        self.lab_img_dir = data_dir+'/'+dtype+'/'+pnp+'/img'
        self.lab_img_list = os.listdir(self.lab_img_dir)
        
        self.transform = transform
        
    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.image_list[index])
        
        # input data (2048, 2048), imageio.core.util.Array
        s = hs.load(image_path)
        x_data = s.data
        
        # label data (2048, 2048), numpy.ndarray
        with open(self.pickle_dir+'/'+self.pickle_list[index],"rb") as fi:
            y_data = pickle.load(fi)
        
        if self.transform:
            x_data = self.transform(x_data)
            y_data = self.transform(y_data)
        
        return x_data, y_data
        
    def __len__(self):
        return len(self.image_list)