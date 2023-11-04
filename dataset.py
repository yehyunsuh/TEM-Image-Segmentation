import os
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import pandas as pd
import numpy as np
import hyperspy.api as hs
import cv2
import torchvision.transforms as transforms
from border import get_border


class CustomDataset(Dataset):
    def __init__(self, 
                 args,
                 dtype,
                 transform = None
                ): 
        self.args = args
        pnp = 'pore'
    
        # csv that contains metadata
        csv_path = args.data_dir+'/'+dtype+'/'+pnp+'/'+dtype+'.csv'
        self.df = pd.read_csv(csv_path)
        
        # file that have .ser files
        self.image_dir = args.data_dir+'/'+dtype+'/'+pnp+'/org_img'
        self.image_list = list(self.df['ser_filename'])
        
        # file that have label's pickle file
        self.pickle_dir = args.data_dir+'/'+dtype+'/'+pnp+'/pickle'
        self.pickle_list = list(self.df['pickle_filename'])
        
        # file that have label image file 
        self.lab_img_dir = args.data_dir+'/'+dtype+'/'+pnp+'/mask'
        self.lab_img_list = list(self.df['jpg(tif)_filename'])
        
        # file that have pore's info pickle files 
        self.info_dir = args.data_dir+'/'+dtype+'/'+pnp+'/info_pickle'
        self.info_list = list(self.df['pickle_filename'])
        
        self.transform = transform
    
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.image_list[index])
        
        # input data (2048, 2048), imageio.core.util.Array
        s = hs.load(image_path)
        x_data = s.data
        
        # label data (2048, 2048), numpy.ndarray
        with open(self.pickle_dir+'/'+self.pickle_list[index],"rb") as fi:
            y_data = pickle.load(fi)
            
        # pore info, list, [# of pore, [# of pore's pixel]]
        with open(self.info_dir+'/'+self.info_list[index],"rb") as fi:
            pixel_info = pickle.load(fi)
        
        num_pore = pixel_info[0]
        num_pixel = pixel_info[1]
        
        if self.transform:
            x_data = self.transform(x_data.copy())
            y_data = self.transform(y_data.copy())
        
        img = cv2.imread(self.lab_img_dir+'/'+self.lab_img_list[index])
        w_mask = get_border(self.args, img)
        w_mask = w_mask.reshape(1,512,512)
        
        return x_data, y_data, w_mask, num_pore, num_pixel   
        

def load_data(args):
    print("---------- Starting Loading Dataset ----------")
    IMAGE_RESIZE = args.image_resize
    BATCH_SIZE = args.batch_size
    
    if args.augmentation:
        train_transform = transforms.Compose([
            transforms.ToTensor(),   
            transforms.Resize((IMAGE_RESIZE, IMAGE_RESIZE), antialias=None),
#             transforms.Normalize(mean = 0.464, std= 0.034),
            transforms.RandomHorizontalFlip(p=0.1), # flip
            transforms.RandomVerticalFlip(p=0.1),    
            transforms.RandomRotation(degrees= (-15, 15)), # rotate
#             transforms.ElasticTransform(), # elastic transform 
        ])
        valid_transform = transforms.Compose([
            transforms.ToTensor(),   
            transforms.Resize((IMAGE_RESIZE, IMAGE_RESIZE), antialias=None),
#             transforms.Normalize(mean = 0.464, std= 0.034),
#             transforms.RandomHorizontalFlip(p=0.8), # flip
#             transforms.RandomVerticalFlip(p=0.6),    
#             transforms.RandomRotation(degrees= (-90, 90)), # rotate
#             transforms.ElasticTransform(), # elastic transform 
         ])
        
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),   
            transforms.Resize((IMAGE_RESIZE, IMAGE_RESIZE), antialias=None),
        ])
    
        valid_transform = transforms.Compose([
            transforms.ToTensor(),   
            transforms.Resize((IMAGE_RESIZE, IMAGE_RESIZE), antialias=None),
        ])

    
    train_dataset = CustomDataset(
        args, 'train', train_transform
    )
    val_dataset = CustomDataset(
        args, 'validation', valid_transform
    )
    
    print('len of train dataset: ', train_dataset.__len__())
    print('len of val dataset: ', val_dataset.__len__()) 
    
    num_workers = 4 * torch.cuda.device_count()
    train_loader = DataLoader(
        train_dataset, shuffle=True, batch_size=BATCH_SIZE, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, shuffle=False, batch_size=1, num_workers=num_workers
    )
    
    print("---------- Loading Dataset Done ----------")
    
    return train_loader, val_loader