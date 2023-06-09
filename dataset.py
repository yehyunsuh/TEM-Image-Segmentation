import os
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import pandas as pd
import numpy as np
import hyperspy.api as hs
import torchvision.transforms as transforms


class CustomDataset(Dataset):
    def __init__(self, 
                 args,
                 dtype, #'train' pr 'validation'
                 pnp, # 'np' or 'pore' 
                 transform = None
                ): 
        self.args = args
    
        # csv that contains metadata
        csv_path = args.data_dir+'/'+dtype+'/'+pnp+'/'+dtype+'.csv'
        self.df = pd.read_csv(csv_path)
        
        # file that have .ser files
        self.image_dir = args.data_dir+'/'+dtype+'/org_img'
        self.image_list = list(self.df['ser_filename'])
        
        # file that have label's pickle file
        self.pickle_dir = args.data_dir+'/'+dtype+'/'+pnp+'/pickle'
        self.pickle_list = list(self.df['pickle_filename'])
        
        # file that have label image file 
        self.lab_img_dir = args.data_dir+'/'+dtype+'/'+pnp+'/img'
        self.lab_img_list = list(self.df['jpg(tif)_filename'])
        
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
        
        if self.transform:
            x_data = self.transform(x_data.copy())
            y_data = self.transform(y_data.copy())
        
        return x_data, y_data
        

def load_data(args):
    print("---------- Starting Loading Dataset ----------")
    IMAGE_RESIZE = args.image_resize
    BATCH_SIZE = args.batch_size
    
    if args.augmentation:
        train_transform = transforms.Compose([
            transforms.ToTensor(),   
            transforms.Resize((IMAGE_RESIZE, IMAGE_RESIZE)),
#             transforms.Normalize(mean = 0.464, std= 0.034),
#             transforms.RandomHorizontalFlip(p=0.8), # flip
#             transforms.RandomVerticalFlip(p=0.6)    
#             transforms.RandomRotation(degrees= (-90, 90)), # rotate
#             transforms.ElasticTransform(), # elastic transform 
        ])
        valid_transform = transforms.Compose([
            transforms.ToTensor(),   
            transforms.Resize((IMAGE_RESIZE, IMAGE_RESIZE)),
#             transforms.Normalize(mean = 0.464, std= 0.034),
#             transforms.RandomHorizontalFlip(p=0.8), # flip
#             transforms.RandomVerticalFlip(p=0.6)    
#             transforms.RandomRotation(degrees= (-90, 90)), # rotate
#             transforms.ElasticTransform(), # elastic transform 
         ])
        
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),   
            transforms.Resize((IMAGE_RESIZE, IMAGE_RESIZE)),
        ])
    
        valid_transform = transforms.Compose([
            transforms.ToTensor(),   
            transforms.Resize((IMAGE_RESIZE, IMAGE_RESIZE)),
        ])

    
    np_train_dataset = CustomDataset(
        args, 'train', 'np', train_transform
    )
    np_val_dataset = CustomDataset(
        args, 'validation', 'np', valid_transform
    )
    
    print('len of np train dataset: ', len(np_train_dataset))
    print('len of np val dataset: ', len(np_val_dataset))     
    
    pore_train_dataset = CustomDataset(
        args, 'train', 'pore', train_transform
    )
    pore_val_dataset = CustomDataset(
        args, 'validation', 'pore', valid_transform
    )
    
    print('len of pore train dataset: ', len(pore_train_dataset))
    print('len of pore val dataset: ', len(pore_val_dataset)) 
    
    num_workers = 4 * torch.cuda.device_count()
    
    np_train_loader = DataLoader(
        np_train_dataset, shuffle=True, batch_size=BATCH_SIZE, num_workers=num_workers
    )
    np_val_loader = DataLoader(
        np_val_dataset, shuffle=False, batch_size=1, num_workers=num_workers
    )
    
    pore_train_loader = DataLoader(
        pore_train_dataset, shuffle=True, batch_size=BATCH_SIZE, num_workers=num_workers
    )
    pore_val_loader = DataLoader(
        pore_val_dataset, shuffle=False, batch_size=1, num_workers=num_workers
    )    
    
    print("---------- Loading Dataset Done ----------")
    
    return np_train_loader, np_val_loader, pore_train_loader, pore_val_loader 