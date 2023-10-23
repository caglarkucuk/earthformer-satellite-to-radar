import glob
import random
import numpy as np
import h5py

import torch
import torchvision
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import pytorch_lightning as pl

import scipy # Upsampling of the lght channel!
from utils.fixedValues import PREPROCESS_SCALE_SEVIR, PREPROCESS_OFFSET_SEVIR # Normalization values for SEVIR dataset

class kucukSevirDataset(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, 
                img_list, 
                nt_start=0, 
                nt_future=24,
        ): # 
        self.nt_start = nt_start
        self.nt_future = nt_future
        self.nt_past = 49 - nt_future - nt_start
        ##
        self.img_list = img_list
        print(f'num ims: {len(img_list)}')

        
    def __len__(self):
        # Denotes the number of batches per epoch
        return len(self.img_list)

    def __getitem__(self, index):
        # Generate one sample of data
        file_name = self.img_list[index]
        eventFile = h5py.File(file_name, 'r')
        ## Work on ir069!
        # Read the data
        ir069 = eventFile['ir069'][:,:,self.nt_start:(self.nt_start+self.nt_past)]
        # Put time dimension to first axis!
        ir069 = np.moveaxis(ir069, 2, 0)
        ## Work on ir107!
        # Read the data
        ir107 = eventFile['ir107'][:,:,self.nt_start:(self.nt_start+self.nt_past)]
        #Put time dimension to first axis!
        ir107 = np.moveaxis(ir107, 2, 0)
        ## Work on lght!
        # Read the data
        lght = eventFile['lght'][:,:,self.nt_start:(self.nt_start+self.nt_past)]
        # Put time dimension to first axis!
        lght = np.moveaxis(lght, 2, 0)
        lght = scipy.ndimage.zoom(lght, (1, 4,4), order=0) # simple nearest interpolation here...
        ## Work on VIL!
        # Read the data
        vil = eventFile['vil'][:,:,(self.nt_start+self.nt_past):(self.nt_start+self.nt_past+self.nt_future)]
        # Put time dimension to first axis!
        vil = np.moveaxis(vil, 2, 0)
        # Add the channel band!
        vil = np.expand_dims(vil, axis=-1)
        #
        ir069 = (ir069 + PREPROCESS_OFFSET_SEVIR['ir069']) * PREPROCESS_SCALE_SEVIR['ir069']
        ir107 = (ir107 + PREPROCESS_OFFSET_SEVIR['ir107']) * PREPROCESS_SCALE_SEVIR['ir107']
        lght = (lght + PREPROCESS_OFFSET_SEVIR['lght']) * PREPROCESS_SCALE_SEVIR['lght']
        vil = (vil + PREPROCESS_OFFSET_SEVIR['vil']) * PREPROCESS_SCALE_SEVIR['vil']
        #
        ir069 = torch.from_numpy(ir069.astype(np.float32))
        ir107 = torch.from_numpy(ir107.astype(np.float32))
        lght = torch.from_numpy(lght.astype(np.float32))
        vil = torch.from_numpy(vil.astype(np.float32))
        # Stack the channels!
        ir_total = torch.stack((ir069, ir107, lght), dim=-1)
        # Create the sample to return!
        sample = {'vil_past': ir_total, 'vil_future': vil, 'name': eventFile.attrs['name']}
        return sample

class kucukSevirDataModule(pl.LightningDataModule):
    """define __init__, setup, and loaders for 3sets of data!"""
    def __init__(self, 
                params, # data_params, training_params, 
                data_dir,
                stage='train', 
                ):
        super().__init__() 
        self.params = params

        ## Let's define the fileList here!
        trainFilesTotal = glob.glob(data_dir+'train/*') 
        random.seed(0)
        random.shuffle(trainFilesTotal)  # shuffle the file list randomly

        self.testFiles = glob.glob(data_dir+'test/*')

        self.trainFiles = trainFilesTotal[:8192] # divides by 32 w/o remainder.... 8566 is total number of files!
        self.trainFiles = trainFilesTotal[2:770] # ##### FATAL!!!
        # self.valFiles = trainFilesTotal[8192:]
        self.valFiles = trainFilesTotal[-2:] ##### FATAL!!!!

    def setup(self, stage=None):
        if stage == 'train' or stage is None:
            self.train_dataset = kucukSevirDataset(img_list=self.trainFiles)
            self.val_dataset = kucukSevirDataset(img_list=self.valFiles)

        if stage == 'test':
            self.test_dataset = kucukSevirDataset(img_list=self.testFiles)
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.params['BATCH_SIZE'], shuffle=True, num_workers=self.params['NUM_WORKERS'])

    def val_dataloader(self):
        # No shuffling!
        return DataLoader(self.val_dataset, batch_size=self.params['BATCH_SIZE'], shuffle=False, num_workers=self.params['NUM_WORKERS'])

    def test_dataloader(self):
        # Batch size is 1 for testing! No shuffling!
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=self.params['NUM_WORKERS'])
    
    #### These properties are added from the original repository!
    @property
    def num_train_samples(self):
        return len(self.train_dataset)

    @property
    def num_val_samples(self):
        return len(self.val_dataset)

    @property
    def num_test_samples(self):
        return len(self.test_dataset)
