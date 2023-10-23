import os
from typing import Union, Dict, Sequence, Tuple, List
import numpy as np
import datetime
import pandas as pd
import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader
from pytorch_lightning import LightningDataModule
from ...config import cfg
from .sevir_dataloader import SEVIRDataLoader


class SEVIRTorchDataset(TorchDataset):

    def __init__(self,
                 seq_len: int = 25,
                 raw_seq_len: int = 49,
                 sample_mode: str = "sequent",
                 stride: int = 12,
                 batch_size: int = 1,
                 layout: str = "NHWT",
                 num_shard: int = 1,
                 rank: int = 0,
                 split_mode: str = "uneven",
                 sevir_catalog: Union[str, pd.DataFrame] = None,
                 sevir_data_dir: str = None,
                 start_date: datetime.datetime = None,
                 end_date: datetime.datetime = None,
                 datetime_filter = None,
                 catalog_filter = "default",
                 shuffle: bool = False,
                 shuffle_seed: int = 1,
                 output_type = np.float32,
                 preprocess: bool = True,
                 rescale_method: str = "01",
                 verbose: bool = False):
        super(SEVIRTorchDataset, self).__init__()
        self.layout = layout
        self.sevir_dataloader = SEVIRDataLoader(
            data_types=["vil", ],
            seq_len=seq_len,
            raw_seq_len=raw_seq_len,
            sample_mode=sample_mode,
            stride=stride,
            batch_size=batch_size,
            layout=layout,
            num_shard=num_shard,
            rank=rank,
            split_mode=split_mode,
            sevir_catalog=sevir_catalog,
            sevir_data_dir=sevir_data_dir,
            start_date=start_date,
            end_date=end_date,
            datetime_filter=datetime_filter,
            catalog_filter=catalog_filter,
            shuffle=shuffle,
            shuffle_seed=shuffle_seed,
            output_type=output_type,
            preprocess=preprocess,
            rescale_method=rescale_method,
            downsample_dict=None,
            verbose=verbose)

    def __getitem__(self, index):
        data_dict = self.sevir_dataloader._idx_sample(index=index)
        return data_dict

    def __len__(self):
        return self.sevir_dataloader.__len__()

    def collate_fn(self, data_dict_list):
        r"""
        Parameters
        ----------
        data_dict_list:  list[Dict[str, torch.Tensor]]

        Returns
        -------
        merged_data: Dict[str, torch.Tensor]
            batch_size = len(data_dict_list) * data_dict["key"].batch_size
        """
        batch_dim = self.layout.find('N')
        data_list_dict = {
            key: [data_dict[key]
                  for data_dict in data_dict_list]
            for key in data_dict_list[0]}
        # TODO: key "mask" is not handled. Temporally fine since this func is not used
        data_list_dict.pop("mask", None)
        merged_dict = {
            key: torch.cat(data_list,
                           dim=batch_dim)
            for key, data_list in data_list_dict.items()}
        merged_dict["mask"] = None
        return merged_dict

    def get_torch_dataloader(self,
                             outer_batch_size=1,
                             collate_fn=None,
                             num_workers=1):
        # TODO: num_workers > 1
        r"""
        We set the batch_size in Dataset by default, so outer_batch_size should be 1.
        In this case, not using `collate_fn` can save time.
        """
        if outer_batch_size == 1:
            collate_fn = lambda x:x[0]
        else:
            if collate_fn is None:
                collate_fn = self.collate_fn
        dataloader = DataLoader(
            dataset=self,
            batch_size=outer_batch_size,
            collate_fn=collate_fn,
            num_workers=num_workers)
        return dataloader


def check_aws():
    r"""
    Check if aws cli is installed.
    """
    if os.system("which aws") != 0:
        raise RuntimeError("AWS CLI is not installed! Please install it first. See https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html")


def download_SEVIR(save_dir=None):
    r"""
    Downloaded dataset is saved in save_dir/sevir
    """

    check_aws()

    if save_dir is None:
        save_dir = cfg.datasets_dir
    sevir_dir = os.path.join(save_dir, "sevir")
    if os.path.exists(sevir_dir):
        raise FileExistsError(f"Path to save SEVIR dataset {sevir_dir} already exists!")
    else:
        os.makedirs(sevir_dir)
        os.system(f"aws s3 cp --no-sign-request s3://sevir/CATALOG.csv "
                  f"{os.path.join(sevir_dir, 'CATALOG.csv')}")
        os.system(f"aws s3 cp --no-sign-request --recursive s3://sevir/data/vil "
                  f"{os.path.join(sevir_dir, 'data', 'vil')}")

import random
import numpy as np
import cv2
import torch
import torchvision

import h5py

class PytorchDataGenerator(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, img_list, model_type = 'simple', validation = False, im_size=200, 
        # shuffle = True, 
        nt_start=0, nt_future=12, num_bins=None): # 
        self.nt_start = nt_start
        self.nt_future = nt_future
        self.nt_past = 49 - nt_future - nt_start
        self.num_bins = num_bins
        self.bins = None
        if num_bins == 12:
            self.bins = np.array([0, 7, 15, 71, 111, 133, 148, 159, 169, 181, 204, 219, 256])
        elif num_bins == 10: # This case copies value ranges from Earthformers!
            self.bins = np.array([0, 16, 31, 59, 74, 100, 133, 160, 181, 219, 256])
        elif num_bins == 7:
            self.bins = np.array([0, 7, 15, 111, 148, 169, 204, 256])
        elif num_bins == 3:
            # bins = np.array([0, 15, 111, 256])
            self.bins = np.array([0, 15, 74, 256]) # NEED TO MAKE NEW CASES WITH 3CLASSES HAVING DIFFERENT THRESHOLDS!
        ##
        self.model_imgs = img_list
        print(f'num ims: {len(img_list)}')
        # self.im_size = im_size
        # self.model_type = model_type
        # self.shuffle = shuffle
        if validation:
            self.crop_flag=0
            self.resize_flag=1
            self.type='validation'
        else:
            self.crop_flag=1
            self.resize_flag=1
            self.type='train'
        
    def __len__(self):
        # Denotes the number of batches per epoch
        return len(self.model_imgs)

    def __getitem__(self, index):
        # Generate one sample of data
        file_name = self.model_imgs[index]

        # if self.shuffle: # and i*batch_size >= len(file_list):
        #     # i=0
        #     np.random.shuffle(file_list)
        # file_chunk = file_list[i*batch_size:(i+1)*batch_size] # Perhaps this is the point to shuffle the list!
        ####
        # for file in file_chunk:
        eventFile = h5py.File(file_name, 'r')
        # eventFile.keys()
        ## Work on ir069!
        # Read the data
        ir069 = eventFile['ir069'][:,:,self.nt_start:(self.nt_start+self.nt_past)]
        # Put time dimension to first axis!
        ir069 = np.moveaxis(ir069, 2, 0)
        # Add the channel band!
        ir069 = np.expand_dims(ir069, axis=-1)
        ## Work on ir107!
        # Read the data
        ir107 = eventFile['ir107'][:,:,self.nt_start:(self.nt_start+self.nt_past)]
        #Put time dimension to first axis!
        ir107 = np.moveaxis(ir107, 2, 0)
        # Add the channel band!
        ir107 = np.expand_dims(ir107, axis=-1)
        ## Work on lght!
        # Read the data
        lght = eventFile['lght'][:,:,self.nt_start:(self.nt_start+self.nt_past)]
        #Put time dimension to first axis!
        lght = np.moveaxis(lght, 2, 0)
        # Add the channel band!
        lght = np.expand_dims(lght, axis=-1)
        ## Work on VIL!
        # Read the data
        # vil = eventFile['vil'][:,:,(self.nt_start+self.nt_past):(self.nt_start+self.nt_past+self.nt_future)] # CLOSED FOR EF!
        vil = eventFile['vil'][:,:,:] # OPENED FOR EF!
        # Put time dimension to first axis!
        # vil = tf.transpose(vil, perm=[2,0,1])
        vil = np.moveaxis(vil, 2, 0) # # CLOSED FOR EF!
        # Add the channel band!
        # vil = tf.expand_dims(vil, axis=-1)
        vil = np.expand_dims(vil, axis=-1)
        if self.num_bins: # Do this only if num_bins is not None!
            ## Time to binarize the VIL!
            vil = np.digitize(vil, self.bins)
            # Make sure the bins are from 0 to num_bins-1
            vil = vil - 1

        ## Z-normalization of the data!!!!
        PREPROCESS_SCALE_SEVIR = {'vis': 1,  # Not utilized in original paper
            'ir069': 1 / 1174.68,
            'ir107': 1 / 2562.43,
            'vil': 1 / 47.54,
            'lght': 1 / 0.60517}
        PREPROCESS_OFFSET_SEVIR = {'vis': 0,  # Not utilized in original paper
            'ir069': 3683.58,
            'ir107': 1552.80,
            'vil': - 33.44,
            'lght': - 0.02990}
        #
        ir069 = (ir069 + PREPROCESS_OFFSET_SEVIR['ir069']) * PREPROCESS_SCALE_SEVIR['ir069']
        ir107 = (ir107 + PREPROCESS_OFFSET_SEVIR['ir107']) * PREPROCESS_SCALE_SEVIR['ir107']
        lght = (lght + PREPROCESS_OFFSET_SEVIR['lght']) * PREPROCESS_SCALE_SEVIR['lght']
        if not self.num_bins:
            vil = (vil + PREPROCESS_OFFSET_SEVIR['vil']) * PREPROCESS_SCALE_SEVIR['vil']
        ####
        ir069 = torch.from_numpy(ir069)
        ir107 = torch.from_numpy(ir107)
        lght = torch.from_numpy(lght)
        vil = torch.from_numpy(vil)
        # #normalise image only if using mobilenet
        # if self.model_type=='mobilenet':
        #     X = torchvision.transforms.functional.normalize(X, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # else:
        #     pass
        # # get labels
        # if 'dog' in file_name:
        #     y=1
        # elif 'cat' in file_name:
        #     y=0
        ##### Not sure what to return....
        # return tuple((ir069, ir107, lght)), tuple(vil)
        # return ir069, ir107, lght, vil # # CLOSED FOR EF!
        # return vil[0:13, :, :, :], vil[13:25, :, :, :] # OPENED FOR EF!
        return vil[0:25, :, :, :] # OPENED FOR EF!
    
    #### This is copy-pasted from the original code!
    def get_torch_dataloader(self,
                             outer_batch_size=1,
                             collate_fn=None,
                             num_workers=1):
        # TODO: num_workers > 1
        r"""
        We set the batch_size in Dataset by default, so outer_batch_size should be 1.
        In this case, not using `collate_fn` can save time.
        """
        if outer_batch_size == 2: # 1:
            collate_fn = lambda x:x[0] # This is the original...
            # collate_fn = lambda x:x #[0]
        else:
            if collate_fn is None:
                collate_fn = self.collate_fn
        dataloader = DataLoader(
            dataset=self,
            batch_size=1, # outer_batch_size,
            collate_fn=collate_fn,
            num_workers=num_workers)
        return dataloader

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class kucukSevirDataset(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, 
                img_list, 
                model_type = 'simple', 
                validation = False, im_size=200, 
                # data_dir = '/home/kucuk/homedata/version2023/',
                # shuffle = True, 
                nt_start=0, nt_future=12, num_bins=None): # 
        # super().__init__()
        self.nt_start = nt_start
        self.nt_future = nt_future
        self.nt_past = 49 - nt_future - nt_start
        self.num_bins = num_bins
        self.bins = None
        if num_bins == 12:
            self.bins = np.array([0, 7, 15, 71, 111, 133, 148, 159, 169, 181, 204, 219, 256])
        elif num_bins == 10: # This case copies value ranges from Earthformers!
            self.bins = np.array([0, 16, 31, 59, 74, 100, 133, 160, 181, 219, 256])
        elif num_bins == 7:
            self.bins = np.array([0, 7, 15, 111, 148, 169, 204, 256])
        elif num_bins == 3:
            # bins = np.array([0, 15, 111, 256])
            self.bins = np.array([0, 15, 74, 256]) # NEED TO MAKE NEW CASES WITH 3CLASSES HAVING DIFFERENT THRESHOLDS!
        ##
        self.img_list = img_list
        print(f'num ims: {len(img_list)}')

        
    def __len__(self):
        # Denotes the number of batches per epoch
        return len(self.img_list)

    def __getitem__(self, index):
        # Generate one sample of data
        file_name = self.img_list[index]

        # if self.shuffle: # and i*batch_size >= len(file_list):
        #     # i=0
        #     np.random.shuffle(file_list)
        # file_chunk = file_list[i*batch_size:(i+1)*batch_size] # Perhaps this is the point to shuffle the list!
        ####
        # for file in file_chunk:
        eventFile = h5py.File(file_name, 'r')
        # eventFile.keys()
        ## Work on ir069!
        # Read the data
        ir069 = eventFile['ir069'][:,:,self.nt_start:(self.nt_start+self.nt_past)]
        # Put time dimension to first axis!
        ir069 = np.moveaxis(ir069, 2, 0)
        # Add the channel band!
        ir069 = np.expand_dims(ir069, axis=-1)
        ## Work on ir107!
        # Read the data
        ir107 = eventFile['ir107'][:,:,self.nt_start:(self.nt_start+self.nt_past)]
        #Put time dimension to first axis!
        ir107 = np.moveaxis(ir107, 2, 0)
        # Add the channel band!
        ir107 = np.expand_dims(ir107, axis=-1)
        ## Work on lght!
        # Read the data
        lght = eventFile['lght'][:,:,self.nt_start:(self.nt_start+self.nt_past)]
        #Put time dimension to first axis!
        lght = np.moveaxis(lght, 2, 0)
        # Add the channel band!
        lght = np.expand_dims(lght, axis=-1)
        ## Work on VIL!
        # Read the data
        # vil = eventFile['vil'][:,:,(self.nt_start+self.nt_past):(self.nt_start+self.nt_past+self.nt_future)] # CLOSED FOR EF!
        vil = eventFile['vil'][:,:,:] # OPENED FOR EF!
        # Put time dimension to first axis!
        # vil = tf.transpose(vil, perm=[2,0,1])
        vil = np.moveaxis(vil, 2, 0) # # CLOSED FOR EF!
        # Add the channel band!
        # vil = tf.expand_dims(vil, axis=-1)
        vil = np.expand_dims(vil, axis=-1)
        if self.num_bins: # Do this only if num_bins is not None!
            ## Time to binarize the VIL!
            vil = np.digitize(vil, self.bins)
            # Make sure the bins are from 0 to num_bins-1
            vil = vil - 1

        ## Z-normalization of the data!!!!
        PREPROCESS_SCALE_SEVIR = {'vis': 1,  # Not utilized in original paper
            'ir069': 1 / 1174.68,
            'ir107': 1 / 2562.43,
            'vil': 1 / 47.54,
            'lght': 1 / 0.60517}
        PREPROCESS_OFFSET_SEVIR = {'vis': 0,  # Not utilized in original paper
            'ir069': 3683.58,
            'ir107': 1552.80,
            'vil': - 33.44,
            'lght': - 0.02990}
        #
        ir069 = (ir069 + PREPROCESS_OFFSET_SEVIR['ir069']) * PREPROCESS_SCALE_SEVIR['ir069']
        ir107 = (ir107 + PREPROCESS_OFFSET_SEVIR['ir107']) * PREPROCESS_SCALE_SEVIR['ir107']
        lght = (lght + PREPROCESS_OFFSET_SEVIR['lght']) * PREPROCESS_SCALE_SEVIR['lght']
        if not self.num_bins:
            vil = (vil + PREPROCESS_OFFSET_SEVIR['vil']) * PREPROCESS_SCALE_SEVIR['vil']
        ####
        ir069 = torch.from_numpy(ir069)
        ir107 = torch.from_numpy(ir107)
        lght = torch.from_numpy(lght)
        vil = torch.from_numpy(vil)
        # #normalise image only if using mobilenet
        # if self.model_type=='mobilenet':
        #     X = torchvision.transforms.functional.normalize(X, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # else:
        #     pass
        # # get labels
        # if 'dog' in file_name:
        #     y=1
        # elif 'cat' in file_name:
        #     y=0
        ##### Not sure what to return....
        # return tuple((ir069, ir107, lght)), tuple(vil)
        # return ir069, ir107, lght, vil # # CLOSED FOR EF!
        # return vil[0:13, :, :, :], vil[13:25, :, :, :] # OPENED FOR EF!
        sample = {'vil_past': vil[0:13, :, :, :], 'vil_future': vil[13:25, :, :, :]}
        return sample
    
    #### This is copy-pasted from the original code!
    def get_torch_dataloader(self,
                             outer_batch_size=1,
                             collate_fn=None,
                             num_workers=1):
        # TODO: num_workers > 1
        r"""
        We set the batch_size in Dataset by default, so outer_batch_size should be 1.
        In this case, not using `collate_fn` can save time.
        """
        if outer_batch_size == 2: # 1:
            collate_fn = lambda x:x[0] # This is the original...
            # collate_fn = lambda x:x #[0]
        else:
            if collate_fn is None:
                collate_fn = self.collate_fn
        dataloader = DataLoader(
            dataset=self,
            batch_size=1, # outer_batch_size,
            collate_fn=collate_fn,
            num_workers=num_workers)
        return dataloader


class SEVIRLightningDataModule(LightningDataModule):

    def __init__(self,
                 ####
                 trainFiles, valFiles, testFiles = None,
                 ####
                 seq_len: int = 25,
                 sample_mode: str = "sequent",
                 stride: int = 12,
                 batch_size: int = 1,
                 layout: str = "NHWT",
                 output_type = np.float32,
                 preprocess: bool = True,
                 rescale_method: str = "01",
                 verbose: bool = False,
                 # datamodule_only
                 dataset_name: str = "sevir",
                 start_date: Tuple[int] = None,
                 train_val_split_date: Tuple[int] = (2019, 1, 1),
                 train_test_split_date: Tuple[int] = (2019, 6, 1),
                 end_date: Tuple[int] = None,
                 num_workers: int = 1,
                 ):
        super(SEVIRLightningDataModule, self).__init__()
        self.seq_len = seq_len
        self.sample_mode = sample_mode
        self.stride = stride
        self.batch_size = batch_size
        self.layout = layout
        self.output_type = output_type
        self.preprocess = preprocess
        self.rescale_method = rescale_method
        self.verbose = verbose
        self.num_workers = num_workers
        ####
        self.trainFiles = trainFiles
        self.valFiles = valFiles
        self.testFiles = testFiles
        ####
        if dataset_name == "sevir":
            sevir_root_dir = os.path.join(cfg.datasets_dir, "sevir")
            catalog_path = os.path.join(sevir_root_dir, "CATALOG.csv")
            raw_data_dir = os.path.join(sevir_root_dir, "data")
            raw_seq_len = 49
            interval_real_time = 5
            img_height = 384
            img_width = 384
        elif dataset_name == "sevir_lr":
            sevir_root_dir = os.path.join(cfg.datasets_dir, "sevir_lr")
            catalog_path = os.path.join(sevir_root_dir, "CATALOG.csv")
            raw_data_dir = os.path.join(sevir_root_dir, "data")
            raw_seq_len = 25
            interval_real_time = 10
            img_height = 128
            img_width = 128
        else:
            raise ValueError(f"Wrong dataset name {dataset_name}. Must be 'sevir' or 'sevir_lr'.")
        self.dataset_name = dataset_name
        self.sevir_root_dir = sevir_root_dir
        self.catalog_path = catalog_path
        self.raw_data_dir = raw_data_dir
        self.raw_seq_len = raw_seq_len
        self.interval_real_time = interval_real_time
        self.img_height = img_height
        self.img_width = img_width
        # train val test split
        self.start_date = datetime.datetime(*start_date) \
            if start_date is not None else None
        self.train_val_split_date = datetime.datetime(*train_val_split_date)
        self.train_test_split_date = datetime.datetime(*train_test_split_date)
        self.end_date = datetime.datetime(*end_date) \
            if end_date is not None else None

    def prepare_data(self) -> None:
        if os.path.exists(self.sevir_root_dir):
            # Further check
            assert os.path.exists(self.catalog_path), f"CATALOG.csv not found! Should be located at {self.catalog_path}"
            assert os.path.exists(self.raw_data_dir), f"SEVIR data not found! Should be located at {self.raw_data_dir}"
        else:
            if self.dataset_name == "sevir":
                download_SEVIR()
            else:  # "sevir_lr"
                raise NotImplementedError

    def setup(self, stage = None) -> None:
        # self.sevir_train = SEVIRTorchDataset(
        #     sevir_catalog=self.catalog_path,
        #     sevir_data_dir=self.raw_data_dir,
        #     raw_seq_len=self.raw_seq_len,
        #     split_mode="uneven",
        #     shuffle=True,
        #     seq_len=self.seq_len,
        #     stride=self.stride,
        #     sample_mode=self.sample_mode,
        #     batch_size=self.batch_size,
        #     layout=self.layout,
        #     num_shard=1, rank=0,
        #     start_date=self.start_date,
        #     end_date=self.train_val_split_date,
        #     output_type=self.output_type,
        #     preprocess=self.preprocess,
        #     rescale_method=self.rescale_method,
        #     verbose=self.verbose,)
        self.sevir_train = kucukSevirDataset(
            self.trainFiles, validation = False)

        self.sevir_val = SEVIRTorchDataset(
            sevir_catalog=self.catalog_path,
            sevir_data_dir=self.raw_data_dir,
            raw_seq_len=self.raw_seq_len,
            split_mode="uneven",
            shuffle=False,
            seq_len=self.seq_len,
            stride=self.stride,
            sample_mode=self.sample_mode,
            batch_size=self.batch_size,
            layout=self.layout,
            num_shard=1, rank=0,
            start_date=self.train_val_split_date,
            end_date=self.train_test_split_date,
            output_type=self.output_type,
            preprocess=self.preprocess,
            rescale_method=self.rescale_method,
            verbose=self.verbose, )
        self.sevir_test = SEVIRTorchDataset(
            sevir_catalog=self.catalog_path,
            sevir_data_dir=self.raw_data_dir,
            raw_seq_len=self.raw_seq_len,
            split_mode="uneven",
            shuffle=False,
            seq_len=self.seq_len,
            stride=self.stride,
            sample_mode=self.sample_mode,
            batch_size=self.batch_size,
            layout=self.layout,
            num_shard=1, rank=0,
            start_date=self.train_test_split_date,
            end_date=self.end_date,
            output_type=self.output_type,
            preprocess=self.preprocess,
            rescale_method=self.rescale_method,
            verbose=self.verbose,)
        self.sevir_predict = SEVIRTorchDataset(
            sevir_catalog=self.catalog_path,
            sevir_data_dir=self.raw_data_dir,
            raw_seq_len=self.raw_seq_len,
            split_mode="uneven",
            shuffle=False,
            seq_len=self.seq_len,
            stride=self.stride,
            sample_mode=self.sample_mode,
            batch_size=self.batch_size,
            layout=self.layout,
            num_shard=1, rank=0,
            start_date=self.train_test_split_date,
            end_date=self.end_date,
            output_type=self.output_type,
            preprocess=self.preprocess,
            rescale_method=self.rescale_method,
            verbose=self.verbose,)
        
    def train_dataloader(self):
        return self.sevir_train.get_torch_dataloader(num_workers=self.num_workers, outer_batch_size=2)

    def val_dataloader(self):
        return self.sevir_val.get_torch_dataloader(num_workers=self.num_workers)

    def test_dataloader(self):
        return self.sevir_test.get_torch_dataloader(num_workers=self.num_workers)

    def predict_dataloader(self):
        return self.sevir_predict.get_torch_dataloader(num_workers=self.num_workers)

    @property
    def num_train_samples(self):
        return len(self.sevir_train)

    @property
    def num_val_samples(self):
        return len(self.sevir_val)

    @property
    def num_test_samples(self):
        return len(self.sevir_test)
