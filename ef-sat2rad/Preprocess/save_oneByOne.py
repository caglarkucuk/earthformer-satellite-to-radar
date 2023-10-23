
# # Creating manual catalogs, once again!
# ## Procedure is as follows:
# 
# 1) Filter for pct_missing==0
# 2) Filter for having 4 modalities as ['ir069','ir107','vil', 'lght']
# 
# 3) Drop duplicates
# 4) Divide for train & test
# 5) Summarise num of samples for both part!

import pandas as pd
import numpy as np
import h5py
import datetime
import gc
from tqdm.autonotebook import tqdm
import pickle

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("-dataMod", "--dataMod", default="test", help="test or train data to work on",
    choices=["train", "test"])
parser.add_argument("-dirBase", "--dirBase", 
    help="Base directory where all other data is kept in")
parser.add_argument("-dirOut", "--dirOut", type=str, 
    help="Sub directory where the data will be written!")
args = vars(parser.parse_args())

dataMod = args["dataMod"]
dirBase = args["dirBase"]
dirOut = args["dirOut"]

def lght_to_grid(data):
    """
    TAKEN FROM SEVIR REPO!

    Converts SEVIR lightning data stored in Nx5 matrix to an LxLx49 tensor representing
    flash counts per pixel per frame
    
    Parameters
    ----------
    data  np.array
       SEVIR lightning event (Nx5 matrix)
       
    Returns
    -------
    np.array 
       LxLx49 tensor containing pixel counts
    """
    FRAME_TIMES = np.arange(-120.0,125.0,5) * 60 # in seconds
    out_size = (48,48,len(FRAME_TIMES))
    if data.shape[0]==0:
        return np.zeros(out_size,dtype=np.float32)

    # filter out points outside the grid
    x,y=data[:,3],data[:,4]
    m=np.logical_and.reduce( [x>=0,x<out_size[0],y>=0,y<out_size[1]] )
    # data=data[m,:]
    if m[m].shape[0] == 0: #data.shape[0]==0:
        return np.zeros(out_size,dtype=np.float32)
    else:
        data=data[m,:]

    # Filter/separate times
    # compute z coodinate based on bin locaiton times
    t=data[:,0]
    z=np.digitize(t,FRAME_TIMES)-1
    z[z==-1]=0 # special case:  frame 0 uses lght from frame 1

    x=data[:,3].astype(np.int64)
    y=data[:,4].astype(np.int64)

    k=np.ravel_multi_index(np.array([y,x,z]),out_size)
    n = np.bincount(k,minlength=np.prod(out_size))
    return np.reshape(n,out_size).astype(np.float32)

def read_lght_data( sample_event, data_path):
    """
    TAKEN FROM SEVIR REPO!

    Reads lght data from SEVIR and maps flash counts onto a grid  
    
    Parameters
    ----------
    sample_event   pd.DataFrame
        SEVIR catalog rows matching a single ID
    data_path  str
        Location of SEVIR data
    
    Returns
    -------
    np.array 
       LxLx49 tensor containing pixel counts for selected event
    
    """
    fn = sample_event[sample_event.img_type=='lght'].squeeze().file_name
    id = sample_event[sample_event.img_type=='lght'].squeeze().id
    with h5py.File(data_path + '/' + fn,'r') as hf:
        data      = hf[id][:] 
    return lght_to_grid(data)

#### 0) Load the catalog!
catalog = pd.read_csv(dirBase+'CATALOG.csv', low_memory=False)

# 1) Filter for pct_missing==0
cat_noZero = catalog[catalog['pct_missing']==0.0]
print("Events with no missing value: ", cat_noZero.shape)

# 2) Filter for having 4 modalities as ['ir069','ir107','vil', 'lght']
img_types = set(['ir069','ir107','vil', 'lght'])

# Group by event id, and filter to only events that have all desired img_types
events = cat_noZero.groupby('id').filter(lambda x: img_types.issubset(set(x['img_type']))).groupby('id')
event_ids = list(events.groups.keys())
print('Found %d events matching' % len(event_ids),img_types)

cat_full = cat_noZero[cat_noZero['id'].isin(event_ids)]
print("Events w/ 4 modalities: ", cat_full.shape)

# 3) Drop duplicates
cat_noZero_noDub = cat_full.drop_duplicates(subset=['id', 'img_type'])
print("Events with no dublicates: ", cat_noZero_noDub.shape)

# 4) Divide for train & test
train_cat_noZero_noDub = cat_noZero_noDub[cat_noZero_noDub['time_utc']<'2019-06-01']
test_cat_noZero_noDub = cat_noZero_noDub[cat_noZero_noDub['time_utc']>='2019-06-01']
print("Train events: ", train_cat_noZero_noDub.shape)
print("Test events: ", test_cat_noZero_noDub.shape)

# 5) Summarise and sanity check!
train_cat_noZero_noDub_ir069 = train_cat_noZero_noDub[train_cat_noZero_noDub['img_type']=='ir069']
test_cat_noZero_noDub_ir069 = test_cat_noZero_noDub[test_cat_noZero_noDub['img_type']=='ir069']
print("Train events w/ ir069: ", train_cat_noZero_noDub_ir069.shape)
print("Test events w/ ir069: ", test_cat_noZero_noDub_ir069.shape)

train_cat_noZero_noDub_vil = train_cat_noZero_noDub[train_cat_noZero_noDub['img_type']=='vil']
test_cat_noZero_noDub_vil = test_cat_noZero_noDub[test_cat_noZero_noDub['img_type']=='vil']
print("Train events w/ vil: ", train_cat_noZero_noDub_vil.shape)
print("Test events w/ vil: ", test_cat_noZero_noDub_vil.shape)

train_cat_noZero_noDub_lght = train_cat_noZero_noDub[train_cat_noZero_noDub['img_type']=='lght']
test_cat_noZero_noDub_lght = test_cat_noZero_noDub[test_cat_noZero_noDub['img_type']=='lght']
print("Train events w/ lght: ", train_cat_noZero_noDub_lght.shape)
print("Test events w/ lght: ", test_cat_noZero_noDub_lght.shape)


if dataMod == 'train':
    myCatalog =  cat_noZero_noDub[cat_noZero_noDub['time_utc']<'2019-06-01']
elif dataMod == 'test':
    myCatalog = cat_noZero_noDub[cat_noZero_noDub['time_utc']>='2019-06-01']

# Divide the catalog for convenience!
catalog_ir069 = myCatalog[myCatalog['img_type'] == 'ir069'].reset_index()
catalog_ir107 = myCatalog[myCatalog['img_type'] == 'ir107'].reset_index()
catalog_lght = myCatalog[myCatalog['img_type'] == 'lght'].reset_index()
catalog_vil = myCatalog[myCatalog['img_type'] == 'vil'].reset_index()

# print(catalog_ir069.shape, catalog_ir107.shape, catalog_lght.shape, catalog_vil.shape)


for i in tqdm(range(len(catalog_ir069))):
    # 1) Read the name!
    tmp_id = catalog_ir069['id'][i]

    ## 2) Read relevant events!
    # Prepare the catalogs
    tmpCat_ir069 = catalog_ir069.loc[catalog_ir069['id'] == tmp_id].reset_index()
    tmpCat_ir107 = catalog_ir107.loc[catalog_ir107['id'] == tmp_id].reset_index()
    tmpCat_lght = catalog_lght.loc[catalog_lght['id'] == tmp_id].reset_index()
    tmpCat_vil = catalog_vil.loc[catalog_vil['id'] == tmp_id].reset_index()

    # Read the events
    with h5py.File(dirBase+tmpCat_ir069['file_name'][0], 'r') as hf:
        tmp_ir069 = hf['ir069'][tmpCat_ir069['file_index'][0]]
    with h5py.File(dirBase+tmpCat_ir107['file_name'][0], 'r') as hf:
        tmp_ir107 = hf['ir107'][tmpCat_ir107['file_index'][0]]
    with h5py.File(dirBase+tmpCat_lght['file_name'][0], 'r') as hf:
        tmp_lght = hf[tmpCat_lght['id'][0]]
        # Rasterise lightning records! 
        tmp_lght = np.int16(lght_to_grid(tmp_lght))
    with h5py.File(dirBase+tmpCat_vil['file_name'][0], 'r') as hf:
        tmp_vil = hf['vil'][tmpCat_vil['file_index'][0]]
    
    ## Make the relevant h5 file!
    with h5py.File(dirOut+dataMod+'/'+tmp_id+'.h5', 'w', libver='latest') as f:
        f.create_dataset('ir069', data=tmp_ir069, 
            compression='gzip', compression_opts=9)
        f.create_dataset('ir107', data=tmp_ir107, 
            compression='gzip', compression_opts=9)
        f.create_dataset('lght', data=tmp_lght, 
            compression='gzip', compression_opts=9)
        f.create_dataset('vil', data=tmp_vil, 
            compression='gzip', compression_opts=9)
        #
        f.attrs['name'] = tmp_id


