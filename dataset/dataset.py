from glob import glob
import os.path as osp
import os
from itertools import combinations
from torch.utils.data import Dataset
import re

def sort_list(l):
    try:
        return list(sorted(l, key=lambda x: int(re.search(r'\d+(?=\.)', x).group())))
    except AttributeError:
        return sorted(l)
    
class FAUST_Dataset(Dataset):
    def __init__(self, data_dir, split='train'):
        """
        Args:
            data_dir: The relative directory that contains all objects.
            src_name: The name of the dataset source, e.g., tosca_off
            obj: Indicate on which object this dataset should collect the different shapes. 
            collection_size: Indicate the size of the collection as each sample.
        """
        self.off_dir = osp.abspath(osp.join(data_dir, 'off'))
        self.off_list = list(sorted(glob(f'{self.off_dir}/*.off')))
        self.vts_dir = osp.abspath(osp.join(data_dir, 'corres'))
        self.vts_file_list = list(sorted(glob(f'{self.vts_dir}/*.vts')))        
        self.vts_file_list = [file for file in self.vts_file_list if osp.splitext(osp.basename(file))[0] in [osp.splitext(osp.basename(off_file))[0] for off_file in self.off_list]]    
        print(f'Found {len(self.off_list)} .off files in {self.off_dir}.')
        print(f'Found {len(self.vts_file_list)} .vts files in {self.vts_dir}.')
        self.split = split
       
        if self.split == 'train':
            self.data = self.off_list[0:80]  
            index = range(len(self.data))
            index_combinations = list(combinations(index, r=2))
            self.collection_list = [(self.data[i], self.data[j]) for i, j in index_combinations]     
            print(f'training with {len(self.collection_list)} combinations of 50 shape')       

        elif self.split == 'valid':
            self.data = self.off_list[80:]  
            index = range(len(self.data))
            index_combinations = list(combinations(index, r=2))
            self.collection_list = [(self.data[i], self.data[j]) for i, j in index_combinations]   
            print(f'validation with {len(self.collection_list)} combinations of 20 shape')

        elif self.split == 'test':
            self.data = self.off_list[80:] 
            self.collection_list = [(self.data[i],) for i in range(0, len(self.data))]
            print(f'testing with {len(self.collection_list)} shape')

    def __len__(self):
        """
        Return the length of the self.shape_collections_list. It is defined as the number of shape collections on the same object.
        """
        return len(self.collection_list)

    def __getitem__(self, idx):
        """
        Return the indices of the idx-th collection in the self.shape_collections_list.

        Args:
            idx: Indicating the index of the shape collection in the list.

        Return:
            tuple of a shape indices collection
        """        
        # print(f'the {idx}-th shape pair = {self.collection_list[idx]}')
        return self.collection_list[idx]
    

class SCAPE_Dataset(Dataset):
    def __init__(self, data_dir, split='train'):
        """
        Args:
            data_dir: The relative directory that contains all objects.
            src_name: The name of the dataset source, e.g., tosca_off
            obj: Indicate on which object this dataset should collect the different shapes. 
            collection_size: Indicate the size of the collection as each sample.
        """
        self.off_dir = osp.abspath(osp.join(data_dir, 'off'))
        self.off_list = list(sorted(glob(f'{self.off_dir}/*.off')))
        self.vts_dir = osp.abspath(osp.join(data_dir, 'corres'))
        self.vts_file_list = list(sorted(glob(f'{self.vts_dir}/*.vts')))        
        # ignore all vts files that are not in the off_file_list
        self.vts_file_list = [file for file in self.vts_file_list if osp.splitext(osp.basename(file))[0] in [osp.splitext(osp.basename(off_file))[0] for off_file in self.off_list]]    
        print(f'Found {len(self.off_list)} .off files in {self.off_dir}.')
        print(f'Found {len(self.vts_file_list)} .vts files in {self.vts_dir}.')
        self.split = split
       
        if self.split == 'train':
            self.data = self.off_list[0:50]
            index = range(len(self.data))
            index_combinations = list(combinations(index, r=2))
            self.collection_list = [(self.data[i], self.data[j]) for i, j in index_combinations]     
            print(f'training with {len(self.collection_list)} combinations of 50 shape')       

        elif self.split == 'valid':
            self.data = self.off_list[51:]     
            index = range(len(self.data))
            index_combinations = list(combinations(index, r=2))
            self.collection_list = [(self.data[i], self.data[j]) for i, j in index_combinations]   
            print(f'validation with {len(self.collection_list)} combinations of 20 shape')

        elif self.split == 'test':
            self.data = self.off_list[51:]     
            self.collection_list = [(self.data[i],) for i in range(0, len(self.data))]
            print(f'testing with {len(self.collection_list)} shape')

    def __len__(self):
        """
        Return the length of the self.shape_collections_list. It is defined as the number of shape collections on the same object.
        """
        return len(self.collection_list)

    def __getitem__(self, idx):
        """
        Return the indices of the idx-th collection in the self.shape_collections_list.

        Args:
            idx: Indicating the index of the shape collection in the list.

        Return:
            tuple of a shape indices collection
        """        
        # print(f'the {idx}-th shape pair = {self.collection_list[idx]}')
        return self.collection_list[idx]
    
class SURREAL_Dataset(Dataset):
    def __init__(self, data_dir, split='train'):
        """
        Args:
            data_dir: The relative directory that contains all objects.
            src_name: The name of the dataset source, e.g., tosca_off
            obj: Indicate on which object this dataset should collect the different shapes. 
            collection_size: Indicate the size of the collection as each sample.
        """
        self.off_dir = osp.abspath(osp.join(data_dir, 'off'))
        self.off_list = list(sorted(glob(f'{self.off_dir}/*.off')))
        print(f'Found {len(self.off_list)} .off files in {self.off_dir}.')
        self.split = split
       
        if self.split == 'train':
            self.data = self.off_list
            index = range(len(self.data))
            index_combinations = list(combinations(index, r=2))
            self.collection_list = [(self.data[i], self.data[j]) for i, j in index_combinations]     
            print(f'training with {len(self.collection_list)} combinations of 5000 shape')
            index_combinations.clear()
        else:
            self.collection_list = []
            print(f'surreal do not need validation or test set')
    def __len__(self):
        """
        Return the length of the self.shape_collections_list. It is defined as the number of shape collections on the same object.
        """
        return len(self.collection_list)

    def __getitem__(self, idx):
        """
        Return the indices of the idx-th collection in the self.shape_collections_list.

        Args:
            idx: Indicating the index of the shape collection in the list.

        Return:
            tuple of a shape indices collection
        """        
        # print(f'the {idx}-th shape pair = {self.collection_list[idx]}')
        return self.collection_list[idx]