import os.path as osp
import os
from glob import glob
import time
from base import BaseDataLoader
import torch
from abc import abstractmethod
import shutil
from utils import pc_normalize, read_shape, extract_number_from_filename, get_operators, compute_hks_autoscale
class MyBaseDataLoader(BaseDataLoader):
    """
    MyBaseDataLoader, collect all the commonly used member variables and member functions
    """
    def __init__(self, 
                 dataset, 
                 batch_size, 
                 n_eig=100,
                 input_type = "xyz", 
                 descriptor=None, 
                 descriptor_dim=None, 
                 descriptor_dir=None,
                 shuffle=True, 
                 drop_last=True, 
                 validation_split=0.0, 
                 num_workers=1, 
                 base_input_dir="input/", 
                 training = True):
        
        self.dataset = dataset        
        self.n_eig = n_eig
        self.input_type = input_type        
        self.descriptor = descriptor
        self.descriptor_dim = descriptor_dim
        self.descriptor_dir = descriptor_dir
        self.split = self.dataset.split
        self.training = training

        super().__init__(dataset, 
                         batch_size, 
                         shuffle, 
                         drop_last, 
                         validation_split, 
                         num_workers, 
                         collate_fn=self._custom_collate_fn)

        self.input_dir = base_input_dir
        
        if not osp.exists(self.input_dir):
            print(f"input_dir {self.input_dir} not exists, start preprocess")
            self._preprocess()

        # Load all lists
        self.vertices_list = sorted(glob(f'{self.input_dir}/vertices/*.pt'))
        self.eVals_list = sorted(glob(f'{self.input_dir}/laplacian/eVals/*.pt'))
        self.eVecs_list  = sorted(glob(f'{self.input_dir}/laplacian/eVecs/*.pt'))
        self.Ls_list = sorted(glob(f'{self.input_dir}/laplacian/Ls/*.pt'))
        self.Ms_list = sorted(glob(f'{self.input_dir}/laplacian/Ms/*.pt'))
        self.desc_list = sorted(glob(f'{self.input_dir}/descriptor/{self.descriptor}/*.pt')) if self.descriptor else None
        
        # ---diffusionNets---
        self.diffusion_dir = osp.join(self.input_dir, "diffusion")
        self.gradX_list = sorted(glob(f'{self.diffusion_dir}/gradX/*.pt'))
        self.gradY_list = sorted(glob(f'{self.diffusion_dir}/gradY/*.pt'))

    @abstractmethod
    def _custom_collate_fn(self, batch):
        '''
        virtual function, should be implemented in the derived class
        '''
        pass

    def _preprocess(self):
        '''
        When no input presented, pre-calculate all needed laplacians and descriptors.
        '''

        print("Pre-calculating...")
        time_start = time.time()
        save_vertices_dir = osp.join(self.input_dir, "vertices")
        os.makedirs(save_vertices_dir, exist_ok=True)
        save_evals_dir = osp.join(self.input_dir, "laplacian/eVals")
        os.makedirs(save_evals_dir, exist_ok=True)
        save_evecs_dir = osp.join(self.input_dir, "laplacian/eVecs")
        os.makedirs(save_evecs_dir, exist_ok=True)
        save_Ls_dir = osp.join(self.input_dir, "laplacian/Ls")
        os.makedirs(save_Ls_dir, exist_ok=True)
        save_Ms_dir = osp.join(self.input_dir, "laplacian/Ms")
        os.makedirs(save_Ms_dir, exist_ok=True)
        if self.descriptor is not None: 
            save_desc_dir = osp.join(self.input_dir, "descriptor/", self.descriptor)
            os.makedirs(save_desc_dir, exist_ok=True)

        # ---diffusionNets---
        diffusion_dir = osp.join(self.input_dir, "diffusion")
        os.makedirs(diffusion_dir, exist_ok=True)
        save_gradX_dir = osp.join(diffusion_dir, "gradX")
        os.makedirs(save_gradX_dir, exist_ok=True)
        save_gradY_dir = osp.join(diffusion_dir, "gradY")
        os.makedirs(save_gradY_dir, exist_ok=True)


        for _, off_path in enumerate(self.dataset.off_list):
            
            print(f'{off_path}...', end='')
            idx = extract_number_from_filename(off_path)
            pcs, faces = read_shape(off_path)
            pcs = pc_normalize(pcs, faces)

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
            pcs_tensor = torch.tensor(pcs, dtype=torch.float32).to(device)
            faces_tensor = torch.tensor(faces, dtype=torch.long).to(device)
            # ---diffusionNets---
            frames, M, L, eVal, eVec, gradX, gradY = get_operators(pcs_tensor, None, k=self.n_eig, cache_dir=None)   
            
            eVal = eVal[1:]
            eVec = eVec[:, 1:]
            torch.save(pcs_tensor, osp.join(save_vertices_dir, f'vertices_{idx:03d}.pt')) # (n, 3)            
            torch.save(eVal.squeeze(), osp.join(save_evals_dir, f'eVal_{idx:03d}.pt')) # Save the eVal as a diagonal matrix (k,)
            torch.save(eVec, osp.join(save_evecs_dir, f'eVec_{idx:03d}.pt')) # (n, k)
            torch.save(L, osp.join(save_Ls_dir, f'L_{idx:03d}.pt')) # (n, n)            
            torch.save(M, osp.join(save_Ms_dir, f'M_{idx:03d}.pt')) # (n,)
            torch.save(gradX, osp.join(save_gradX_dir, f'gradX_{idx:03d}.pt')) # (n, n)
            torch.save(gradY, osp.join(save_gradY_dir, f'gradY_{idx:03d}.pt')) # (n, n)


            if self.descriptor is not None: 
                if self.descriptor == "hks":
                    hks = compute_hks_autoscale(eVal.unsqueeze(0), eVec.unsqueeze(0), self.descriptor_dim).squeeze()
                    torch.save(hks, osp.join(save_desc_dir, f'{self.descriptor}_{idx:03d}.pt'))
                elif self.descriptor == 'deep':
                    if os.path.isdir(self.descriptor_dir):
                        for file_name in os.listdir(self.descriptor_dir):
                            src_file_path = os.path.join(self.descriptor_dir, file_name)
                            dst_file_path = os.path.join(save_desc_dir, file_name)
                            shutil.copy(src_file_path, dst_file_path)              
                    else:                            
                        print(f"The specified path {self.descriptor_dir} is not a directory.")                                                               
                else:
                    raise ValueError(f"Invalid descriptor type: {self.descriptor}")
            print('DONE')
        print("Pre-calculating finished. Time consumed in second: ", time.time() - time_start)

    def pad_data(self, data, max_length, padding_value=0):

        if not self.training:
            return data
        
        current_length = data.shape[0]
        
        # eVals, Ms
        if len(data.shape) == 1:
            if current_length < max_length:
                pad_size = max_length - current_length
                data = torch.cat([data, torch.full((pad_size,), padding_value, dtype=data.dtype,device=data.device)], dim=0)
        
        elif len(data.shape) == 2:
            if data.shape[1] != data.shape[0]:  # eVecs, descriptors, vertices
                pad_size = max_length - current_length
                data = torch.cat([data, torch.full((pad_size, data.shape[1]), padding_value, dtype=data.dtype,device=data.device)], dim=0)
            else:  # Ls, gradXs, gradYs
                pad_size = max_length - current_length
                data = torch.cat([data, torch.full((pad_size, data.shape[1]), padding_value, dtype=data.dtype,device=data.device)], dim=0)
                data = torch.cat([data, torch.full((max_length, pad_size), padding_value, dtype=data.dtype,device=data.device)], dim=1)
        
        return data
class FAUST_DataLoader(MyBaseDataLoader):
    """
    DataLoader for the SCAPE_Dataset
    """
    def __init__(self, 
                 dataset, 
                 batch_size, 
                 n_eig = 100,
                 input_type = "xyz", 
                 descriptor=None, 
                 descriptor_dim=None, 
                 descriptor_dir=None,                 
                 shuffle=True, 
                 drop_last=True, 
                 validation_split=0.0, 
                 num_workers=1, 
                 base_input_dir="input/", 
                 training=True):
        super().__init__(dataset, 
                         batch_size, 
                         n_eig,
                         input_type,
                         descriptor, 
                         descriptor_dim,
                         descriptor_dir, 
                         shuffle, 
                         drop_last, 
                         validation_split, 
                         num_workers, 
                         base_input_dir, 
                         training)


    def _custom_collate_fn(self, batch):
        """
        vertices (n,3)
        eVals (k,)
        eVecs (n,k)
        Ls (n,n)
        Ms (n,)
        descriptors (n,descriptor_dim)
        gradXs (n,n)
        gradYs (n,n)
        """
        
        max_length = 0

        # caculate the max length from whole batch
        for data in batch:
            for pcs_path in data:
                idx = extract_number_from_filename(pcs_path)
                vertices_length = torch.load(self.vertices_list[idx]).shape[0]
                if vertices_length > max_length:
                    max_length = vertices_length
        
        vertices = [] 
        eVals = []
        eVecs = []
        Ls = []
        Ms = []
        descriptors = []
        gradXs = []
        gradYs = []

        for data in batch:
            
            file_names = data
            vertices_collection = []
            eVals_collection = []
            eVecs_collection = []
            Ls_collection = []
            Ms_collection = []
            descriptors_collection = []
            gradX_collection = []
            gradY_collection = []

            for pcs_path in data:
                idx = extract_number_from_filename(pcs_path)
                vertices_collection.append(self.pad_data(torch.load(self.vertices_list[idx]), max_length))
                eVals_collection.append(torch.load(self.eVals_list[idx]))
                eVecs_collection.append(self.pad_data(torch.load(self.eVecs_list[idx]), max_length))
                
                L = torch.load(self.Ls_list[idx])
                if L.is_sparse:
                    L = L.to_dense() if L.is_sparse else L

                L = self.pad_data(L, max_length)
                Ls_collection.append(L)
                            
                Ms_collection.append(self.pad_data(torch.load(self.Ms_list[idx]), max_length))

                descriptors_collection.append(self.pad_data(torch.load(self.desc_list[idx]), max_length))

                # ---diffusionNets---

                gradX = torch.load(self.gradX_list[idx])
                if gradX.is_sparse:
                    gradX = gradX.to_dense() if gradX.is_sparse else gradX
                gradX = self.pad_data(gradX, max_length)
                gradX_collection.append(gradX)           

                gradY = torch.load(self.gradY_list[idx])
                if gradY.is_sparse:
                    gradY = gradY.to_dense() if gradY.is_sparse else gradY
                gradY = self.pad_data(gradY, max_length)
                gradY_collection.append(gradY)

                
            vertices.append(torch.stack(vertices_collection))
            eVals.append(torch.stack(eVals_collection))
            eVecs.append(torch.stack(eVecs_collection))
            Ls.append(torch.stack(Ls_collection))
            Ms.append(torch.stack(Ms_collection))
            descriptors.append(torch.stack(descriptors_collection))

            # ---diffusionNets---
            gradXs.append(torch.stack(gradX_collection))
            gradYs.append(torch.stack(gradY_collection))
            
        return file_names, torch.stack(vertices), torch.stack(eVals), torch.stack(eVecs), torch.stack(Ls), torch.stack(Ms), torch.stack(descriptors), torch.stack(gradXs), torch.stack(gradYs)

class SCAPE_DataLoader(MyBaseDataLoader):
    """
    DataLoader for the SCAPE_Dataset
    """
    def __init__(self, 
                 dataset, 
                 batch_size, 
                 n_eig = 100,
                 input_type = "xyz", 
                 descriptor=None, 
                 descriptor_dim=None, 
                 descriptor_dir=None,                 
                 shuffle=True, 
                 drop_last=True, 
                 validation_split=0.0, 
                 num_workers=1, 
                 base_input_dir="input/", 
                 training=True):
        super().__init__(dataset, 
                         batch_size, 
                         n_eig,
                         input_type,
                         descriptor, 
                         descriptor_dim,
                         descriptor_dir, 
                         shuffle, 
                         drop_last, 
                         validation_split, 
                         num_workers, 
                         base_input_dir, 
                         training)


    def _custom_collate_fn(self, batch):
        """
        vertices (n,3)
        eVals (k,)
        eVecs (n,k)
        Ls (n,n)
        Ms (n,)
        descriptors (n,descriptor_dim)
        gradXs (n,n)
        gradYs (n,n)
        """
        
        max_length = 0

        # caculate the max length from whole batch
        for data in batch:
            for pcs_path in data:
                idx = extract_number_from_filename(pcs_path)
                idx = idx - 1 if idx >= 52 else idx # since SCAPE do not have mesh051.off
                # idx = idx - 52 # test with four shapes for verification
                vertices_length = torch.load(self.vertices_list[idx]).shape[0]
                if vertices_length > max_length:
                    max_length = vertices_length
        
        vertices = [] 
        eVals = []
        eVecs = []
        Ls = []
        Ms = []
        descriptors = []
        gradXs = []
        gradYs = []

        for data in batch:
            
            file_names = data
            vertices_collection = []
            eVals_collection = []
            eVecs_collection = []
            Ls_collection = []
            Ms_collection = []
            descriptors_collection = []
            gradX_collection = []
            gradY_collection = []

            for pcs_path in data:
                idx = extract_number_from_filename(pcs_path)
                idx = idx - 1 if idx >= 52 else idx # since SCAPE do not have mesh051.off
                # idx = idx - 52 # test with four shapes for verification
                vertices_collection.append(self.pad_data(torch.load(self.vertices_list[idx]), max_length))
                eVals_collection.append(torch.load(self.eVals_list[idx]))
                eVecs_collection.append(self.pad_data(torch.load(self.eVecs_list[idx]), max_length))
                
                L = torch.load(self.Ls_list[idx])
                if L.is_sparse:
                    L = L.to_dense() if L.is_sparse else L

                L = self.pad_data(L, max_length)
                Ls_collection.append(L)
                            
                Ms_collection.append(self.pad_data(torch.load(self.Ms_list[idx]), max_length))

                descriptors_collection.append(self.pad_data(torch.load(self.desc_list[idx]), max_length))

                # ---diffusionNets---

                gradX = torch.load(self.gradX_list[idx])
                if gradX.is_sparse:
                    gradX = gradX.to_dense() if gradX.is_sparse else gradX
                gradX = self.pad_data(gradX, max_length)
                gradX_collection.append(gradX)           

                gradY = torch.load(self.gradY_list[idx])
                if gradY.is_sparse:
                    gradY = gradY.to_dense() if gradY.is_sparse else gradY
                gradY = self.pad_data(gradY, max_length)
                gradY_collection.append(gradY)

                
            vertices.append(torch.stack(vertices_collection))
            eVals.append(torch.stack(eVals_collection))
            eVecs.append(torch.stack(eVecs_collection))
            Ls.append(torch.stack(Ls_collection))
            Ms.append(torch.stack(Ms_collection))
            descriptors.append(torch.stack(descriptors_collection))

            # ---diffusionNets---
            gradXs.append(torch.stack(gradX_collection))
            gradYs.append(torch.stack(gradY_collection))
            
        return file_names, torch.stack(vertices), torch.stack(eVals), torch.stack(eVecs), torch.stack(Ls), torch.stack(Ms), torch.stack(descriptors), torch.stack(gradXs), torch.stack(gradYs)


class SURREAL_DataLoader(MyBaseDataLoader):
    """
    DataLoader for the SURREAL_Dataset
    """
    def __init__(self, 
                 dataset, 
                 batch_size, 
                 n_eig = 100,
                 input_type = "xyz", 
                 descriptor=None, 
                 descriptor_dim=None, 
                 descriptor_dir=None,                 
                 shuffle=True, 
                 drop_last=True, 
                 validation_split=0.0, 
                 num_workers=1, 
                 base_input_dir="input/", 
                 training=True):
        super().__init__(dataset, 
                         batch_size, 
                         n_eig,
                         input_type,
                         descriptor, 
                         descriptor_dim,
                         descriptor_dir, 
                         shuffle, 
                         drop_last, 
                         validation_split, 
                         num_workers, 
                         base_input_dir, 
                         training)


    def _custom_collate_fn(self, batch):
        """
        vertices (n,3)
        eVals (k,)
        eVecs (n,k)
        Ls (n,n)
        Ms (n,)
        descriptors (n,descriptor_dim)
        gradXs (n,n)
        gradYs (n,n)
        """
        
        vertices = [] 
        eVals = []
        eVecs = []
        Ls = []
        Ms = []
        descriptors = []
        gradXs = []
        gradYs = []

        for data in batch:
            
            file_names = data
            vertices_collection = []
            eVals_collection = []
            eVecs_collection = []
            Ls_collection = []
            Ms_collection = []
            descriptors_collection = []
            gradX_collection = []
            gradY_collection = []

            for pcs_path in data:
                print("load data from: ", pcs_path)
                idx = extract_number_from_filename(pcs_path)
                idx = idx - 1
                # example: /usr/stud/zehu/project/data/surreal5k/off/surreal_1335.off
                # idx is 1334      
                          
                vertices_collection.append(torch.load(self.vertices_list[idx]))
                eVals_collection.append(torch.load(self.eVals_list[idx]))
                eVecs_collection.append(torch.load(self.eVecs_list[idx]))
                
                L = torch.load(self.Ls_list[idx])
                if L.is_sparse:
                    L = L.to_dense() if L.is_sparse else L
                Ls_collection.append(L)
                            
                Ms_collection.append(torch.load(self.Ms_list[idx]))
                
                descriptors_collection.append(torch.load(self.desc_list[idx]))

                # ---diffusionNets---

                gradX = torch.load(self.gradX_list[idx])
                if gradX.is_sparse:
                    gradX = gradX.to_dense() if gradX.is_sparse else gradX
                gradX_collection.append(gradX)           

                gradY = torch.load(self.gradY_list[idx])
                if gradY.is_sparse:
                    gradY = gradY.to_dense() if gradY.is_sparse else gradY
                gradY_collection.append(gradY)

                
            vertices.append(torch.stack(vertices_collection))
            eVals.append(torch.stack(eVals_collection))
            eVecs.append(torch.stack(eVecs_collection))
            Ls.append(torch.stack(Ls_collection))
            Ms.append(torch.stack(Ms_collection))
            descriptors.append(torch.stack(descriptors_collection))

            # ---diffusionNets---
            gradXs.append(torch.stack(gradX_collection))
            gradYs.append(torch.stack(gradY_collection))
            
        return file_names, torch.stack(vertices), torch.stack(eVals), torch.stack(eVecs), torch.stack(Ls), torch.stack(Ms), torch.stack(descriptors), torch.stack(gradXs), torch.stack(gradYs)