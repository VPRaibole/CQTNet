import os
import torch
from torch.utils.data import Dataset
import numpy as np
import librosa
from torchvision import transforms
import random
import PIL
import torch.nn.functional as F

class CQT(Dataset):
    def __init__(self, mode='train', out_length=None):
        self.indir = '/content/projectData/youtube_hpcp_npy/'
        self.mode = mode
        if mode == 'train': 
            filepath = 'data/SHS100K-TRAIN_6'
        elif mode == 'val':
            filepath = 'data/SHS100K-VAL'
        elif mode == 'test': 
            filepath = 'data/SHS100K-TEST'
        elif mode == 'songs80': 
            self.indir = 'data/covers80_cqt_npy/'
            filepath = 'data/songs80_list.txt'
        
        with open(filepath, 'r') as fp:
            self.file_list = [line.rstrip() for line in fp]
        self.out_length = out_length

    def pad_or_truncate(self, data, target_length, target_freq=84):
        # Ensure data is 3D: (channels, frequency, time)
        if data.ndim == 2:
            data = data.unsqueeze(0)  # Add channel dimension
        
        _, freq, current_length = data.shape
        
        # Pad or truncate frequency dimension
        if freq > target_freq:
            data = data[:, :target_freq, :]
        elif freq < target_freq:
            pad_freq = target_freq - freq
            data = F.pad(data, (0, 0, 0, pad_freq), mode='constant', value=0)
        
        # Pad or truncate time dimension
        if current_length > target_length:
            data = data[:, :, :target_length]
        elif current_length < target_length:
            pad_time = target_length - current_length
            data = F.pad(data, (0, pad_time), mode='constant', value=0)
        
        return data

    def __getitem__(self, index):
        transform_train = transforms.Compose([
            lambda x: self.SpecAugment(x),
            lambda x: self.SpecAugment(x),
            lambda x: self.change_speed(x, 0.7, 1.3),
            lambda x: x.astype(np.float32) / (np.max(np.abs(x)) + 1e-6),
            lambda x: self.cut_data(x, self.out_length),
            lambda x: torch.Tensor(x),
            lambda x: x.unsqueeze(0),  # Add channel dimension
        ])
        
        transform_test = transforms.Compose([
            lambda x: x.astype(np.float32) / (np.max(np.abs(x)) + 1e-6),
            lambda x: self.cut_data_front(x, self.out_length),
            lambda x: torch.Tensor(x),
            lambda x: x.unsqueeze(0),  # Add channel dimension
        ])
        
        filename = self.file_list[index].strip()
        set_id, version_id = filename.split('.')[0].split('_')
        set_id, version_id = int(set_id), int(version_id)
        in_path = self.indir + filename + '.npy'
        data = np.load(in_path)

        if self.mode == 'train':
            data = transform_train(data)
        else:
            data = transform_test(data)

        # Pad or truncate to a fixed length and frequency
        data = self.pad_or_truncate(data, 400, 84)

        return data, int(set_id)

    def __len__(self):
        return len(self.file_list)

    def SpecAugment(self, data):
        F = 24
        f = np.random.randint(F)
        f0 = np.random.randint(84 - f)
        data[f0:f0 + f, :] *= 0
        return data

    def change_speed(self, data, l=0.7, r=1.3):
        new_len = int(data.shape[1] * np.random.uniform(l, r))
        maxx = np.max(data) + 1
        data0 = PIL.Image.fromarray((data * 255.0 / maxx).astype(np.uint8))
        transform = transforms.Compose([
            transforms.Resize(size=(84, new_len)),
        ])
        new_data = transform(data0)
        return np.array(new_data) / 255.0 * maxx

    def cut_data(self, data, out_length):
        if out_length is not None:
            if data.shape[1] > out_length:
                max_offset = data.shape[1] - out_length
                offset = np.random.randint(max_offset)
                data = data[:, offset:(out_length+offset)]
            else:
                offset = out_length - data.shape[1]
                data = np.pad(data, ((0, 0), (0, offset)), "constant")
        if data.shape[1] < 200:
            offset = 200 - data.shape[1]
            data = np.pad(data, ((0, 0), (0, offset)), "constant")
        return data

    def cut_data_front(self, data, out_length):
        if out_length is not None:
            if data.shape[1] > out_length:
                data = data[:, :out_length]
            else:
                offset = out_length - data.shape[1]
                data = np.pad(data, ((0, 0), (0, offset)), "constant")
        if data.shape[1] < 200:
            offset = 200 - data.shape[1]
            data = np.pad(data, ((0, 0), (0, offset)), "constant")
        return data

if __name__ == '__main__':
    train_dataset = CQT('train', 394)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, num_workers=12, shuffle=True)

##############
# import os,sys
# from torchvision import transforms
# import torch, torch.utils
# import numpy as np
# from torch.utils.data import DataLoader
# from torch.utils.data import Dataset
# import random
# import bisect
# import torchvision
# import PIL


# def pad_or_truncate(self, data, target_length):
#     if data.shape[1] > target_length:
#         return data[:, :target_length]
#     elif data.shape[1] < target_length:
#         pad_width = ((0, 0), (0, target_length - data.shape[1]))
#         return np.pad(data, pad_width, mode='constant')
#     else:
#         return data


# def cut_data(data, out_length):
#     if out_length is not None:
#         if data.shape[0] > out_length:
#             max_offset = data.shape[0] - out_length
#             offset = np.random.randint(max_offset)
#             data = data[offset:(out_length+offset),:]
#         else:
#             offset = out_length - data.shape[0]
#             data = np.pad(data, ((0,offset),(0,0)), "constant")
#     if data.shape[0] < 200:
#         offset = 200 - data.shape[0]
#         data = np.pad(data, ((0,offset),(0,0)), "constant")
#     return data

# def cut_data_front(data, out_length):
#     if out_length is not None:
#         if data.shape[0] > out_length:
#             max_offset = data.shape[0] - out_length
#             offset = 0
#             data = data[offset:(out_length+offset),:]
#         else:
#             offset = out_length - data.shape[0]
#             data = np.pad(data, ((0,offset),(0,0)), "constant")
#     if data.shape[0] < 200:
#         offset = 200 - data.shape[0]
#         data = np.pad(data, ((0,offset),(0,0)), "constant")
#     return data

# def shorter(feature, mean_size=2):
#     length, height  = feature.shape
#     new_f = np.zeros((int(length/mean_size),height),dtype=np.float64)
#     for i in range(int(length/mean_size)):
#         new_f[i,:] = feature[i*mean_size:(i+1)*mean_size,:].mean(axis=0)
#     return new_f

# def change_speed(data, l=0.7, r=1.5): # change data.shape[0]
#     new_len = int(data.shape[0]*np.random.uniform(l,r))
#     maxx = np.max(data)+1
#     data0 = PIL.Image.fromarray((data*255.0/maxx).astype(np.uint8))
#     transform = transforms.Compose([
#         transforms.Resize(size=(new_len,data.shape[1])), 
#     ])
#     new_data = transform(data0)
#     return np.array(new_data)/255.0*maxx

# def SpecAugment(data):
#     F = 24
#     f = np.random.randint(F)
#     f0 = np.random.randint(84-f)
#     data[f0:f0+f,:]*=0
#     return data

# class CQT(Dataset):
#     def __init__(self, mode='train', out_length=None):
#         self.indir = '/content/projectData/youtube_hpcp_npy/'
#         self.mode=mode
#         if mode == 'train': 
#             filepath='data/SHS100K-TRAIN_6'
#         elif mode == 'val':
#             filepath='data/SHS100K-VAL'
#         # elif mode == 'songs350': 
#         #     self.indir = 'data/you350_cqt_npy/'
#         #     filepath='data/you350_list.txt'
#         elif mode == 'test': 
#             filepath='data/SHS100K-TEST'
#         elif mode == 'songs80': 
#             self.indir = 'data/covers80_cqt_npy/'
#             filepath = 'data/songs80_list.txt'
#         # elif mode == 'Mazurkas':
#         #     self.indir = 'data/Mazurkas_cqt_npy/'
#         #     filepath = 'data/Mazurkas_list.txt'
#         with open(filepath, 'r') as fp:
#             self.file_list = [line.rstrip() for line in fp]
#         self.out_length = out_length
#     def __getitem__(self, index):
#         transform_train = transforms.Compose([
#             lambda x: SpecAugment(x), #SpecAugment 频谱增强一次
#             lambda x: SpecAugment(x), #SpecAugment 频谱增强 x 2
#             lambda x : x.T,
#             lambda x : change_speed(x, 0.7, 1.3), # 速度随机变化
#             lambda x : x.astype(np.float32) / (np.max(np.abs(x))+ 1e-6),
#             lambda x : cut_data(x, self.out_length),
#             lambda x : torch.Tensor(x),
#             lambda x : x.permute(1,0).unsqueeze(0),
#         ])
#         transform_test = transforms.Compose([
#             lambda x : x.T,
#             #lambda x : x-np.mean(x),
#             lambda x : x.astype(np.float32) / (np.max(np.abs(x))+ 1e-6),
#             lambda x : cut_data_front(x, self.out_length),
#             lambda x : torch.Tensor(x),
#             lambda x : x.permute(1,0).unsqueeze(0),
#         ])
#         filename = self.file_list[index].strip()
#         set_id, version_id = filename.split('.')[0].split('_')
#         set_id, version_id = int(set_id), int(version_id)
#         in_path = self.indir+filename+'.npy'
#         data = np.load(in_path) # from 12xN to Nx12

#         if self.mode == 'train':
#             data = transform_train(data)
#         else:
#             data = transform_test(data)
#         data = self.pad_or_truncate(data, 400) ################################################################
#         return data, int(set_id)
#     def __len__(self):
#         return len(self.file_list)

    
# if __name__=='__main__':
#     train_dataset = HPCP('train', 394)
#     trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, num_workers=12, shuffle=True)
