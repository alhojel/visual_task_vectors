"""Based on https://github.com/Seokju-Cho/Volumetric-Aggregation-Transformer/blob/main/data/pascal.py
"""
import os
from PIL import Image
from scipy.io import loadmat
import numpy as np
import torch
from torch.utils.data import Dataset
from mae_utils import PURPLE, YELLOW
import json
import random

random.seed(10)

class DatasetPASCAL(Dataset):
    def __init__(self, datapath, fold, image_transform, mask_transform, padding: bool = 1, use_original_imgsize: bool = False, flipped_order: bool = False,
                 reverse_support_and_query: bool = False, random: bool = False, ensemble: bool = False, purple: bool = False, query_support_list_file=None, iters=1000, avoid_list=None, type="val", task = 0):
        self.fold = fold
        self.nfolds = 4
        self.flipped_order = flipped_order
        self.nclass = 20
        self.padding = padding
        self.random = random
        self.ensemble = ensemble
        self.purple = purple
        self.use_original_imgsize = use_original_imgsize
        self.type = type
        
        self.iters = iters

        self.task = task

        self.img_path = os.path.join(datapath, 'images/')
        self.image_transform = image_transform
        self.reverse_support_and_query = reverse_support_and_query
        self.mask_transform = mask_transform

        # Load all image metadata
        self.img_metadata = [f for f in os.listdir(self.img_path) if os.path.isfile(os.path.join(self.img_path, f)) and f.endswith('.png')]


    def __len__(self):
        return self.iters

    def __getitem__(self, idx):
        
        idx %= len(self.img_metadata)  # for testing, as n_images < 1000
        query_name, support_name, class_sample_query, class_sample_support = self.sample_episode(idx)
            

        query_img, support_img, org_qry_imsize = self.load_frame(query_name, support_name)

        #Here do the three tasks
        if self.task is None:
            grid = []
            grid.append(self.lowlight_grid(query_img, support_img))
            grid.append(self.neutral_copy_grid(query_img, support_img))
            grid.append(self.inpaint_black_grid(query_img, support_img, r=1))
            grid.append(self.colorization_grid(query_img, support_img))
        if self.task==0:
            grid=None
        if self.task==1:
            grid=self.lowlight_grid(query_img, support_img)
        if self.task==2:
            grid=self.neutral_copy_grid(query_img, support_img)
        if self.task==3:
            grid=self.inpaint_black_grid(query_img, support_img, r=1)
        if self.task==4:
            grid=self.colorization_grid(query_img, support_img)

        
        batch = {'query_name': query_name,
                 'support_name': support_name,
                 'grid': grid}

        return batch


    def load_frame(self, query_name, support_name):
        query_img = self.read_img(query_name)
        support_img = self.read_img(support_name)
        org_qry_imsize = query_img.size

        return query_img, support_img, org_qry_imsize

    def read_img(self, img_name):
        r"""Return RGB image in PIL Image"""
        return Image.open(os.path.join(self.img_path, img_name))

    def sample_episode(self, idx):
        if len(self.img_metadata) < 2:
            raise ValueError("Not enough images to form an episode.")

        # Randomly select two different images
        query_idx = random.randint(0, len(self.img_metadata) - 1)
        support_idx = query_idx
        while support_idx == query_idx:
            support_idx = random.randint(0, len(self.img_metadata) - 1)

        query_name = self.img_metadata[query_idx]
        support_name = self.img_metadata[support_idx]

        # Placeholder for class samples, not used in this context
        class_sample_query = None
        class_sample_support = None

        return query_name, support_name, class_sample_query, class_sample_support
    
    
    def neutral_copy_grid(self, query_img, support_img):
        
        query_img = self.image_transform(query_img)
        support_img = self.image_transform(support_img)
        grid = self.create_grid_from_images_colorization(support_img, support_img, query_img, query_img)
        
        return grid
    
    def bw_grid(self, query_img, support_img):
        
        query_mask, query_img = self.mask_transform[1](query_img), self.image_transform(query_img)
        support_mask, support_img = self.mask_transform[1](support_img), self.image_transform(support_img)
        grid = self.create_grid_from_images_colorization(support_img, support_mask, query_img, query_mask)
        
        return grid
    
    def inpaint_black_grid(self, query_img, support_img, r=0):
        
        query_img = self.image_transform(query_img)
        support_img = self.image_transform(support_img)

        query_mask = query_img.clone()
        support_mask = support_img.clone()

        _, h, w = query_mask.shape
        square_size = min(h, w) // 4
        if r == 0:
            h_start2 = h_start = (h - square_size) // 2
            w_start2 = w_start = (w - square_size) // 2
        if r == 1:
            h_start2 = h_start = random.randint(0, h - square_size)
            w_start2 = w_start = random.randint(0, w - square_size)
        if r == 2:
            h_start = random.randint(0, h - square_size)
            w_start = random.randint(0, w - square_size)
            h_start2 = random.randint(0, h - square_size)
            w_start2 = random.randint(0, w - square_size)
        # Set the square to black for all channels
        query_mask[:, h_start:h_start+square_size, w_start:w_start+square_size] = 0
        support_mask[:, h_start2:h_start2+square_size, w_start2:w_start2+square_size] = 0
        
        grid = self.create_grid_from_images_colorization(support_mask, support_img, query_mask, query_img)
        
        return grid

    def lowlight_grid(self, query_img, support_img):
        
        query_img = self.image_transform(query_img)
        support_img = self.image_transform(support_img)
        grid = self.create_grid_from_images_colorization(0.5*support_img, support_img, 0.5*query_img, query_img)
        
        return grid

    def create_grid_from_images_colorization(self, support_img, support_mask, query_img, query_mask):
        
        if support_img.shape[0] != 1:
            support_img = support_img[0]
        if support_mask.shape[0] != 1:
            support_mask = support_mask[0]
        if query_img.shape[0] != 1:
            query_img = query_img[0]
        if query_mask.shape[0] != 1:
            query_mask = query_mask[0]


        support_img = support_img.unsqueeze(0) if support_img.dim() == 2 else support_img
        support_mask = support_mask.unsqueeze(0) if support_mask.dim() == 2 else support_mask
        query_img = query_img.unsqueeze(0) if query_img.dim() == 2 else query_img
        query_mask = query_mask.unsqueeze(0) if query_mask.dim() == 2 else query_mask
        
        if self.reverse_support_and_query:
            support_img, support_mask, query_img, query_mask = query_img, query_mask, support_img, support_mask
        canvas = torch.ones((support_img.shape[0], 2 * support_img.shape[1] + 2 * self.padding,
                             2 * support_img.shape[2] + 2 * self.padding))
        canvas[:, :support_img.shape[1], :support_img.shape[2]] = support_img
        if self.flipped_order:
            canvas[:, :support_img.shape[1], -support_img.shape[2]:] = query_img
            canvas[:, -query_img.shape[1]:, -support_img.shape[2]:] = query_mask
            canvas[:, -query_img.shape[1]:, :query_img.shape[2]] = support_mask
        else:
            canvas[:, -query_img.shape[1]:, :query_img.shape[2]] = query_img
            canvas[:, :support_img.shape[1], -support_img.shape[2]:] = support_mask
            canvas[:, -query_img.shape[1]:, -support_img.shape[2]:] = query_mask

        return canvas
    
    def neutral_copy_grid(self, query_img, support_img):
        
        query_img = self.image_transform(query_img)
        support_img = self.image_transform(support_img)
        grid = self.create_grid_from_images_colorization(support_img, support_img, query_img, query_img)
        
        return grid
    
    def bw_grid(self, query_img, support_img):
        
        query_mask, query_img = self.mask_transform[1](query_img), self.image_transform(query_img)
        support_mask, support_img = self.mask_transform[1](support_img), self.image_transform(support_img)
        grid = self.create_grid_from_images_colorization(support_img, support_mask, query_img, query_mask)
        
        return grid