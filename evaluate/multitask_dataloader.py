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

class DatasetPASCAL(Dataset):
    def __init__(self, datapath, fold, image_transform, mask_transform, padding: bool = 1, use_original_imgsize: bool = False, flipped_order: bool = False,
                 reverse_support_and_query: bool = False, random: bool = False, ensemble: bool = False, purple: bool = False, query_support_list_file=None, iters=1000, avoid_list=None):
        self.fold = fold
        self.nfolds = 4
        self.flipped_order = flipped_order
        self.nclass = 20
        self.padding = padding
        self.random = random
        self.ensemble = ensemble
        self.purple = purple
        self.use_original_imgsize = use_original_imgsize

        self.iters = iters

        self.query_support_list_file = query_support_list_file
        self.avoid_list = avoid_list
        if query_support_list_file is not None:
            with open(query_support_list_file, 'r') as file:
                self.query_support_pairs = json.load(file)

        if avoid_list is not None:
            with open(avoid_list, 'r') as file:
                self.avoid_pairs = json.load(file)

        self.img_path = os.path.join(datapath, 'VOCdevkit/VOC2012/JPEGImages/')
        self.ann_path = os.path.join(datapath, 'VOCdevkit/VOC2012/SegmentationClassAug/')
        self.image_transform = image_transform
        self.reverse_support_and_query = reverse_support_and_query
        self.mask_transform = mask_transform

        self.class_ids = self.build_class_ids()
        self.img_metadata = self.build_img_metadata()
        self.img_metadata_classwise = self.build_img_metadata_classwise()

    def __len__(self):
        return self.iters if self.query_support_list_file is None else len(self.query_support_pairs)

    def __getitem__(self, idx):
        
        if self.query_support_list_file is not None:
            # Use query and support names from the JSON file
            pair = self.query_support_pairs[idx % len(self.query_support_pairs)]
            query_name = pair['query_name']
            support_name = pair['support_name']

            for ele in self.img_metadata:
                if ele[0] == query_name:
                    class_sample_query = class_sample_support = ele[1]
        else:
            # Original random sampling method
            idx %= len(self.img_metadata)  # for testing, as n_images < 1000
            query_name, support_name, class_sample_query, class_sample_support = self.sample_episode(idx)
            

        query_img, query_cmask, support_img, support_cmask, org_qry_imsize = self.load_frame(query_name, support_name)

        #Here do the three tasks
        grid=[]

        grid.append(self.segmentation_grid(query_img, query_cmask, support_img, support_cmask, class_sample_query, class_sample_support))
        grid.append(self.colorization_grid(query_img, support_img))
        #grid.append(self.neutral_copy_grid(query_img, support_img))
        grid.append(self.bw_grid(query_img, support_img))
        grid.append(self.lowlight_grid(query_img, support_img))
        grid.append(self.inpaint_black_grid(query_img, support_img, r=1))
        grid.append(self.inpaint_black_grid(query_img, support_img, r=2))
        
        batch = {'query_name': query_name,
                 'support_name': support_name,
                 'grid': grid}

        return batch

    def extract_ignore_idx(self, mask, class_id, purple):
        mask = np.array(mask)
        boundary = np.floor(mask / 255.)
        if not purple:
            mask[mask != class_id + 1] = 0
            mask[mask == class_id + 1] = 255
            return Image.fromarray(mask), boundary
        color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        for x in range(mask.shape[0]):
            for y in range(mask.shape[1]):
                if mask[x,y] != class_id + 1:
                    color_mask[x, y] = np.array(PURPLE)
                else:
                    color_mask[x, y] = np.array(YELLOW)
        return Image.fromarray(color_mask), boundary


    def load_frame(self, query_name, support_name):
        query_img = self.read_img(query_name)
        query_mask = self.read_mask(query_name)
        support_img = self.read_img(support_name)
        support_mask = self.read_mask(support_name)
        org_qry_imsize = query_img.size

        return query_img, query_mask, support_img, support_mask, org_qry_imsize

    def read_mask(self, img_name):
        r"""Return segmentation mask in PIL Image"""
        mask = Image.open(os.path.join(self.ann_path, img_name) + '.png')
        return mask

    def read_img(self, img_name):
        r"""Return RGB image in PIL Image"""
        return Image.open(os.path.join(self.img_path, img_name) + '.jpg')

    def sample_episode(self, idx):
        """Returns the index of the query, support and class."""
        query_name, class_sample = self.img_metadata[idx]
        if not self.random:
            support_class = class_sample
        else: 
            support_class = np.random.choice([k for k in self.img_metadata_classwise.keys() if self.img_metadata_classwise[k]], 1, replace=False)[0]
        while True:  # keep sampling support set if query == support
            support_name = np.random.choice(self.img_metadata_classwise[support_class], 1, replace=False)[0]
            if query_name != support_name: 
                break
        return query_name, support_name, class_sample, support_class

    def build_class_ids(self):
        nclass_trn = self.nclass // self.nfolds
        class_ids_val = [self.fold * nclass_trn + i for i in range(nclass_trn)]
        return class_ids_val

    def build_img_metadata(self):

        def read_metadata(split, fold_id):
            cwd = os.path.dirname(os.path.abspath(__file__))
            fold_n_metadata = os.path.join(cwd, 'splits/pascal/%s/fold%d.txt' % (split, fold_id))
            with open(fold_n_metadata, 'r') as f:
                fold_n_metadata = f.read().split('\n')[:-1]
            fold_n_metadata = [[data.split('__')[0], int(data.split('__')[1]) - 1] for data in fold_n_metadata]
            return fold_n_metadata


        img_metadata = []

        if self.query_support_list_file is not None:
            img_metadata = read_metadata('val', 0) + read_metadata('val',1) + read_metadata('val', 2) + read_metadata('val', 3)
        else:
            img_metadata = read_metadata('val', self.fold)

            if self.avoid_list is not None:
                print("Before Filter", len(img_metadata))
                img_metadata = [ele for ele in img_metadata if ele[0] not in [item["query_name"] for item in self.avoid_pairs]]
                img_metadata = [ele for ele in img_metadata if ele[0] not in [item["support_name"] for item in self.avoid_pairs]]
                print("After Filter", len(img_metadata))
                
        
        print('Total (val) images are : %d' % len(img_metadata))

        return img_metadata

    def build_img_metadata_classwise(self):
        img_metadata_classwise = {}
        for class_id in range(self.nclass):
            img_metadata_classwise[class_id] = []

        for img_name, img_class in self.img_metadata:
            img_metadata_classwise[img_class] += [img_name]
        return img_metadata_classwise
    
    def segmentation_grid(self, query_img, query_cmask, support_img, support_cmask, class_sample_query, class_sample_support):
        if self.image_transform:
            query_img = self.image_transform(query_img)
        query_mask, query_ignore_idx = self.extract_ignore_idx(query_cmask, class_sample_query, purple=self.purple)
        if self.mask_transform:
            query_mask = self.mask_transform[0](query_mask)
                
        if self.image_transform:
            support_img = self.image_transform(support_img)
        support_mask, support_ignore_idx = self.extract_ignore_idx(support_cmask, class_sample_support, purple=self.purple)
        if self.mask_transform:
            support_mask = self.mask_transform[0](support_mask)
        grid = self.create_grid_from_images_segmentation(support_img, support_mask, query_img, query_mask, flip=self.flipped_order)
        if self.ensemble:
            grid2 = self.create_grid_from_images_segmentation(support_img, support_mask, query_img, query_mask, (not self.flipped_order))


            support_purple_mask, _ = self.extract_ignore_idx(support_cmask, class_sample_support,
                                                                    purple=True)
            if self.mask_transform:
                support_purple_mask = self.mask_transform[0](support_purple_mask)

            grid3 = self.create_grid_from_images_segmentation(support_img, support_purple_mask, query_img, query_mask,
                                                flip=self.flipped_order)

            grid4 = self.create_grid_from_images_segmentation(support_img, support_purple_mask, query_img, query_mask,
                                                flip=(not self.flipped_order))

            grid = grid, grid2, grid3, grid4
        return grid
    
    def create_grid_from_images_segmentation(self, support_img, support_mask, query_img, query_mask, flip: bool = False):
        if self.reverse_support_and_query:
            support_img, support_mask, query_img, query_mask = query_img, query_mask, support_img, support_mask
        canvas = torch.ones((support_img.shape[0], 2 * support_img.shape[1] + 2 * self.padding, 2 * support_img.shape[2] + 2 * self.padding))
        canvas[:, :support_img.shape[1], :support_img.shape[2]] = support_img
        if flip:
            canvas[:, :support_img.shape[1], -support_img.shape[2]:] = query_img
            canvas[:, -query_img.shape[1]:, -support_img.shape[2]:] = query_mask
            canvas[:, -query_img.shape[1]:, :query_img.shape[2]] = support_mask
        else:
            canvas[:, -query_img.shape[1]:, :query_img.shape[2]] = query_img
            canvas[:, :support_img.shape[1], -support_img.shape[2]:] = support_mask
            canvas[:, -query_img.shape[1]:, -support_img.shape[2]:] = query_mask

        return canvas
    
    def colorization_grid(self, query_img, support_img):
        
        query_img, query_mask = self.mask_transform[1](query_img), self.image_transform(query_img)
        support_img, support_mask = self.mask_transform[1](support_img), self.image_transform(support_img)
        grid = self.create_grid_from_images_colorization(support_img, support_mask, query_img, query_mask)
        
        return grid

    def create_grid_from_images_colorization(self, support_img, support_mask, query_img, query_mask):
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