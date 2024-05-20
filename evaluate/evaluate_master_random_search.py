import os.path
from tqdm import trange
import rl_dataloader
from evaluate_detection.box_ops import to_rectangle
from evaluate_detection.canvas_ds import CanvasDataset
from reasoning_dataloader import *
import torchvision
from mae_utils import *
import argparse
from pathlib import Path
from segmentation_utils import *
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import numpy as np
import os
from contextlib import ExitStack
import torch
import torchvision.transforms as T
from PIL import Image
import pickle
import torch.optim as optim
import torch.nn as nn
import random


def get_args():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--output_dir', default='../output_dir/')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--base_dir', default='/home/ahojel/datasets/', help='pascal base dir')
    parser.add_argument('--seed', default=15, type=int)
    parser.add_argument('--t', default=[0, 0, 0], type=float, nargs='+')
    parser.add_argument('--ckpt', help='model checkpoint')
    parser.add_argument('--split', default=0 , type=int)
    parser.add_argument('--purple', default=0, type=int)
    parser.add_argument('--flip', default=0, type=int)
    parser.add_argument('--save_images', default=None, type=int, help='Save images')
    parser.add_argument('--query_support_list_file', default=None, type=str, help='Directory of query support list file')
    parser.add_argument('--train_images', default=100, type=int)
    parser.add_argument('--random_train_images', default=0, type=int)
    parser.add_argument('--eval_iters', default=100, type=int)
    parser.add_argument('--store_latents', default=None, type=str, help='Where to store latents')
    parser.add_argument('--zero_shot', default=0, type=int)
    parser.add_argument('--task', default=None, type=int)
    parser.add_argument('--granularity', default=1, type=int)
    parser.add_argument('--load_model', default=None, type=str, help='Where to load model from')

    return parser


ordered_layer_list = [30, 25, 31, 28, 27, 29,  3, 24, 23, 22,  0, 26, 13, 21, 10, 16,  5, 11, 4,  2, 20,  7,  9, 12,  8,  6, 15, 19,  1, 18, 17, 14]
ordered_head_list = [[25, 2], [30, 7], [28, 3], [27, 8], [21, 2], [3, 9], [24, 3], [26, 10], [24, 15], [16, 8], [30, 0], [3, 11], [30, 3], [28, 9], [29, 15], [0, 0], [10, 2], [26, 0], [26, 7], [29, 4], [22, 14], [31, 13], [25, 7], [29, 2], [0, 10], [23, 13], [29, 7], [30, 5], [30, 11], [10, 12], [23, 8], [31, 1], [12, 5], [30, 8], [30, 14], [11, 11], [27, 3], [28, 13], [30, 12], [22, 0], [23, 5], [25, 10], [24, 4], [13, 6], [27, 0], [0, 14], [22, 9], [24, 7], [24, 10], [23, 14], [4, 12], [29, 12], [31, 5], [26, 14], [27, 9], [6, 4], [30, 2], [16, 13], [31, 11], [31, 12], [22, 11], [11, 13], [31, 15], [5, 2], [31, 0], [27, 4], [8, 2], [21, 15], [27, 14], [5, 14], [25, 1], [22, 13], [11, 5], [12, 2], [3, 4], [3, 7], [28, 6], [25, 15], [2, 8], [13, 1], [5, 13], [29, 8], [24, 5], [31, 10], [7, 12], [6, 2], [28, 5], [27, 1], [9, 15], [31, 8], [17, 14], [28, 0], [8, 9], [26, 6], [5, 3], [31, 7], [15, 3], [31, 6], [4, 15], [21, 8], [10, 0], [31, 3], [25, 8], [23, 10], [3, 1], [13, 9], [31, 4], [0, 4], [28, 11], [0, 8], [30, 10], [3, 6], [29, 10], [7, 7], [23, 3], [31, 2], [30, 4], [0, 9], [23, 15], [4, 5], [25, 13], [13, 11], [9, 9], [8, 3], [18, 11], [20, 9], [0, 5], [27, 13], [22, 8], [1, 10], [28, 15], [3, 2], [1, 9], [21, 1], [5, 8], [31, 14], [4, 9], [5, 15], [28, 14], [28, 7], [1, 13], [11, 8], [0, 1], [29, 0], [13, 0], [20, 2], [0, 2], [19, 14], [2, 0], [20, 11], [13, 8], [29, 6], [5, 6], [20, 3], [25, 4], [10, 14], [3, 15], [24, 14], [20, 8], [15, 5], [18, 9], [25, 12], [28, 1], [20, 1], [15, 1], [27, 7], [19, 11], [21, 14], [5, 4], [18, 3], [17, 5], [21, 6], [28, 2], [22, 10], [16, 1], [25, 14], [28, 10], [14, 15], [13, 3], [9, 13], [0, 3], [13, 12], [22, 12], [13, 15], [2, 10], [26, 8], [31, 9], [3, 13], [9, 14], [28, 12], [29, 9], [16, 5], [4, 6], [1, 5], [7, 13], [1, 3], [6, 12], [27, 6], [2, 5], [3, 10], [1, 12], [23, 7], [25, 5], [27, 10], [10, 6], [23, 9], [2, 13], [6, 0], [27, 5], [23, 12], [23, 0], [1, 1], [9, 1], [22, 7], [2, 4], [3, 5], [18, 15], [2, 6], [25, 9], [3, 8], [22, 2], [13, 2], [22, 6], [7, 0], [14, 8], [15, 11], [4, 1], [6, 9], [14, 5], [19, 6], [25, 0], [13, 10], [10, 15], [19, 5], [10, 1], [16, 2], [20, 15], [2, 14], [7, 1], [21, 7], [4, 3], [2, 12], [29, 3], [10, 7], [16, 7], [13, 7], [30, 6], [16, 6], [10, 9], [27, 2], [7, 14], [28, 8], [16, 9], [9, 3], [15, 15], [8, 4], [16, 12], [24, 13], [28, 4], [23, 4], [2, 7], [18, 6], [3, 3], [7, 9], [8, 13], [24, 8], [22, 3], [26, 9], [3, 12], [15, 0], [22, 4], [14, 7], [21, 11], [30, 13], [0, 12], [24, 12], [13, 13], [29, 14], [27, 12], [22, 5], [11, 1], [17, 0], [21, 10], [30, 15], [13, 14], [19, 3], [4, 11], [2, 3], [10, 3], [26, 12], [11, 3], [19, 10], [17, 15], [5, 7], [22, 1], [29, 5], [3, 0], [8, 1], [15, 12], [7, 2], [2, 2], [11, 0], [16, 10], [14, 12], [11, 6], [27, 15], [23, 1], [9, 8], [25, 11], [26, 2], [19, 1], [3, 14], [7, 8], [24, 11], [24, 0], [17, 8], [21, 0], [25, 3], [18, 13], [12, 13], [2, 9], [29, 1], [0, 15], [1, 7], [23, 11], [27, 11], [4, 4], [26, 1], [17, 12], [9, 10], [6, 1], [4, 13], [5, 10], [12, 9], [19, 4], [5, 0], [0, 13], [9, 12], [14, 2], [6, 7], [19, 0], [20, 14], [25, 6], [18, 5], [29, 13], [24, 1], [7, 5], [6, 11], [12, 11], [18, 2], [16, 11], [29, 11], [15, 7], [16, 4], [23, 6], [14, 0], [11, 15], [20, 10], [16, 15], [10, 4], [20, 12], [12, 0], [23, 2], [17, 7], [12, 3], [14, 6], [30, 1], [17, 4], [20, 5], [21, 9], [19, 12], [4, 8], [9, 2], [20, 6], [11, 7], [14, 13], [9, 6], [12, 12], [15, 13], [12, 15], [20, 4], [11, 14], [21, 12], [13, 4], [7, 4], [24, 6], [19, 13], [17, 6], [17, 3], [12, 14], [13, 5], [17, 9], [7, 6], [12, 10], [19, 15], [18, 4], [0, 6], [24, 2], [4, 10], [4, 0], [8, 12], [14, 1], [30, 9], [14, 14], [4, 2], [15, 9], [11, 2], [26, 3], [12, 8], [10, 13], [5, 5], [26, 15], [19, 2], [8, 0], [4, 14], [12, 7], [21, 3], [11, 12], [18, 0], [11, 4], [6, 5], [10, 10], [15, 6], [14, 9], [9, 5], [9, 7], [15, 10], [10, 5], [8, 6], [16, 0], [8, 7], [4, 7], [18, 10], [14, 4], [12, 6], [11, 10], [16, 3], [15, 8], [22, 15], [20, 7], [6, 15], [17, 13], [8, 8], [8, 14], [19, 8], [19, 7], [8, 10], [17, 2], [10, 8], [14, 3], [1, 2], [6, 10], [6, 6], [8, 11], [12, 1], [10, 11], [20, 0], [18, 12], [15, 14], [7, 15], [21, 5], [2, 1], [14, 11], [15, 2], [14, 10], [1, 6], [6, 13], [21, 13], [17, 10], [24, 9], [8, 15], [19, 9], [9, 11], [7, 11], [18, 8], [21, 4], [5, 11], [12, 4], [15, 4], [2, 15], [7, 3], [6, 14], [7, 10], [11, 9], [18, 14], [9, 0], [18, 7], [9, 4], [8, 5], [17, 1], [20, 13], [1, 11], [26, 11], [16, 14], [18, 1], [5, 1], [0, 7], [6, 3], [2, 11], [26, 13], [17, 11], [26, 5], [5, 9], [26, 4], [1, 8], [6, 8], [5, 12], [1, 0], [0, 11], [1, 15], [1, 4], [1, 14]]

class JointModel(nn.Module):
    def __init__(self, args, prompting_model, num_variables, train_ds, eval_ds, task_tensor, load_model=None):
        super().__init__()
        self.prompting_model = prompting_model
        self.num_variables = num_variables
        self.bernoullis = []#[[25,2,2], [25,2,1], [30,7,2], [27,8,2], [10,2,1], [10,12,1], [8,3,1], [8,9,1], [23,15,0], [5,14,1], [6,0,1], [26,7,2], [25,0,2], [24, 15, 2]]

        self.eps = 1e-6
        self.batch_size = 5*args.train_images if args.task is None else args.train_images

        self.train_ds = train_ds
        self.eval_ds = eval_ds

        self.task_tensor = task_tensor

        self.areas_to_check = [None]

        """ if load_model is not None:
            self.bernoullis = pickle.load(open(load_model, 'rb'))
            self.bernoullis = [torch.tensor(bernoulli, requires_grad=True) for bernoulli in self.bernoullis] """

    def get_good_init(self, args, canvases, sample_count):

        all_areas = [[layer, head, quadrant] for layer in range(24,32) for head in range(0,16) for quadrant in [1,2]]+[[layer, head, quadrant] for layer in range(0,24) for head in range(0,16) for quadrant in [1]]
        
        initializations = []
        for sample in range(sample_count):

            random.shuffle(all_areas)
            rand_select = [area for area in all_areas if random.choice([True, False])]
            current = rand_select + self.bernoullis
            initializations.append(current)

        initializations.append(self.bernoullis+all_areas)

        loss_list = []

        for init in initializations:
            save = False
            areas_to_patch = init

            indices = self.construct_indices(areas_to_patch)
            for i in trange(0,self.batch_size):

                canvas = canvases[i%len(canvases)]

                if args.task is not None:
                    current_injection = self.task_tensor
                else:
                    current_injection = self.task_tensor[i%len(self.task_tensor)]
            
                with torch.no_grad():        
                    if args.zero_shot:
                        indices_premask = []
                        indices_postmask = []

                        drop_indices = [1+a for a in calculate_quadrant_indices(14, 14, 1)]+[1+a for a in calculate_quadrant_indices(14, 14, 2)]
                        
                        indices_postmask += [1+a for a in calculate_quadrant_indices(14, 14, 3)] + [1+a for a in calculate_quadrant_indices(14, 14, 4)]
                        #indices_premask = list(range(0,148))#
                        indices_premask = [0]+list(range(99,148))

                        if indices is not None:
                            original_image, generated_result, _ = _generate_result_for_canvas(args, self.prompting_model, canvas, premask_pass_indices = indices_premask, attention_heads=indices, attention_injection=current_injection, postmask_pass_indices = indices_postmask, drop_indices = drop_indices)
                        else: 
                            original_image, generated_result, _ = _generate_result_for_canvas(args, self.prompting_model, canvas, premask_pass_indices = indices_premask, postmask_pass_indices = indices_postmask, drop_indices = drop_indices)
                    else:
                        original_image, generated_result, _ = _generate_result_for_canvas(args, self.prompting_model, canvas, attention_heads=indices, attention_injection=current_injection)
                    
                
                if args.task == 0:
                    loss = -1*self.loss_iou(generated_result, original_image)
                else:
                    if args.task is None:
                        if i%len(self.task_tensor) == 0:
                            loss = -1*self.loss_iou(generated_result, original_image)
                        else:
                            loss = self.loss_mse(original_image, generated_result)
                    else:
                        loss = self.loss_mse(original_image, generated_result)

                loss_list.append(loss)

        loss_list = torch.tensor(loss_list)
        loss_list_copy = torch.tensor(loss_list)

        if args.task is None:
            for index in range(len(self.task_tensor)):
                current = loss_list[index::len(self.task_tensor)]
                loss_list[index::len(self.task_tensor)] = (current-current.mean())/(current.std() + self.eps)
        else:
            loss_list = (loss_list-loss_list.mean())/(loss_list.std() + self.eps)

        reshaped_tensor = loss_list.reshape(len(initializations), self.batch_size)
        averages = reshaped_tensor.mean(axis=1)
        best_index = torch.argmin(averages)

        best_loss = loss_list_copy.reshape(len(initializations), self.batch_size).mean(axis=1)[best_index]     

        self.bernoullis = initializations[best_index]
        return best_loss, initializations[best_index]
        

    def loss_mse(self, target, ours):
        ours = (torch.permute(ours / 255., (2, 0, 1)) - torch.tensor(imagenet_mean, dtype=torch.float32).to(ours.device)[:, None, None]) / torch.tensor(imagenet_std, dtype=torch.float32).to(ours.device)[:, None, None]
        target = (torch.permute(target.to(ours.device) / 255., (2, 0, 1)) - torch.tensor(imagenet_mean, dtype=torch.float32).to(ours.device)[:, None, None]) / torch.tensor(imagenet_std, dtype=torch.float32).to(ours.device)[:, None, None]

        target = target[:, 113:, 113:]
        ours = ours[:, 113:, 113:]
        mse = torch.mean((target - ours) ** 2)
        return mse.item()

    def loss_iou(self, original_image, generated_result):
        fg_color=WHITE
        bg_color=BLACK

        original_image = round_image(original_image, [WHITE, BLACK])
        generated_result = round_image(generated_result, [WHITE, BLACK], t=args.t)

        target = original_image[113:, 113:].to(original_image)
        ours = generated_result[113:, 113:].to(original_image)

        fg_color = torch.tensor(fg_color, dtype=torch.float32, device=target.device)
        seg_orig = ((target - fg_color[None, None, :]) == 0).all(dim=2)
        seg_our = ((ours - fg_color[None, None, :]) == 0).all(dim=2)
        iou = torch.sum(seg_orig & seg_our).float() / torch.sum(seg_orig | seg_our).float()

        return iou
    
    def construct_indices(self, sampled_patches):
        
        """ if args.granularity==0:
            sampled_patches = sampled_patches.reshape(32,16)
        
            indices = sampled_patches.nonzero()
            expanded_indices = []

            for element in indices:
                layer = int(element[0].item())
                head = int(element[1].item())
                if element[0] < 24:
                    for a in range(0, 50 if args.zero_shot else 148):
                        expanded_indices.append([layer, head, a])
                else:
                    for a in range(0, 99 if args.zero_shot else 197):
                        expanded_indices.append([layer, head, a])
            indices = expanded_indices

            return indices
            """
        if sampled_patches==[]:
            return None
        
        if args.granularity==1:
        
            indices = sampled_patches
            expanded_indices = []

            for element in indices:
                
                head = element[0]
                layer = element[1]
                quadrant = element[2]
                
                if quadrant == 0:
                    expanded_indices.append([head, layer, 0])
                elif quadrant == 1:
                    if head<24:
                        for a in range(1, 50):
                            expanded_indices.append([head,layer, a])
                    else:
                        for a in q1:
                            expanded_indices.append([head, layer, a+1])
                elif quadrant == 2:
                    for a in q2:
                        expanded_indices.append([head, layer, a+1])
            indices = expanded_indices

            return indices
        
            """  if args.granularity==2:
        
            indices = sampled_patches.nonzero()
            expanded_indices = []

            for element in indices:
                index = int(element.item())
                total_elements_per_layer_first_24_heads = 16 * 50  # 16 layers, 2 elements per layer
                total_elements_per_layer_next_8_heads = 16 * 99  # 16 layers, 3 elements per layer
                total_elements_first_24_heads = 24 * total_elements_per_layer_first_24_heads
                
                if index < total_elements_first_24_heads:
                    head = index // total_elements_per_layer_first_24_heads
                    layer = (index % total_elements_per_layer_first_24_heads) // 50
                    quadrant = index % 50
                else:
                    adjusted_index = index - total_elements_first_24_heads
                    head = 24 + (adjusted_index // total_elements_per_layer_next_8_heads)
                    layer = (adjusted_index % total_elements_per_layer_next_8_heads) // 99
                    quadrant = adjusted_index % 99
                
                expanded_indices.append([head, layer, quadrant])
            indices = expanded_indices


            return indices """
    
    def forward(self, args, canvases, iter):

        

        self.areas_to_check = [list(x) for x in set(tuple(x) for x in self.areas_to_check if x is not None)]
        self.areas_to_check = [x for x in self.areas_to_check if x not in self.bernoullis]
        self.areas_to_check.append(None)  # Adding None back to the list as required

        t_holder = self.areas_to_check

        random.shuffle(self.areas_to_check)
        self.areas_to_check = self.areas_to_check[:100]

        loss_list = []

        for addition in trange(len(self.areas_to_check)):
            addition = self.areas_to_check[addition]
            save = False
            areas_to_patch = self.bernoullis

            if addition is not None:
                
                areas_to_patch = areas_to_patch + [addition]
            else:
                save=True

            indices = self.construct_indices(areas_to_patch)
            for i in range(0,self.batch_size):

                canvas = canvases[i%len(canvases)]

                if args.task is not None:
                    current_injection = self.task_tensor
                else:
                    current_injection = self.task_tensor[i%len(self.task_tensor)]
            
                with torch.no_grad():        
                    if args.zero_shot:
                        indices_premask = []
                        indices_postmask = []

                        drop_indices = [1+a for a in calculate_quadrant_indices(14, 14, 1)]+[1+a for a in calculate_quadrant_indices(14, 14, 2)]
                        
                        indices_postmask += [1+a for a in calculate_quadrant_indices(14, 14, 3)] + [1+a for a in calculate_quadrant_indices(14, 14, 4)]
                        #indices_premask = list(range(0,148))#
                        indices_premask = [0]+list(range(99,148))

                        if indices is not None:
                            original_image, generated_result, _ = _generate_result_for_canvas(args, self.prompting_model, canvas, premask_pass_indices = indices_premask, attention_heads=indices, attention_injection=current_injection, postmask_pass_indices = indices_postmask, drop_indices = drop_indices)
                        else: 
                            original_image, generated_result, _ = _generate_result_for_canvas(args, self.prompting_model, canvas, premask_pass_indices = indices_premask, postmask_pass_indices = indices_postmask, drop_indices = drop_indices)
                    else:
                        original_image, generated_result, _ = _generate_result_for_canvas(args, self.prompting_model, canvas, attention_heads=indices, attention_injection=current_injection)
                    
                
                if args.task == 0:
                    loss = -1*self.loss_iou(generated_result, original_image)
                else:
                    if args.task is None:
                        if i%len(self.task_tensor) == 0:
                            loss = -1*self.loss_iou(generated_result, original_image)
                        else:
                            loss = self.loss_mse(original_image, generated_result)
                    else:
                        loss = self.loss_mse(original_image, generated_result)

                loss_list.append(loss)

                if save and i == 0:
                    image = generated_result.detach().cpu().numpy()
                    plt.figure()
                    plt.imshow(image)
                    plt.axis('off')  # Turn off axis numbers and ticks
                    image_save_path = os.path.join(args.output_dir, f'core_{iter}_{len(self.bernoullis)}.png')
                    plt.savefig(image_save_path, bbox_inches='tight', pad_inches=0)   # Save the image without padding and no axis
                    plt.close()  # Close the plot to prevent it from displaying in the notebook or script output

        for removal in trange(len(self.bernoullis)):
            removal = self.bernoullis[removal]
            save = False
            areas_to_patch = [area for area in self.bernoullis if area != removal]
                
            indices = self.construct_indices(areas_to_patch)
            for i in range(0,self.batch_size):

                canvas = canvases[i%len(canvases)]

                if args.task is not None:
                    current_injection = self.task_tensor
                else:
                    current_injection = self.task_tensor[i%len(self.task_tensor)]
            
                with torch.no_grad():        
                    if args.zero_shot:
                        indices_premask = []
                        indices_postmask = []

                        drop_indices = [1+a for a in calculate_quadrant_indices(14, 14, 1)]+[1+a for a in calculate_quadrant_indices(14, 14, 2)]
                        
                        indices_postmask += [1+a for a in calculate_quadrant_indices(14, 14, 3)] + [1+a for a in calculate_quadrant_indices(14, 14, 4)]
                        #indices_premask = list(range(0,148))#
                        indices_premask = [0]+list(range(99,148))

                        if indices is not None:
                            original_image, generated_result, _ = _generate_result_for_canvas(args, self.prompting_model, canvas, premask_pass_indices = indices_premask, attention_heads=indices, attention_injection=current_injection, postmask_pass_indices = indices_postmask, drop_indices = drop_indices)
                        else: 
                            original_image, generated_result, _ = _generate_result_for_canvas(args, self.prompting_model, canvas, premask_pass_indices = indices_premask, postmask_pass_indices = indices_postmask, drop_indices = drop_indices)
                    else:
                        original_image, generated_result, _ = _generate_result_for_canvas(args, self.prompting_model, canvas, attention_heads=indices, attention_injection=current_injection)
                    
                
                if args.task == 0:
                    loss = -1*self.loss_iou(generated_result, original_image)
                else:
                    if args.task is None:
                        if i%len(self.task_tensor) == 0:
                            loss = -1*self.loss_iou(generated_result, original_image)
                        else:
                            loss = self.loss_mse(original_image, generated_result)
                    else:
                        loss = self.loss_mse(original_image, generated_result)

                loss_list.append(loss)

        loss_list = torch.tensor(loss_list)
        loss_list_copy = torch.tensor(loss_list)

        if args.task is None:
            for index in range(len(self.task_tensor)):
                current = loss_list[index::len(self.task_tensor)]
                loss_list[index::len(self.task_tensor)] = (current-current.mean())/(current.std() + self.eps)
        else:
            loss_list = (loss_list-loss_list.mean())/(loss_list.std() + self.eps)

        reshaped_tensor = loss_list.reshape(len(self.areas_to_check)+len(self.bernoullis), self.batch_size)
        averages = reshaped_tensor.mean(axis=1)

        added = []
        removed = []
        
        for change in range(3):
            best_index = torch.argsort(averages)[change]

            try:
                best_loss = loss_list_copy.reshape(len(self.areas_to_check)+len(self.bernoullis), self.batch_size).mean(axis=1)[best_index]
            except:
                best_loss = None
            
            if best_index < len(self.areas_to_check):
                best_move = [self.areas_to_check[best_index]]
                print("Added:", best_move)
            else:
                best_move = [self.bernoullis[best_index-len(self.areas_to_check)]]
                print("Removed:", best_move)

            if best_move[0] is None:
                return best_loss, generated_result, added, removed, True
            
            if best_index < len(self.areas_to_check):
                self.bernoullis = self.bernoullis + best_move
                added.append(best_move)
                t_holder.pop(best_index)
            else:
                self.bernoullis.pop(best_index-len(self.areas_to_check))
                removed.append(best_move)
                t_holder = t_holder + best_move            

        self.areas_to_check = t_holder

        return best_loss, generated_result, added, removed, False

    def train(self, args, num_itr):

        canvases = []
        for idx in range(len(self.train_ds)):
            if args.task is not None:
                canvas = self.train_ds[idx]['grid']
                canvas = (canvas - imagenet_mean[:, None, None]) / imagenet_std[:, None, None]

                canvases.append(canvas)
            else:
                canvas = self.train_ds[idx]['grid']
                for element in canvas:
                    element = (element - imagenet_mean[:, None, None]) / imagenet_std[:, None, None]
                    canvases.append(element)

        start_layer_count = 8
        start_layers = ordered_layer_list[:start_layer_count]
        
        for layer in start_layers:
            if layer<=23:
                self.areas_to_check += [[layer, head, quadrant] for head in range(0,16) for quadrant in [0,1]]
            if layer>=24:
                self.areas_to_check += [[layer, head, quadrant] for head in range(0,16) for quadrant in [0,1,2]]
        
        self.get_good_init(args, canvases, 100)

        

        start_layer_count = 3
        start_layers = ordered_layer_list[:start_layer_count]

        self.areas_to_check = [None]
        
        for layer in start_layers:
            if layer<=23:
                self.areas_to_check += [[layer, head, quadrant] for head in range(0,16) for quadrant in [1]]
            if layer>=24:
                self.areas_to_check += [[layer, head, quadrant] for head in range(0,16) for quadrant in [1,2]]

        
        for i in trange(num_itr):
            best_loss, res, added, removed, kill = self.forward(args, canvases, i)

            print("Iter:",i,"Loss:",best_loss, "Layer Count:", start_layer_count, "Checking:", len(self.areas_to_check), "Bernoullis:", len(self.bernoullis))

            with open(os.path.join(args.output_dir,'log.txt'), 'a') as log:
                current_metric = {}
                current_metric["iter"] = i
                current_metric["train_loss"] = best_loss
                current_metric["layers_checking"] = start_layers
                current_metric["added"] = added
                current_metric["removed"] = removed
                current_metric["all_areas"] = self.bernoullis

                log.write(str(current_metric) + '\n')

            eval_iou, eval_indices = self.eval(args, i)
            print("Iter:",i,"Eval:",eval_iou, "Layer Count:", start_layer_count)

            with open(os.path.join(args.output_dir,'log.txt'), 'a') as log:
                current_metric = {}
                current_metric["iter"] = i
                current_metric["eval_loss"] = eval_iou
                current_metric["all_areas"] = self.bernoullis

                log.write(str(current_metric) + '\n')
            
            """if i % 50 == 0:

                eval_iou, eval_indices = self.eval(args)

                with open(os.path.join(args.output_dir,'log.txt'), 'a') as log:
                    current_metric = {}
                    current_metric["iter"] = i
                    current_metric["eval_loss"] = eval_iou
                    current_metric["eval_patch_count"] = eval_indices
                    current_metric["lr"] = args.lr
                    current_metric["init"] = args.init
                    current_metric["granularity"] = args.granularity
                    current_metric["batch_size"] = self.batch_size
                    current_metric["reg_strength"] = self.regularization_strength
                    current_metric["images_per_batch"] = args.train_images
                    current_metric["task"] = args.task
                    current_metric["load_model"] = "True" if args.load_model is not None else "False"

                    log.write(str(current_metric) + '\n')
                
                # Save self.bernoullis to a pickle file with suffix {args.lr}_{args.init}_{i}
                bernoullis_save_path = os.path.join(args.output_dir, f'bernoullis_{args.task}_{args.granularity}_{self.regularization_strength}_{args.train_images}_{args.lr}_{args.init}_{i}.pkl')
                with open(bernoullis_save_path, 'wb') as f:
                    pickle.dump([bernoulli.detach().cpu().numpy() for bernoulli in self.bernoullis], f) """

            if i % 5 == 0 and i!=0:
                start_layer_count+=1
                start_layers = ordered_layer_list[:start_layer_count]

                self.areas_to_check = [None]
                
                for layer in start_layers:
                    if layer<=23:
                        self.areas_to_check += [[layer, head, quadrant] for head in range(0,16) for quadrant in [1]]
                    if layer>=24:
                        self.areas_to_check += [[layer, head, quadrant] for head in range(0,16) for quadrant in [1,2]]

            
        return res.detach().cpu().numpy()
    
    def eval(self, args, iter):

        if args.task is None:
            curr_injection = self.task_tensor[0]
        else:
            curr_injection = self.task_tensor
        
        indices = self.construct_indices(self.bernoullis)
        
        loss_holder = []
        for idx in trange(len(self.eval_ds)):

            canvas = self.eval_ds[idx]['grid']
            canvas = (canvas - imagenet_mean[:, None, None]) / imagenet_std[:, None, None]

            with torch.no_grad():        
                if args.zero_shot:
                    indices_premask = []
                    indices_postmask = []

                    drop_indices = [1+a for a in calculate_quadrant_indices(14, 14, 1)]+[1+a for a in calculate_quadrant_indices(14, 14, 2)]
                    
                    indices_postmask += [1+a for a in calculate_quadrant_indices(14, 14, 3)] + [1+a for a in calculate_quadrant_indices(14, 14, 4)]
                    #indices_premask = list(range(0,148))#
                    indices_premask = [0]+list(range(99,148))

                    original_image, generated_result, _ = _generate_result_for_canvas(args, self.prompting_model, canvas, premask_pass_indices = indices_premask, attention_heads=indices, attention_injection=curr_injection, postmask_pass_indices = indices_postmask, drop_indices = drop_indices)

                    if idx%5==0:
                    
                        image = generated_result.detach().cpu().numpy()
                        plt.figure()
                        plt.imshow(image)
                        plt.axis('off')  # Turn off axis numbers and ticks
                        image_save_dir = os.path.join(args.output_dir, 'rip')
                        if not os.path.exists(image_save_dir):
                            os.makedirs(image_save_dir)
                        image_save_path = os.path.join(image_save_dir, f'train_{iter}_{idx}.png')
                        plt.savefig(image_save_path, bbox_inches='tight', pad_inches=0)  # Save the image without padding and no axis
                        plt.close()  # Close the plot to prevent it from displaying in the notebook or script output


                        image = original_image.detach().cpu().numpy()
                        plt.figure()
                        plt.imshow(image)
                        plt.axis('off')  # Turn off axis numbers and ticks
                        image_save_dir = os.path.join(args.output_dir, 'rip')
                        if not os.path.exists(image_save_dir):
                            os.makedirs(image_save_dir)
                        image_save_path = os.path.join(image_save_dir, f'train_{iter}_{idx}_original.png')
                        plt.savefig(image_save_path, bbox_inches='tight', pad_inches=0)  # Save the image without padding and no axis
                        plt.close()  # Close the plot to prevent it from displaying in the notebook or script output
                 
                else:
                    original_image, generated_result, _ = _generate_result_for_canvas(args, self.prompting_model, canvas, attention_heads=indices, attention_injection=self.task_tensor)
                
            if args.task is None:
                #import pdb; breakpoint()
                loss = self.loss_iou(original_image, generated_result).item()
            elif args.task == 0:
                loss = self.loss_iou(original_image, generated_result).item()
            else:
                loss = self.loss_mse(original_image, generated_result)
            loss_holder.append(loss)

        eval_mean_iou = np.mean(loss_holder)

        return eval_mean_iou, len(indices)

def _generate_result_for_canvas(args, model, canvas, premask_pass_indices = None, postmask_pass_indices = None, attention_heads=None, attention_injection=None, drop_indices = None):
    """canvas is already in the right range."""

    ids_shuffle, len_keep = generate_mask_for_evaluation()
    if attention_heads is not None:
        attention_heads = torch.tensor(attention_heads, dtype=torch.int64).to(args.device)
        

    _, im_paste, _, latents = generate_image(canvas.unsqueeze(0).to(args.device), model, ids_shuffle.to(args.device),
                                    len_keep, device=args.device, premask_pass_indices = premask_pass_indices, postmask_pass_indices = postmask_pass_indices, attention_heads = attention_heads, attention_injection = attention_injection, record=False, drop_indices = drop_indices)

    canvas = torch.einsum('chw->hwc', canvas)
    canvas = torch.clip((canvas * imagenet_std + imagenet_mean) * 255, 0, 255).int()
    assert canvas.shape == im_paste.shape, (canvas.shape, im_paste.shape)
    return canvas, im_paste, latents

def evaluate(args):
    padding = 1
    image_transform = torchvision.transforms.Compose(
        [torchvision.transforms.Resize((224 // 2 - padding, 224 // 2 - padding), 3),
         torchvision.transforms.ToTensor()])
    mask_transform = [torchvision.transforms.Compose(
        [torchvision.transforms.Resize((224 // 2 - padding, 224 // 2 - padding), 3),
         torchvision.transforms.ToTensor()]), torchvision.transforms.Compose(
        [torchvision.transforms.Resize((224 // 2 - padding, 224 // 2 - padding), 3),
         torchvision.transforms.Grayscale(3),
         torchvision.transforms.ToTensor()])]

    ds = rl_dataloader.DatasetPASCAL(args.base_dir, fold=args.split, image_transform=image_transform, mask_transform=mask_transform,
                         flipped_order=args.flip, purple=args.purple, query_support_list_file=args.query_support_list_file, iters=(1 + args.random_train_images)*args.train_images, type="trn", task=args.task)
    
    eval_ds = rl_dataloader.DatasetPASCAL(args.base_dir, fold=args.split, image_transform=image_transform, mask_transform=mask_transform,
                         flipped_order=args.flip, purple=args.purple, query_support_list_file=args.query_support_list_file, iters=args.eval_iters, type="val", task=0)
    
    model = prepare_model(args.ckpt, arch=args.model)
    _ = model.to(args.device)

    with open('/home/ahojel/visual_prompting_vid/task_vectors.pkl', 'rb') as f:
        injection_master = pickle.load(f)

    tasks = ["segmentation", "lowlight enhance", "segmentation_neutral", "inpaint single random", "colorization"]

    
    if args.task is not None:
        task = tasks[args.task]
    
        enc_inj = torch.tensor(injection_master["encoder"][task]).to(args.device)
        dec_inj = torch.tensor(injection_master["decoder"][task]).to(args.device)
                
        injection = [enc_inj,dec_inj]
    else:
        injection = []
        for task_element in tasks:
            enc_inj = torch.tensor(injection_master["encoder"][task_element]).to(args.device)
            dec_inj = torch.tensor(injection_master["decoder"][task_element]).to(args.device)
            injection.append([enc_inj,dec_inj])

    
    if args.granularity==0:
        params = 24*16+8*16
    elif args.granularity==1:
        params = 24*16*2+8*16*3
    elif args.granularity==2:
        params = 24*16*50+8*16*99

    rl_model = JointModel(args, model, params, ds, eval_ds, injection, args.load_model)
    rl_model = rl_model.to(args.device)

    
    rl_model.train(args, num_itr=60)

def determine_quartile(z, q1, q2, q3, q4):
    """Determine the quartile group for a given z value."""
    holder = z
    z = z[-1]
    if holder[0]<=23:
        if z == 0:
            return 'q0'
        elif z-1 in q1:
            return 'q1'
        elif z-1 in q2:
            return 'q2'
        else:
            return 'q3'
    else:
        if z == 0:
            return 'q0'
        elif z-1 in q1:
            return 'q1'
        elif z-1 in q2:
            return 'q2'
        elif z-1 in q3:
            return 'q3'
        elif z-1 in q4:
            return 'q4'

def rank_coordinates_fine_grained(coord_value_pairs, q1, q2, q3, q4):
    # Unzip the list of pairs into separate lists
    coordinates, values = zip(*coord_value_pairs)
    
    # Group coordinates by (x, y) and then by quartile
    groups = defaultdict(lambda: defaultdict(list))
    for coord, value in zip(coordinates, values):
        xy_group = tuple(coord[:2])
        quartile = determine_quartile(coord, q1, q2, q3, q4)
        groups[xy_group][quartile].append((coord[2], value))
    
    # Calculate average value for each fine-grained group and sort groups by this average
    fine_grained_averages = {}
    for xy_group, quartiles in groups.items():
        for quartile, members in quartiles.items():
            avg = np.mean([value for _, value in members])
            fine_grained_averages[(xy_group, quartile)] = avg
    
    sorted_fine_grained_groups = sorted(fine_grained_averages.keys(), key=lambda x: fine_grained_averages[x], reverse=True)
    
    # Sort members within each fine-grained group by their z value
    for xy_group, quartiles in groups.items():
        for quartile in quartiles:
            quartiles[quartile] = sorted(quartiles[quartile], key=lambda x: x[0])
    
    # Compile the ranked list based on fine-grained group average and then by z within each group
    ranked_list = []
    for group in sorted_fine_grained_groups:
        xy_group, quartile = group
        for z, _ in groups[xy_group][quartile]:
            ranked_list.append([*xy_group, z])
    
    return ranked_list


def rank_coordinates_zipped(coord_value_pairs):
    # Unzip the list of pairs into separate lists
    coordinates, values = zip(*coord_value_pairs)
    
    # Group coordinates by (x, y)
    groups = defaultdict(list)
    for coord, value in zip(coordinates, values):
        groups[tuple(coord[:2])].append((coord[2], value))
    
    # Calculate average value for each group and sort groups by this average
    group_averages = {group: np.mean([value for _, value in members]) for group, members in groups.items()}
    sorted_groups = sorted(group_averages.keys(), key=lambda x: group_averages[x], reverse=True)
    
    # Sort members within each group by their z value
    for group in groups:
        groups[group] = sorted(groups[group], key=lambda x: x[0])
    
    # Compile the ranked list based on group average and then by z within each group
    ranked_list = []
    for group in sorted_groups:
        for z, _ in groups[group]:
            ranked_list.append([*group, z])
    
    return ranked_list

     

def evaluate_segmentation(original_image, generated_result, args):
    if args.purple:
        original_image = round_image(original_image, [YELLOW, PURPLE])
    else:
        original_image = round_image(original_image, [WHITE, BLACK])

    if args.purple:
        generated_result = round_image(generated_result, [YELLOW, PURPLE], t=args.t)
    else:
        generated_result = round_image(generated_result, [WHITE, BLACK], t=args.t)

    if args.purple:
        current_metric = calculate_metric(args, original_image, generated_result, fg_color=YELLOW, bg_color=PURPLE)
    else:
        current_metric = calculate_metric(args, original_image, generated_result, fg_color=WHITE, bg_color=BLACK)

    return current_metric
    
def evaluate_mse(target, ours):
    ours = (torch.permute(ours / 255., (2, 0, 1)) - torch.tensor(imagenet_mean, dtype=torch.float32).to(ours.device)[:, None, None]) / torch.tensor(imagenet_std, dtype=torch.float32).to(ours.device)[:, None, None]
    target = (torch.permute(target.to(ours.device) / 255., (2, 0, 1)) - torch.tensor(imagenet_mean, dtype=torch.float32).to(ours.device)[:, None, None]) / torch.tensor(imagenet_std, dtype=torch.float32).to(ours.device)[:, None, None]

    target = target[:, 113:, 113:]
    ours = ours[:, 113:, 113:]
    mse = torch.mean((target - ours) ** 2)
    return mse.item()

def calculate_quadrant_indices(rows, cols, quadrant):
    """
    Calculate the start and end indices for each quadrant in the flattened tensor.
    """
    row_start, row_end = 0, 7
    col_start, col_end = 0, 7
    
    if quadrant == 2:  # Top Right
        col_start, col_end = 7, 14
    elif quadrant == 3:  # Bottom Left
        row_start, row_end = 7, 14
    elif quadrant == 4:  # Bottom Right
        row_start, row_end = 7, 14
        col_start, col_end = 7, 14

    indices = []
    for row in range(row_start, row_end):
        for col in range(col_start, col_end):
            index = row * rows + col
            indices.append(index)
    
    return indices


q1 = calculate_quadrant_indices(14, 14, 1)
q2 = calculate_quadrant_indices(14, 14, 2)
q3 = calculate_quadrant_indices(14, 14, 3)
q4 = calculate_quadrant_indices(14, 14, 4)


if __name__ == '__main__':
    args = get_args()

    args = args.parse_args()
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    evaluate(args)



