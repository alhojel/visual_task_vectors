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
    parser.add_argument('--train_images', default=10, type=int)
    parser.add_argument('--random_train_images', default=0, type=int)
    parser.add_argument('--eval_iters', default=300, type=int)
    parser.add_argument('--store_latents', default=None, type=str, help='Where to store latents')
    parser.add_argument('--zero_shot', default=0, type=int)
    parser.add_argument('--task', default=None, type=int)
    parser.add_argument('--granularity', default=1, type=int)
    parser.add_argument('--k', default=17, type=int)
    parser.add_argument('--load_model', default=None, type=str, help='Where to load model from')

    return parser

class JointModel(nn.Module):
    def __init__(self, args, prompting_model, num_variables, train_ds, eval_ds, task_tensor, load_model=None):
        super().__init__()
        self.prompting_model = prompting_model
        self.num_variables = num_variables
        self.bernoullis =[]#[[25,2], [26,0], [10,12], [26,0,2], [10,12,1], [16,13,1], [25,10,1], [25,10,2]]

        self.eps = 1e-6
        self.batch_size = 5*args.train_images if args.task is None else args.train_images

        self.train_ds = train_ds
        self.eval_ds = eval_ds

        self.task_tensor = task_tensor

        self.areas_to_check = [None]

        """ if load_model is not None:
            self.bernoullis = pickle.load(open(load_model, 'rb'))
            self.bernoullis = [torch.tensor(bernoulli, requires_grad=True) for bernoulli in self.bernoullis] """


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
        
            
        if sampled_patches==[]:
            return None
        
        if args.granularity==1:
        
            indices = sampled_patches
            expanded_indices = []

            for element in indices:
                
                layer = element[0]
                head = element[1]
                try:
                    quadrant = element[2]
                except:
                    import pdb; breakpoint()
                
                if quadrant == 0:
                    expanded_indices.append([layer, head, 0])
                elif quadrant == 1:
                    if layer<24:
                        for a in range(1, 50):
                            expanded_indices.append([layer,head, a])
                    else:
                        for a in q1:
                            expanded_indices.append([layer, head, a+1])
                elif quadrant == 2:
                    for a in q2:
                        expanded_indices.append([layer, head, a+1])
            indices = expanded_indices

            return indices
        
        if args.granularity==2:
            indices = sampled_patches
            expanded_indices = []

            for element in indices:
                
                layer = element[0]
                head = element[1]
                
                if head<24:
                    for a in range(50):
                        expanded_indices.append([layer,head, a])
                else:
                    for a in range(99):
                        expanded_indices.append([layer, head, a+1])
                
            indices = expanded_indices

            return indices
        
        if args.granularity==3:
            indices = sampled_patches
            expanded_indices = []

            for element in indices:
                
                layer = element[0]
                
                if layer<24:
                    for a in range(50):
                        for head in range(16):
                            expanded_indices.append([layer,head, a])
                else:
                    for a in range(99):
                        for head in range(16):
                            expanded_indices.append([layer, head, a])
                
            indices = expanded_indices

            return indices
    def get_good_init(self, args, canvases, sample_count, all_areas):

        initializations = []
        for sample in range(sample_count):

            random.shuffle(all_areas)
            rand_select = [area for area in all_areas if random.random() < 0.3]
            current = rand_select + self.bernoullis
            current = [list(x) for x in set(tuple(x) for x in current if x is not None)]

            initializations.append(current)

        loss_list = []

        for init in trange(len(initializations)):
            init = initializations[init]
            areas_to_patch = init

            indices = self.construct_indices(areas_to_patch)
            for i in range(0,self.batch_size):

                canvas = canvases[i]

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
                            original_image, generated_result, _, ce_loss = _generate_result_for_canvas(args, self.prompting_model, canvas, premask_pass_indices = indices_premask, attention_heads=indices, attention_injection=current_injection, postmask_pass_indices = indices_postmask, drop_indices = drop_indices)
                        else: 
                            original_image, generated_result, _, ce_loss = _generate_result_for_canvas(args, self.prompting_model, canvas, premask_pass_indices = indices_premask, postmask_pass_indices = indices_postmask, drop_indices = drop_indices)
                    else:
                        original_image, generated_result, _, ce_loss = _generate_result_for_canvas(args, self.prompting_model, canvas, attention_heads=indices, attention_injection=current_injection)
                    
                
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
    
        
    def forward(self, args, canvases, iter):

        #self.areas_to_check = [list(x) for x in set(tuple(x) for x in self.areas_to_check if x is not None)]
        self.areas_to_check = [x for x in self.areas_to_check if x not in self.bernoullis]

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
                            original_image, generated_result, _, ce_loss = _generate_result_for_canvas(args, self.prompting_model, canvas, premask_pass_indices = indices_premask, attention_heads=indices, attention_injection=current_injection, postmask_pass_indices = indices_postmask, drop_indices = drop_indices)
                        else: 
                            original_image, generated_result, _, ce_loss = _generate_result_for_canvas(args, self.prompting_model, canvas, premask_pass_indices = indices_premask, postmask_pass_indices = indices_postmask, drop_indices = drop_indices)
                    else:
                        original_image, generated_result, _, ce_loss = _generate_result_for_canvas(args, self.prompting_model, canvas, attention_heads=indices, attention_injection=current_injection)
                    
                
                """ if args.task == 0:
                    loss = -1*self.loss_iou(generated_result, original_image)
                else:
                    if args.task is None:
                        if i%len(self.task_tensor) == 0:
                            loss = -1*self.loss_iou(generated_result, original_image)
                        else:
                            loss = self.loss_mse(original_image, generated_result)
                    else:
                        loss = self.loss_mse(original_image, generated_result) """
                loss = ce_loss
                loss_list.append(loss)

                if save and i == 0:
                    image = generated_result.detach().cpu().numpy()
                    plt.figure()
                    plt.imshow(image)
                    plt.axis('off')  # Turn off axis numbers and ticks
                    image_save_path = os.path.join(args.output_dir, f'core_{iter}_{len(self.bernoullis)}.png')
                    plt.savefig(image_save_path, bbox_inches='tight', pad_inches=0)   # Save the image without padding and no axis
                    plt.close()  # Close the plot to prevent it from displaying in the notebook or script output

        removal_holder = [ele for ele in self.bernoullis]
        for removal in trange(len(removal_holder)):
            removal = removal_holder[removal]
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
                            original_image, generated_result, _, ce_loss = _generate_result_for_canvas(args, self.prompting_model, canvas, premask_pass_indices = indices_premask, attention_heads=indices, attention_injection=current_injection, postmask_pass_indices = indices_postmask, drop_indices = drop_indices)
                        else: 
                            original_image, generated_result, _, ce_loss = _generate_result_for_canvas(args, self.prompting_model, canvas, premask_pass_indices = indices_premask, postmask_pass_indices = indices_postmask, drop_indices = drop_indices)
                    else:
                        original_image, generated_result, _, ce_loss = _generate_result_for_canvas(args, self.prompting_model, canvas, attention_heads=indices, attention_injection=current_injection)
                    
                
                """ if args.task == 0:
                    loss = -1*self.loss_iou(generated_result, original_image)
                else:
                    if args.task is None:
                        if i%len(self.task_tensor) == 0:
                            loss = -1*self.loss_iou(generated_result, original_image)
                        else:
                            loss = self.loss_mse(original_image, generated_result)
                    else:
                        loss = self.loss_mse(original_image, generated_result) """
                
                loss = ce_loss
                loss_list.append(loss)

        loss_list = torch.tensor(loss_list)
        loss_list_copy = torch.tensor(loss_list)

        #if args.task is None:
        #    for index in range(len(self.task_tensor)):
        #        current = loss_list[index::len(self.task_tensor)]
        #        loss_list[index::len(self.task_tensor)] = (current-current.mean())/(current.std() + self.eps)
        #else:
            #loss_list = (loss_list-loss_list.mean())/(loss_list.std() + self.eps)

        reshaped_tensor = loss_list.reshape(len(self.areas_to_check)+len(removal_holder), self.batch_size)
        averages = reshaped_tensor.mean(axis=1)
        
        best_index = torch.argsort(averages, descending=False)[0]

        best_loss = loss_list_copy.reshape(len(self.areas_to_check)+len(removal_holder), self.batch_size).mean(axis=1)[best_index]
        
        if best_index < len(self.areas_to_check):
            best_move = [self.areas_to_check[best_index]]
            print("Added:", best_move)
        else:
            best_move = [removal_holder[best_index-len(self.areas_to_check)]]
            print("Removed:", best_move)

        if best_move[0] is not None:
        
            if best_index < len(self.areas_to_check):
                self.bernoullis = self.bernoullis + best_move
            else:
                self.bernoullis.pop(self.bernoullis.index(best_move[0]))

        return best_loss, generated_result, self.areas_to_check, averages

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
        
        if args.granularity==1:
            self.areas_to_check = [[layer, head, quadrant] for head in range(0,16) for quadrant in [0,1] for layer in range(0,24)] + [[layer, head, quadrant] for head in range(0,16) for quadrant in [0,1,2] for layer in range(24,32)] + [None]
        if args.granularity==2:
            self.areas_to_check = [[layer, head] for head in range(0,16) for layer in range(0,32)] + [None]
        if args.granularity==3:
            self.areas_to_check = [[layer] for layer in range(0,32)] + [None]

        for i in trange(num_itr):
            
            best_loss, res, move, averages = self.forward(args, canvases, i)

            print("Iter:",i, "Loss:",best_loss, "Bernoullis:", len(self.bernoullis), "Move:", move)

            with open(os.path.join(args.output_dir,'log.txt'), 'a') as log:
                current_metric = {}
                current_metric["iter"] = i
                current_metric["train_loss"] = best_loss
                current_metric["move"] = move
                current_metric["all_areas"] = self.bernoullis
                current_metric["averages"] = averages.tolist()

                log.write(str(current_metric) + '\n')
            
            import pickle
            with open(os.path.join(args.output_dir, str(args.task)+"_averages.pkl"), "wb") as f:
                pickle.dump((averages.tolist(),move), f)

            eval_iou, eval_indices = self.eval(args, i)

            print("Iter:",i,"Eval:",eval_iou)

            with open(os.path.join(args.output_dir,'log.txt'), 'a') as log:
                current_metric = {}
                current_metric["iter"] = i
                current_metric["eval_loss"] = eval_iou
                current_metric["all_areas"] = self.bernoullis

                log.write(str(current_metric) + '\n')
                
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

                    original_image, generated_result, _, ce_loss = _generate_result_for_canvas(args, self.prompting_model, canvas, premask_pass_indices = indices_premask, attention_heads=indices, attention_injection=curr_injection, postmask_pass_indices = indices_postmask, drop_indices = drop_indices)

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
                    original_image, generated_result, _, ce_loss = _generate_result_for_canvas(args, self.prompting_model, canvas, attention_heads=indices, attention_injection=self.task_tensor)
                
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
        

    _, im_paste, _, latents, ce_loss = generate_image(canvas.unsqueeze(0).to(args.device), model, ids_shuffle.to(args.device),
                                    len_keep, device=args.device, premask_pass_indices = premask_pass_indices, postmask_pass_indices = postmask_pass_indices, attention_heads = attention_heads, attention_injection = attention_injection, record=False, drop_indices = drop_indices)

    canvas = torch.einsum('chw->hwc', canvas)
    canvas = torch.clip((canvas * imagenet_std + imagenet_mean) * 255, 0, 255).int()
    assert canvas.shape == im_paste.shape, (canvas.shape, im_paste.shape)
    return canvas, im_paste, latents, ce_loss

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
                         flipped_order=args.flip, purple=args.purple, query_support_list_file=args.query_support_list_file, iters=args.train_images, type="trn", task=args.task)
    
    eval_ds = rl_dataloader.DatasetPASCAL(args.base_dir, fold=args.split, image_transform=image_transform, mask_transform=mask_transform,
                         flipped_order=args.flip, purple=args.purple, query_support_list_file=args.query_support_list_file, iters=args.eval_iters, type="val", task=0 if args.task is None else args.task)
    
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
    elif args.granularity==2:
        params = 24*16*50+8*16*99
    elif args.granularity==3:
        params = 1

    rl_model = JointModel(args, model, params, ds, eval_ds, injection, args.load_model)
    rl_model = rl_model.to(args.device)

    
    rl_model.train(args, num_itr=1)

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



