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
    parser.add_argument('--train_images', default=16, type=int)
    parser.add_argument('--random_train_images', default=0, type=int)
    parser.add_argument('--eval_iters', default=1000, type=int)
    parser.add_argument('--store_latents', default=None, type=str, help='Where to store latents')
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--init', default=-1.0, type=float)
    parser.add_argument('--zero_shot', default=0, type=int)
    parser.add_argument('--replace_tokens', default=0, type=int)
    parser.add_argument('--restrict_area', default=0, type=int)
    parser.add_argument('--task', default=None, type=int)
    parser.add_argument('--eval_task', default=0, type=int)
    parser.add_argument('--granularity', default=0, type=int)
    parser.add_argument('--regularization_strength', default=0, type=float)
    parser.add_argument('--setup', default="None", type=str)
    parser.add_argument('--load_model', default=None, type=str, help='Where to load model from')

    return parser


class JointModel(nn.Module):
    def __init__(self, args, prompting_model, num_variables, train_ds, eval_ds, task_tensor, load_model=None):
        super().__init__()
        self.prompting_model = prompting_model
        self.num_variables = num_variables
        self.bernoullis = [torch.tensor(args.init, requires_grad=True) for _ in range(self.num_variables)]
        self.optim = optim.Adam(self.bernoullis, lr=args.lr)
        self.eps = 1e-6
        self.batch_size = 64*5
        self.regularization_strength = args.regularization_strength

        self.train_ds = train_ds
        self.eval_ds = eval_ds

        self.task_tensor = task_tensor

        if load_model is not None:
            self.bernoullis = pickle.load(open(load_model, 'rb'))
            self.bernoullis = [torch.tensor(bernoulli, requires_grad=True) for bernoulli in self.bernoullis]

        if args.setup=="GT":
            self.bernoullis = [torch.tensor(-100.0, requires_grad=True) for bernoulli in self.bernoullis]

        if args.restrict_area==1:
            for i in range(24*16*2):
                self.bernoullis[i] = torch.tensor(-10.0, requires_grad=True)
        elif args.restrict_area==2:
            for i in range(24*16*2, len(self.bernoullis)):
                self.bernoullis[i] = torch.tensor(-10.0, requires_grad=True)


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
        
        if args.granularity==0:
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

        if args.granularity==1:
        
            indices = sampled_patches.nonzero()
            expanded_indices = []

            for element in indices:
                index = int(element.item())
                total_elements_per_layer_first_24_heads = 16 * 2  # 16 layers, 2 elements per layer
                total_elements_per_layer_next_8_heads = 16 * 3  # 16 layers, 3 elements per layer
                total_elements_first_24_heads = 24 * total_elements_per_layer_first_24_heads
                
                if index < total_elements_first_24_heads:
                    head = index // total_elements_per_layer_first_24_heads
                    layer = (index % total_elements_per_layer_first_24_heads) // 2
                    quadrant = index % 2
                else:
                    adjusted_index = index - total_elements_first_24_heads
                    head = 24 + (adjusted_index // total_elements_per_layer_next_8_heads)
                    layer = (adjusted_index % total_elements_per_layer_next_8_heads) // 3
                    quadrant = adjusted_index % 3
                
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
            #import pdb; breakpoint()
            indices = expanded_indices

            return indices
        
        if args.granularity==2:
        
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


            return indices
    
    def forward(self, args, canvases):

        sigmoid_tensor = torch.stack([torch.sigmoid(bernoulli).clamp(min=self.eps, max=1-self.eps) for bernoulli in self.bernoullis])
        sigmoid_tensor = sigmoid_tensor.unsqueeze(0).expand(self.batch_size, self.num_variables)
        prob_dist = torch.distributions.Bernoulli(sigmoid_tensor)
        sampled_patches = prob_dist.sample() # TODO: to reduce variance, sample multiple times?

        loss_list = []
        for i in range(0,self.batch_size):
            
            indices = self.construct_indices(sampled_patches[i])
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

                    original_image, generated_result, _ = _generate_result_for_canvas(args, self.prompting_model, canvas, premask_pass_indices = indices_premask, attention_heads=indices, attention_injection=current_injection, postmask_pass_indices = indices_postmask, drop_indices = drop_indices)
                else:
                    original_image, generated_result, _ = _generate_result_for_canvas(args, self.prompting_model, canvas, attention_heads=indices, attention_injection=current_injection)
                
                        # image = generated_result.detach().cpu().numpy()
                        # plt.figure()
                        # plt.imshow(image)
                        # plt.axis('off')  # Turn off axis numbers and ticks
                        # image_save_dir = os.path.join(args.output_dir, 'rip')
                        # if not os.path.exists(image_save_dir):
                        #     os.makedirs(image_save_dir)
                        # image_save_path = os.path.join(image_save_dir, f'train_{i}.png')
                        # plt.savefig(image_save_path, bbox_inches='tight', pad_inches=0)  # Save the image without padding and no axis
                        # plt.close()  # Close the plot to prevent it from displaying in the notebook or script output
            
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
        loss_printer = torch.tensor(loss_list).mean()
        log_prob = prob_dist.log_prob(sampled_patches).mean(-1)

        if args.task is  None:
            for index in range(len(self.task_tensor)):
                current = loss_list[index::len(self.task_tensor)]
                loss_list[index::len(self.task_tensor)] = (current-current.mean())/(current.std() + self.eps)

        minus_r = (loss_list - loss_list.mean())/(loss_list.std() + self.eps)
                
        minus_r = minus_r.detach()
        loss = (log_prob*minus_r).mean() + self.regularization_strength*torch.mean(torch.stack([torch.sigmoid(bernoulli).clamp(min=self.eps, max=1-self.eps) for bernoulli in self.bernoullis]))
        return loss, generated_result, loss_printer

    def run_eval(self, args):

        eval_iou, eval_indices = self.eval(args)

        with open(os.path.join(args.output_dir,'log.txt'), 'a') as log:
            current_metric = {}
            current_metric["eval_loss"] = eval_iou
            current_metric["eval_patch_count"] = eval_indices
            current_metric["granularity"] = args.granularity
            current_metric["task"] = args.task
            current_metric["split"] = args.split
            current_metric["load_model"] = args.load_model

            log.write(str(current_metric) + '\n')
                
        return None
    
    
    def eval(self, args):

        sigmoid_tensor = torch.stack([torch.sigmoid(bernoulli).clamp(min=self.eps, max=1-self.eps) for bernoulli in self.bernoullis])
        sigmoid_tensor = sigmoid_tensor

        if args.task is None:
            curr_injection = self.task_tensor[0]
        else:
            curr_injection = self.task_tensor
        
        loss_holder = []
        for idx in trange(len(self.eval_ds)):

            prob_dist = torch.distributions.Bernoulli(sigmoid_tensor)
            sampled_patches = prob_dist.sample() # TODO: to reduce variance, sample multiple times?

            indices = self.construct_indices(sampled_patches)

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

                    """ if idx%100==0:
                    
                        image = generated_result.detach().cpu().numpy()
                        plt.figure()
                        plt.imshow(image)
                        plt.axis('off')  # Turn off axis numbers and ticks
                        image_save_dir = os.path.join(args.output_dir, 'rip')
                        if not os.path.exists(image_save_dir):
                            os.makedirs(image_save_dir)
                        image_save_path = os.path.join(image_save_dir, f'train_{args.task}_{idx}.png')
                        plt.savefig(image_save_path, bbox_inches='tight', pad_inches=0)  # Save the image without padding and no axis
                        plt.close()  # Close the plot to prevent it from displaying in the notebook or script output


                        image = original_image.detach().cpu().numpy()
                        plt.figure()
                        plt.imshow(image)
                        plt.axis('off')  # Turn off axis numbers and ticks
                        image_save_dir = os.path.join(args.output_dir, 'rip')
                        if not os.path.exists(image_save_dir):
                            os.makedirs(image_save_dir)
                        image_save_path = os.path.join(image_save_dir, f'train_{args.task}_{idx}_original.png')
                        plt.savefig(image_save_path, bbox_inches='tight', pad_inches=0)  # Save the image without padding and no axis
                        plt.close()  # Close the plot to prevent it from displaying in the notebook or script output """
                 
                else:
                    original_image, generated_result, _ = _generate_result_for_canvas(args, self.prompting_model, canvas, attention_heads=indices, attention_injection=self.task_tensor)
                
            if args.eval_task is None:
                loss = self.loss_iou(original_image, generated_result).item()
            elif args.eval_task == 0 or args.eval_task == 6:
                loss = self.loss_iou(original_image, generated_result).item()
            else:
                loss = self.loss_mse(original_image, generated_result)
            loss_holder.append(loss)

            image = generated_result.detach().cpu().numpy()

            if args.setup=="GT":
                image = original_image.detach().cpu().numpy()
            
            plt.figure()
            plt.imshow(image)
            plt.axis('off')  # Turn off axis numbers and ticks
            image_save_dir = os.path.join(args.output_dir, 'rip')
            if not os.path.exists(image_save_dir):
                os.makedirs(image_save_dir)
            image_save_path = os.path.join(image_save_dir, f'{args.task}_{args.split}_{idx}_{args.setup}.png')
            plt.savefig(image_save_path, bbox_inches='tight', pad_inches=0)  # Save the image without padding and no axis
            plt.close()  # Close the plot to prevent it from displaying in the notebook or script output


            with open(os.path.join(args.output_dir,'log-images.txt'), 'a') as log:
                current_metric = {}
                current_metric["file_name"] = f"{args.task}_{args.split}_{idx}_{args.setup}"
                current_metric["task"] = args.task
                current_metric["split"] = args.split
                current_metric["idx"] = idx
                current_metric["setup"] = args.setup
                current_metric["metric"] = loss

                log.write(str(current_metric) + '\n')

        eval_mean_iou = np.mean(loss_holder)

        return eval_mean_iou, len(indices)


def _generate_result_for_canvas(args, model, canvas, premask_pass_indices = None, postmask_pass_indices = None, attention_heads=None, attention_injection=None, drop_indices = None):
    """canvas is already in the right range."""

    ids_shuffle, len_keep = generate_mask_for_evaluation()
    if attention_heads is not None:
        attention_heads = torch.tensor(attention_heads, dtype=torch.int64).to(args.device)
        

    _, im_paste, _, latents, celoss = generate_image(canvas.unsqueeze(0).to(args.device), model, ids_shuffle.to(args.device),
                                    len_keep, device=args.device, premask_pass_indices = premask_pass_indices, postmask_pass_indices = postmask_pass_indices, attention_heads = attention_heads, attention_injection = attention_injection, record=False, drop_indices = drop_indices, replace_tokens = True if args.replace_tokens else False)

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
    
    model = prepare_model(args.ckpt, arch=args.model)
    _ = model.to(args.device)

    with open('/home/ahojel/visual_prompting_vid/task_vectors.pkl', 'rb') as f:
        injection_master = pickle.load(f)

    tasks = ["segmentation", "lowlight enhance", "segmentation_neutral", "inpaint single random", "colorization", "inpaint_ll", "inpaint_seg"]

    if args.task is not None:
        task = tasks[args.task]

        if args.task==5:
            enc_inj = torch.tensor(injection_master["encoder"]["lowlight enhance"]).to(args.device) + torch.tensor(injection_master["encoder"]["inpaint single random"]).to(args.device) - torch.tensor(injection_master["encoder"]["segmentation_neutral"]).to(args.device)
            dec_inj = torch.tensor(injection_master["decoder"]["lowlight enhance"]).to(args.device) + torch.tensor(injection_master["decoder"]["inpaint single random"]).to(args.device) - torch.tensor(injection_master["decoder"]["segmentation_neutral"]).to(args.device)
        elif args.task==6:
            enc_inj = torch.tensor(injection_master["encoder"]["segmentation"]).to(args.device) + torch.tensor(injection_master["encoder"]["inpaint single random"]).to(args.device) - torch.tensor(injection_master["encoder"]["segmentation_neutral"]).to(args.device)
            dec_inj = torch.tensor(injection_master["decoder"]["segmentation"]).to(args.device) + torch.tensor(injection_master["decoder"]["inpaint single random"]).to(args.device) - torch.tensor(injection_master["decoder"]["segmentation_neutral"]).to(args.device)
        else:
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
    
    for split in [2]:
        args.split = split
        eval_ds = rl_dataloader.DatasetPASCAL(args.base_dir, fold=split, image_transform=image_transform, mask_transform=mask_transform,
                         flipped_order=args.flip, purple=args.purple, query_support_list_file=args.query_support_list_file, iters=args.eval_iters, type="val", task= args.task if args.task is not None else 0)

        rl_model = JointModel(args, model, params, ds, eval_ds, injection, args.load_model)
        rl_model = rl_model.to(args.device)

        rl_model.run_eval(args)

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

    args.eval_task = args.task
    evaluate(args)



