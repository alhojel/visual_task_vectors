import os.path
from tqdm import trange
import multitask_dataloader
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
import h5py
import random
from collections import defaultdict
import numpy as np
import os
from contextlib import ExitStack
import torch
import torchvision.transforms as T
from PIL import Image
import pickle

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

def get_args():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--output_dir', default='../output_dir/')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--base_dir', default='/shared/yossi_gandelsman/code/occlusionwalk/pascal', help='pascal base dir')
    parser.add_argument('--seed', default=15, type=int)
    parser.add_argument('--t', default=[0, 0, 0], type=float, nargs='+')
    parser.add_argument('--ckpt', help='model checkpoint')
    parser.add_argument('--split', default=3 , type=int)
    parser.add_argument('--purple', default=0, type=int)
    parser.add_argument('--flip', default=0, type=int)
    parser.add_argument('--save_images', default=None, type=int, help='Save images')
    parser.add_argument('--query_support_list_file', default=None, type=str, help='Directory of query support list file')
    parser.add_argument('--iters', default=1000, type=int)
    parser.add_argument('--store_latents', default=None, type=str, help='Where to store latents')
    parser.add_argument('--rank_start', default=0, type=int)
    parser.add_argument('--rank_end', default=50000, type=int)
    parser.add_argument('--task', default=0, type=int)
    parser.add_argument('--area', default=0, type=int)
    parser.add_argument('--grouping', default=0, type=int)
    parser.add_argument('--random', default=0, type=int)

    return parser

def _generate_result_for_canvas(args, model, canvas, premask_pass_indices = None, postmask_pass_indices = None, attention_heads=None, attention_injection=None, drop_indices = None):
    """canvas is already in the right range."""

    ids_shuffle, len_keep = generate_mask_for_evaluation()
    if attention_heads is not None:
        attention_heads = torch.tensor(attention_heads).to(args.device)
        

    _, im_paste, _, latents = generate_image(canvas.unsqueeze(0).to(args.device), model, ids_shuffle.to(args.device),
                                    len_keep, device=args.device, premask_pass_indices = premask_pass_indices, postmask_pass_indices = postmask_pass_indices, attention_heads = attention_heads, attention_injection = attention_injection, record=False, drop_indices = drop_indices)

    canvas = torch.einsum('chw->hwc', canvas)
    canvas = torch.clip((canvas.cpu().detach() * imagenet_std + imagenet_mean) * 255, 0, 255).int().numpy()
    assert canvas.shape == im_paste.shape, (canvas.shape, im_paste.shape)
    return np.uint8(canvas), np.uint8(im_paste), latents

def evaluate(args):
    with open(os.path.join(args.output_dir, 'log.txt'), 'w') as log:
        log.write(str(args) + '\n')
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

    ds = multitask_dataloader.DatasetPASCAL(args.base_dir, fold=args.split, image_transform=image_transform, mask_transform=mask_transform,
                         flipped_order=args.flip, purple=args.purple, query_support_list_file=args.query_support_list_file, iters=args.iters)
    
    model = prepare_model(args.ckpt, arch=args.model)
    _ = model.to(args.device)

    
    captions_subset = ['segmentation', 'colorization',  "segmentation_neutral", 'lowlight enhance', 'inpaint single random'][args.task]

    with open('/home/ahojel/visual_prompting_vid/task_vectors.pkl', 'rb') as f:
        injection_master = pickle.load(f)

    
    with open('/home/ahojel/visual_prompting_vid/master_rankings.pkl', 'rb') as f:
        master = pickle.load(f)

    encoder_coordinates = master["encoder"]["coords"]
    decoder_coordinates = master["decoder"]["coords"]

    encoder_values = master["encoder"]["values"]
    decoder_values = master["decoder"]["values"]

    q1 = calculate_quadrant_indices(14, 14, 1)
    q2 = calculate_quadrant_indices(14, 14, 2)
    q3 = calculate_quadrant_indices(14, 14, 3)
    q4 = calculate_quadrant_indices(14, 14, 4)
    
    filtered_encoder_coordinates = []
    filtered_encoder_values = []
    for encoder_element in range(len(encoder_coordinates)):
        if encoder_coordinates[encoder_element][-1] >= 99:
            filtered_encoder_coordinates.append([encoder_coordinates[encoder_element][0], encoder_coordinates[encoder_element][1], encoder_coordinates[encoder_element][2]-98])
            filtered_encoder_values.append(encoder_values[encoder_element])
        elif encoder_coordinates[encoder_element][-1] == 0:
            filtered_encoder_coordinates.append([encoder_coordinates[encoder_element][0], encoder_coordinates[encoder_element][1], encoder_coordinates[encoder_element][2]])
            filtered_encoder_values.append(encoder_values[encoder_element])

    filtered_decoder_coordinates = []
    filtered_decoder_values = []

    for decoder_element in range(len(decoder_coordinates)):
        pos = decoder_coordinates[decoder_element][-1]
        if pos-1 in q3:
            filtered_decoder_coordinates.append([decoder_coordinates[decoder_element][0], decoder_coordinates[decoder_element][1], decoder_coordinates[decoder_element][2]-98])
            filtered_decoder_values.append(decoder_values[decoder_element])
        if pos-1 in q4:
            filtered_decoder_coordinates.append([decoder_coordinates[decoder_element][0], decoder_coordinates[decoder_element][1], decoder_coordinates[decoder_element][2]-98])
            filtered_decoder_values.append(decoder_values[decoder_element])
        elif pos == 0:
            filtered_decoder_coordinates.append([decoder_coordinates[decoder_element][0], decoder_coordinates[decoder_element][1], decoder_coordinates[decoder_element][2]])
            filtered_decoder_values.append(decoder_values[decoder_element])


    
    
    # Combine encoder and decoder coordinates and values, then sort them based on the values to get a global ranking
    
    groups_holder = {}
    for area_type in ["both"]:#, "decoder", "encoder"]:
        combined_coords_values = list(zip(encoder_coordinates + decoder_coordinates, encoder_values + decoder_values))
        #decoder_combined_coords_values = list(zip(decoder_coordinates, decoder_values))
        #encoder_combined_coords_values = list(zip(encoder_coordinates, encoder_values))
    
        if args.random:
            random.shuffle(combined_coords_values)
        else:
            #encoder_coords_headrank = rank_coordinates_zipped(encoder_combined_coords_values)
            #encoder_coords_quadrank = rank_coordinates_fine_grained(encoder_combined_coords_values, q1, q2, q3, q4)
            #encoder_combined_coords_values.sort(key=lambda x: x[1], reverse=True)  # Assuming higher values have higher ranks


            #decoder_coords_headrank = rank_coordinates_zipped(decoder_combined_coords_values)
            #decoder_coords_quadrank = rank_coordinates_fine_grained(decoder_combined_coords_values, q1, q2, q3, q4)
            #decoder_combined_coords_values.sort(key=lambda x: x[1], reverse=True)  # Assuming higher values have higher ranks


            coords_headrank = rank_coordinates_zipped(combined_coords_values)
            coords_quadrank = rank_coordinates_fine_grained(combined_coords_values, q1, q2, q3, q4)
            combined_coords_values.sort(key=lambda x: x[1], reverse=True)

        
        
        if args.grouping == 0:
            #encoder_ranked_joint_group = [coord for coord, _ in encoder_combined_coords_values]  # Extract sorted coordinates
            #decoder_ranked_joint_group = [coord for coord, _ in decoder_combined_coords_values]  # Extract sorted coordinates
            ranked_joint_group = [coord for coord, _ in combined_coords_values]
            grouping="tokens"
        elif args.grouping == 1:
            #encoder_ranked_joint_group = encoder_coords_quadrank
            #decoder_ranked_joint_group = decoder_coords_quadrank
            ranked_joint_group = coords_quadrank
            grouping="quadrants"
        elif args.grouping == 2:
            #encoder_ranked_joint_group = encoder_coords_headrank
            #decoder_ranked_joint_group = decoder_coords_headrank
            ranked_joint_group = coords_headrank
            grouping="heads"

        #estep_size = (args.erank_end-args.erank_start)//10
        #dstep_size = (args.drank_end-args.drank_start)//10
        step_size = (args.rank_end-args.rank_start)//20

        groups_holder[area_type] = [ranked_joint_group[:a] for a in range(args.rank_start, args.rank_end, step_size)] #+ [encoder_ranked_neutral[:a] for a in range(encoder_start, encoder_end,10)] + [encoder_mid]

        #groups_holder[area_type] = [encoder_ranked_joint_group[:a]+decoder_ranked_joint_group[:b] for a in range(args.erank_start, args.erank_end, estep_size) for b in range(args.drank_start, args.drank_end, dstep_size)] #+ [encoder_ranked_neutral[:a] for a in range(encoder_start, encoder_end,10)] + [encoder_mid]
    
    for idx in trange(len(ds)):
        for area in ["both"]:#["decoder", "encoder", "both"]:
            args.area = area
            groups = groups_holder[area]
            canvas = ds[idx]['grid']

            query_name = ds[idx]['query_name']
            support_name = ds[idx]['support_name']

            
            i = args.task

            vector_type = captions_subset

            enc_inj = torch.tensor(injection_master["encoder"][vector_type]).to(args.device)
            dec_inj = torch.tensor(injection_master["decoder"][vector_type]).to(args.device)
                    
            injection = [enc_inj,dec_inj]

            gen_holder = []
            label_holder = []

            curr_canvas = (canvas[i] - imagenet_mean[:, None, None]) / imagenet_std[:, None, None]

            copy_canvas = curr_canvas.clone()
            midpoint = copy_canvas.shape[2] // 2
            left_half = copy_canvas[:, :, :midpoint]
            copy_canvas[:, :, midpoint:] = left_half


            original_image, generated_result, latents = _generate_result_for_canvas(args, model, curr_canvas)

            original_image = np.uint8(torch.clip(torch.einsum('chw->hwc', (curr_canvas*imagenet_std[:, None, None])+imagenet_mean[:, None, None]) * 255, 0, 255).int().numpy())

            gen_holder.append(original_image)
            label_holder.append("Ground Truth")

            gen_holder.append(generated_result)
            label_holder.append("Actual Prompt")

            curr_canvas = copy_canvas

            og2, gen2, latents = _generate_result_for_canvas(args, model, curr_canvas)

            gen_holder.append(gen2)
            label_holder.append("BaselineSet")

            with open(os.path.join(args.output_dir, 'log.txt'), 'a') as log:
                current_metric = {}
                current_metric["query_name"] = query_name
                current_metric["support_name"] = support_name
                current_metric["baseline"] = "True"
                current_metric["task"] = vector_type
                current_metric["area"] = args.area
                current_metric["random"] = args.random
                current_metric["split"] = args.split
                current_metric["metric"] = evaluate_mse(original_image, generated_result, args)["mse"]
                if i == 0:
                    h = evaluate_segmentation(original_image, generated_result, args)
                    current_metric["iou"] = h["iou"]
                    current_metric["accuracy"] = h["accuracy"]
            
                log.write(str(current_metric) + '\n')

            
            for index in range(len(groups)):

                                    
                if groups[index] == []:
                    og2, gen2, latents = _generate_result_for_canvas(args, model, curr_canvas)
                else:

                    og2, gen2, latents = _generate_result_for_canvas(args, model, curr_canvas, attention_heads=groups[index], attention_injection = injection)#
                
                with open(os.path.join(args.output_dir, 'log.txt'), 'a') as log:
                    current_metric = {}
                    current_metric["query_name"] = query_name
                    current_metric["support_name"] = support_name
                    current_metric["baseline"] = "False"
                    #a_val = (index // ((args.erank_end - args.erank_start) // estep_size)) * estep_size + args.erank_start
                    #b_val = (index % ((args.drank_end - args.drank_start) // dstep_size)) * dstep_size + args.drank_start
                    #current_metric["k_encoder"] = a_val
                    #current_metric["k_decoder"] = b_val
                    current_metric["k"] = len(groups[index])
                    current_metric["task"] = vector_type
                    current_metric["split"] = args.split
                    current_metric["random"] = args.random
                    current_metric["area"] = args.area
                    current_metric["grouping"] = grouping
                    current_metric["metric"] = evaluate_mse(original_image, gen2, args)["mse"]
                    if i == 0:
                        h = evaluate_segmentation(original_image, gen2, args)
                        #h = evaluate_segmentation(original_image, gen2, args)
                        current_metric["iou"] = h["iou"]
                        current_metric["accuracy"] = h["accuracy"]
                
                    log.write(str(current_metric) + '\n')

                gen_holder.append(gen2)
                #label_holder.append(str(a_val)+","+str(b_val))
                label_holder.append(str(len(groups[index])))

            
            if args.output_dir and args.save_images is not None and idx % args.save_images == 0:
                    
                gen_holder = [np.array(img) for img in gen_holder]

                # Determine the number of images
                num_images = len(gen_holder)
                # Adjusting rows and columns to accommodate the images in rows
                
                num_cols = 10  # Displaying two images per row
                num_rows = (num_images // num_cols) if num_images % num_cols == 0 else (num_images // num_cols) + 1

                fig_size = (num_cols * 2, num_rows * 2)
                fig, axs = plt.subplots(num_rows, num_cols, figsize=fig_size, squeeze=False)
                
                # Displaying images in rows
                for img_index in range(num_images):
                    row = img_index // num_cols
                    col = img_index % num_cols
                    axs[row, col].imshow(gen_holder[img_index])
                    axs[row, col].axis('off')
                    axs[row, col].set_title(label_holder[img_index])

                # Hide any unused subplots
                for ax_index in range(img_index + 1, num_rows * num_cols):
                    row = ax_index // num_cols
                    col = ax_index % num_cols
                    axs[row, col].axis('off')

                plt.tight_layout()
                # # Displaying the first two images on their own row
                # for c in range(2):
                #     if c < num_images:  # Check to avoid index out of range
                #         axs[0, c].imshow(gen_holder[c])
                #         axs[0, c].axis('off')
                #         axs[0, c].set_title(label_holder[c])
                
                # # Displaying the rest of the images with the actual structure
                # for r in range(num_rows):
                #     for c in range(num_cols):
                #         index = (r-1) * num_cols + c  # Adjusting index to skip the first two images
                #         if index < num_images:  # Check to avoid index out of range
                #             axs[r, c].imshow(gen_holder[index])
                #             axs[r, c].axis('off')
                #             axs[r, c].set_title(label_holder[index])
                #         else:
                #             axs[r, c].axis('off')  # Hide axis if no image

                # Displaying all images in one row
                for c in range(num_cols):
                    if c < num_images:  # Check to avoid index out of range
                        axs[0, c].imshow(gen_holder[c])
                        axs[0, c].axis('off')
                        axs[0, c].set_title(label_holder[c])
                    else:
                        axs[0, c].axis('off') 

                plt.tight_layout()
                plt.subplots_adjust(bottom=0.1)  # Adjust as needed
                plt.savefig(os.path.join(args.output_dir, f'combined_{args.split}_{vector_type}_{area}_{args.grouping}_{args.random}_{idx}.png'))
                plt.show() 


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
    
def evaluate_mse(target, ours, args):
    ours = (np.transpose(ours/255., [2, 0, 1]) - imagenet_mean[:, None, None]) / imagenet_std[:, None, None]
    target = (np.transpose(target/255., [2, 0, 1]) - imagenet_mean[:, None, None]) / imagenet_std[:, None, None]

    target = target[:, 113:, 113:]
    ours = ours[:, 113:, 113:]
    mse = np.mean((target - ours)**2)
    return {'mse': mse}

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



if __name__ == '__main__':
    args = get_args()

    args = args.parse_args()
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    evaluate(args)
