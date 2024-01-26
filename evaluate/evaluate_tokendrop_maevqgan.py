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
import pickle

import numpy as np
import os
from contextlib import ExitStack
import torch
import torchvision.transforms as T
from PIL import Image

from improv.pipelines.pipeline_improv import  IMProvPipeline

best_prompts = ["Left - input image, Right - Black and white foreground/background segmentation", "Left - input image, Right - Visually accurate colorization", None, "Left - input image, Right - Visually accurate light increase",  "Left - input image, Right - Inpainted image", None]
nc_best_prompt = "Left - input image, Right - unchanged image"

def get_args():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--output_dir', default='../output_dir/')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--base_dir', default='/shared/yossi_gandelsman/code/occlusionwalk/pascal', help='pascal base dir')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--t', default=[0, 0, 0], type=float, nargs='+')
    parser.add_argument('--ckpt', help='model checkpoint')
    parser.add_argument('--split', default=0, type=int)
    parser.add_argument('--purple', default=0, type=int)
    parser.add_argument('--flip', default=0, type=int)
    parser.add_argument('--save_images', default=None, type=int, help='Save images')
    parser.add_argument('--query_support_list_file', default=None, type=str, help='Directory of query support list file')
    parser.add_argument('--iters', default=1000, type=int)
    parser.add_argument('--isolated_layer', default=0, type=int)
    parser.add_argument('--task_vector', default="/home/ahojel/visual_prompting_vid/vectors/baseline.pkl", type=str, help='What task vector to use')
    parser.add_argument('--avoid_list', default=None, type=str, help='Directory of query support list file to avoid')



    return parser



def _generate_result_for_canvas(args, model, canvas, encoder_task_vector=None, decoder_task_vector=None, convex="False", drop_indices=None):
    """canvas is already in the right range."""
    
    ids_shuffle, len_keep = generate_mask_for_evaluation()
    if decoder_task_vector is not None:
        _, im_paste, _, latents = generate_image(canvas.unsqueeze(0).to(args.device), model, ids_shuffle.to(args.device),
                                    len_keep = len_keep, e_vec = None, d_vec = decoder_task_vector.to(args.device), device=args.device, convex=convex)
    else:
        _, im_paste, _, latents = generate_image(canvas.unsqueeze(0).to(args.device), model, ids_shuffle.to(args.device),
                                    len_keep = len_keep, e_vec = None, d_vec = None, device=args.device, drop_indices=drop_indices, convex=convex)
    canvas = torch.einsum('chw->hwc', canvas)
    canvas = torch.clip((canvas.cpu().detach() * imagenet_std + imagenet_mean) * 255, 0, 255).int().numpy()
    assert canvas.shape == im_paste.shape, (canvas.shape, im_paste.shape)
    return np.uint8(canvas), np.uint8(im_paste), latents
    
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

    ds = multitask_dataloader.DatasetPASCAL(args.base_dir, fold=args.split, image_transform=image_transform, mask_transform=mask_transform,
                         flipped_order=args.flip, purple=args.purple, query_support_list_file=args.query_support_list_file, iters=args.iters, avoid_list=args.avoid_list)
    
    
    model = prepare_model(args.ckpt, arch=args.model)
        
    _ = model.to(args.device)

    captions = ['label_segmentation', 'label_colorization', "label_uncolor", "label_lowlight enhance", 'label_inpaint single random', 'label_inpaint double random']

    for idx in trange(len(ds)):
        canvas = ds[idx]['grid']

        query_name = ds[idx]['query_name']
        support_name = ds[idx]['support_name']
        
        for i in range(len(canvas)):
            if i != 0:
                continue
            if captions[i] == "label_uncolor" or captions[i] == 'label_inpaint double random':
                continue

            gen_holder = []
            og_holder = []
            label_holder = []
            metric_holder = []

            original_image = np.uint8(torch.clip(torch.einsum('chw->hwc', canvas[i]) * 255, 0, 255).int().numpy())

            og_holder.append(original_image)
            label_holder.append("Ground Truth")

            #Original prompt
            curr_canvas = (canvas[i] - imagenet_mean[:, None, None]) / imagenet_std[:, None, None]
            
            original_image, generated_result, latents = _generate_result_for_canvas(args, model, curr_canvas)

            og_holder.append(generated_result)
            label_holder.append("Actual Prompt")
                    
            
            with open(os.path.join(args.output_dir, 'log.txt'), 'a') as log:
                current_metric = {}
                current_metric["query_name"] = query_name
                current_metric["support_name"] = support_name
                current_metric["baseline"] = "True"
                current_metric["task"] = captions[i]
                current_metric["metric"] = evaluate_mse(original_image, generated_result, args)["mse"]
                if i == 0:
                    h = evaluate_segmentation(original_image, generated_result, args)
                    current_metric["iou"] = h["iou"]
                    current_metric["accuracy"] = h["accuracy"]
            
                log.write(str(current_metric) + '\n')
            

            curr_canvas = (canvas[i] - imagenet_mean[:, None, None]) / imagenet_std[:, None, None]

            for tokens in ["top_half", "top_left", "top_right", "bottom_left"]:

                indices=[]
            
                if tokens == "top_half":
                    indices = [1+a for a in calculate_quadrant_indices(14, 14, 1)]
                    indices.extend([1+a for a in calculate_quadrant_indices(14, 14, 2)])
                
                if tokens == "top_left":
                    indices = [1+a for a in calculate_quadrant_indices(14, 14, 1)]
                if tokens == "top_right":
                    indices = [1+a for a in calculate_quadrant_indices(14, 14, 2)]
                if tokens == "bottom_left":
                    indices = [1+a for a in calculate_quadrant_indices(14, 14, 3)]

                original_image, generated_result, latents = _generate_result_for_canvas(args, model, curr_canvas, drop_indices=indices)

                original_image = np.uint8(torch.clip(torch.einsum('chw->hwc', canvas[i]) * 255, 0, 255).int().numpy())

                gen_holder.append(generated_result)
                label = "Dropped: "+tokens
                label_holder.append(label)

                with open(os.path.join(args.output_dir, 'log.txt'), 'a') as log:
                    current_metric = {}
                    current_metric["query_name"] = query_name
                    current_metric["support_name"] = support_name
                    current_metric["task"] = captions[i]
                    current_metric["dropped_tokens"] = tokens
                    current_metric["metric"] = evaluate_mse(original_image, generated_result, args)["mse"]
                    if i == 0:
                        h = evaluate_segmentation(original_image, generated_result, args)
                        current_metric["iou"] = h["iou"]
                        current_metric["accuracy"] = h["accuracy"]
                        metric_holder.append((current_metric["metric"], h["iou"], label, generated_result))
                    else:
                        metric_holder.append((current_metric["metric"], label, generated_result))

                    current_metric["r_metric"] = evaluate_mse(og_holder[1], generated_result, args)["mse"]

                    if i == 0:
                        h = evaluate_segmentation(og_holder[1], generated_result, args)
                        current_metric["r_iou"] = h["iou"]
                        current_metric["r_accuracy"] = h["accuracy"]
                
                    log.write(str(current_metric) + '\n')

            if args.output_dir and args.save_images is not None and idx % args.save_images == 0:
                
                og_holder = [np.array(img) for img in og_holder]
                gen_holder = [np.array(img) for img in gen_holder]

                # Determine the number of images
                num_images = len(og_holder)+len(gen_holder)
                num_rows = 2
                num_cols = 1 + len(gen_holder)//num_rows  # Select num_rows so that all the images fit

                fig_size = (num_cols * 3, num_rows * 3)
                fig, axs = plt.subplots(num_rows, num_cols, figsize=fig_size)
                
                for c in range(len(og_holder)):
                    axs[c, 0].imshow(og_holder[c])
                    axs[c, 0].axis('off') 
                    axs[c, 0].set_title(label_holder[c])

                gen_counter = 0
                for r in range(num_rows):
                    for c in range(1, num_cols):
                        if gen_counter < len(gen_holder):
                            axs[r, c].imshow(gen_holder[gen_counter])
                            axs[r, c].axis('off') 
                            axs[r, c].set_title(label_holder[gen_counter + len(og_holder)])
                            gen_counter += 1

                plt.tight_layout()
                plt.subplots_adjust(bottom=0.1)  # Adjust as needed
                plt.savefig(os.path.join(args.output_dir, f'combined_{idx}_{captions[i]}_2.png'))
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

