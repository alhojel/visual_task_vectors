from tqdm import trange
import multitask_dataloader
from reasoning_dataloader import *
import torchvision
from mae_utils import *
import argparse
from pathlib import Path
from segmentation_utils import *
import numpy as np
import pickle
import torch


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
    parser.add_argument('--split', default=0, type=int)
    parser.add_argument('--purple', default=0, type=int)
    parser.add_argument('--flip', default=0, type=int)
    parser.add_argument('--num_collections', default=100, type=int)
    parser.add_argument('--save_images', default=None, type=int, help='Save images')
    parser.add_argument('--query_support_list_file', default=None, type=str, help='Directory of query support list file')
    parser.add_argument('--iters', default=1000, type=int)

    return parser

def write_latent(file_path, label, latent):
    file_path = file_path + '/' + label + '_mean_activations.pkl'

    current_activations_encoder = torch.stack(latent[:24])
    current_activations_decoder = torch.stack(latent[24:])

    try:
        with open(file_path, 'rb') as file:
            content = pickle.load(file)
            running_count = content[0]
            running_mean_activations_encoder = content[1]
            running_mean_activations_decoder = content[2]

        running_mean_activations_encoder = (running_mean_activations_encoder * running_count + current_activations_encoder) / (running_count + 1)
        running_mean_activations_decoder = (running_mean_activations_decoder * running_count + current_activations_decoder) / (running_count + 1)

    except FileNotFoundError:
        running_count = 0
        running_mean_activations_encoder = current_activations_encoder
        running_mean_activations_decoder = current_activations_decoder
    
    content = [running_count + 1, running_mean_activations_encoder, running_mean_activations_decoder]

    with open(file_path, 'wb') as file:
        pickle.dump(content, file)

def _generate_result_for_canvas(args, model, canvas, collect_activations=False):
    """canvas is already in the right range."""

    ids_shuffle, len_keep = generate_mask_for_evaluation()
    _, im_paste, _, latents, _ = generate_image(canvas.unsqueeze(0).to(args.device), model, ids_shuffle.to(args.device),
                                    len_keep, device=args.device, collect_activations = collect_activations)

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
    
    model = prepare_model(args.ckpt, arch=args.model)
    _ = model.to(args.device)
    tasks = ["segmentation", "lowlight_enhance", "identity", "inpaint", "colorization"]

    if not os.path.exists(os.path.join(args.output_dir, 'filtered_pairs.pkl')):

        query_pair_list = {}
        metric_list = {}

        for task in tasks:
            query_pair_list[task] = []
            metric_list[task] = []

        for split in [1,2,3]:
            ds = multitask_dataloader.DatasetPASCAL(args.base_dir, fold=split, image_transform=image_transform, mask_transform=mask_transform,
                         flipped_order=args.flip, purple=args.purple, iters=1000, type="trn")
            
            for idx in trange(len(ds)):
                canvas = ds[idx]['grid']
                q_name = ds[idx]['query_name']
                s_name = ds[idx]['support_name']

                for i in range(len(canvas)):

                    curr_canvas = (canvas[i] - imagenet_mean[:, None, None]) / imagenet_std[:, None, None]
                    original_image, generated_result, _ = _generate_result_for_canvas(args, model, curr_canvas, collect_activations=False)

                    if i == 0:
                        metric = iou(original_image, generated_result)
                    else:
                        metric = mse(original_image, generated_result)

                    query_pair_list[tasks[i]].append({'query_name':q_name, 'support_name':s_name})
                    metric_list[tasks[i]].append(metric)
        
        # Create a dictionary to store the task-specific data
        task_specific_data = {}
        for task_idx, task in enumerate(tasks):
            curr_pair_list = query_pair_list[tasks[task_idx]]
            curr_metric_list = metric_list[tasks[task_idx]]

            # Pair each metric with its corresponding pair and sort by metric (smallest first)
            paired_list = sorted(zip(curr_metric_list, curr_pair_list), key=lambda x: x[0])

            # Remove duplicates based on 'support_name', keeping the one with the best metric
            unique_support = {}
            for metric, pair in paired_list:
                support_name = pair['support_name']
                if support_name not in unique_support or unique_support[support_name][0] > metric:
                    unique_support[support_name] = (metric, pair)

            # Now remove duplicates based on 'query_name', again keeping the best metric
            unique_query = {}
            for metric, pair in unique_support.values():
                query_name = pair['query_name']
                if query_name not in unique_query or unique_query[query_name][0] > metric:
                    unique_query[query_name] = (metric, pair)

            # Extract the final list of pairs after all filtering
            final_pairs = [pair for _, pair in unique_query.values()]

            # Store the task-specific data
            task_specific_data[task] = {
                'query_pair_list': final_pairs,
                'metric_list': [pair[0] for pair in unique_query.values()]
            }

        # Save the task-specific data to a pickle file
        with open(os.path.join(args.output_dir, 'filtered_pairs.pkl'), 'wb') as f:
            pickle.dump(task_specific_data, f)

    with open(os.path.join(args.output_dir, 'filtered_pairs.pkl'), 'rb') as f:
        data = pickle.load(f)
    
    for task_idx, task in enumerate(tasks):

        query_pairs = data[task]["query_pair_list"]
        metrics = data[task]["metric_list"]
        ranked_pairs = sorted(zip(query_pairs, metrics), key=lambda x: x[1], reverse=True if task_idx==0 else False)
        top_query_pairs = [pair[0] for pair in ranked_pairs[:args.num_collections]]

        ds = multitask_dataloader.DatasetPASCAL(args.base_dir, fold=args.split, image_transform=image_transform, mask_transform=mask_transform,
                         flipped_order=args.flip, purple=args.purple, query_support_list=top_query_pairs, iters=args.iters, type="trn", task=task_idx)
        
        for idx in trange(len(ds)):
            canvas = ds[idx]['grid']

            curr_canvas = (canvas - imagenet_mean[:, None, None]) / imagenet_std[:, None, None]
            original_image, generated_result, latents = _generate_result_for_canvas(args, model, curr_canvas, collect_activations=True)
        
            write_latent(args.output_dir, tasks[task_idx], latents)


def mse(target, ours):
    ours = (torch.permute(ours / 255., (2, 0, 1)) - torch.tensor(imagenet_mean, dtype=torch.float32).to(ours.device)[:, None, None]) / torch.tensor(imagenet_std, dtype=torch.float32).to(ours.device)[:, None, None]
    target = (torch.permute(target.to(ours.device) / 255., (2, 0, 1)) - torch.tensor(imagenet_mean, dtype=torch.float32).to(ours.device)[:, None, None]) / torch.tensor(imagenet_std, dtype=torch.float32).to(ours.device)[:, None, None]

    target = target[:, 113:, 113:]
    ours = ours[:, 113:, 113:]
    mse = torch.mean((target - ours) ** 2)
    return mse.item()

def iou(original_image, generated_result):
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
if __name__ == '__main__':
    args = get_args()

    args = args.parse_args()
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    evaluate(args)