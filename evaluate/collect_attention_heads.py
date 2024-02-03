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
    parser.add_argument('--split', default=0, type=int)
    parser.add_argument('--purple', default=0, type=int)
    parser.add_argument('--flip', default=0, type=int)
    parser.add_argument('--save_images', default=None, type=int, help='Save images')
    parser.add_argument('--query_support_list_file', default=None, type=str, help='Directory of query support list file')
    parser.add_argument('--iters', default=1000, type=int)
    parser.add_argument('--store_latents', default=None, type=str, help='Where to store latents')


    return parser

def write_latent(file_path, pass_id, latent, label):
    """
    Writes latent data, a string label, and a metric to an HDF5 file for a specific pass.
    Creates the file and/or groups if they don't exist.

    :param file_path: Path to the HDF5 file
    :param pass_id: Identifier for the forward pass
    :param latent: Latent data as a PyTorch tensor
    :param label: Label for the forward pass as a string
    :param metric: Numerical metric associated with the pass
    """
    # Ensure the latent data is a NumPy array
    if isinstance(latent, torch.Tensor):
        latent = latent.cpu().numpy()


    file_path += '.hdf5'

    # Open or create the HDF5 file
    with h5py.File(file_path, 'a') as h5file:
        # Create or get group for pass
        pass_group = h5file.require_group(f'pass_{pass_id}')

        # Create a subgroup for the label
        label_group = pass_group.require_group(f'{label}')

        # Create datasets for latent and metric within the label group
        label_group.create_dataset('encoder_latent', data=np.array(latent[:24]), compression="gzip", compression_opts=2)
        label_group.create_dataset('decoder_latent', data=np.array(latent[24:]), compression="gzip", compression_opts=2)



def _generate_result_for_canvas(args, model, canvas, premask_pass_indices = None, postmask_pass_indices = None, attention_heads=False):
    """canvas is already in the right range."""

    ids_shuffle, len_keep = generate_mask_for_evaluation()
    _, im_paste, _, latents = generate_image(canvas.unsqueeze(0).to(args.device), model, ids_shuffle.to(args.device),
                                    len_keep, device=args.device, attention_heads = attention_heads)

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

    captions_subset = ["segmentation", "colorization", "lowlight enhance", "inpaint single random"]
    captions = ["segmentation", "colorization", "uncolor", "lowlight enhance", "inpaint single random", "inpaint double random"]

    for idx in trange(len(ds)):
        canvas = ds[idx]['grid']

        query_name = ds[idx]['query_name']
        support_name = ds[idx]['support_name']

        
        for i in range(len(canvas)):

            if captions[i] not in captions_subset:
                continue

            curr_canvas = (canvas[i] - imagenet_mean[:, None, None]) / imagenet_std[:, None, None]
            original_image, generated_result, latents = _generate_result_for_canvas(args, model, curr_canvas, attention_heads=True)

            if args.store_latents:
                try:
                    write_latent(args.store_latents, f'{query_name}___{support_name}', latents, captions[i])
                except Exception as e:
                    print(f"Failed to write latent for {query_name}___{support_name}. Error: {e}")
            

            curr_canvas = canvas[i].clone().detach()
            midpoint = curr_canvas.shape[2] // 2
            left_half = curr_canvas[:, :, :midpoint]
            curr_canvas[:, :, midpoint:] = left_half

            og2, gen2, latents = _generate_result_for_canvas(args, model, curr_canvas)

            
            if args.store_latents:
                try:
                    write_latent(args.store_latents, f'{query_name}___{support_name}', latents, captions[i]+"_neutral")
                except Exception as e:
                    print(f"Failed to write latent for {query_name}___{support_name}. Error: {e}")
            
if __name__ == '__main__':
    args = get_args()

    args = args.parse_args()
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    evaluate(args)
