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
import itertools
import numpy as np
import os
from contextlib import ExitStack
import torch
import torchvision.transforms as T
from PIL import Image
import pickle


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
    parser.add_argument('--decoder_start', default=0, type=int)
    parser.add_argument('--decoder_end', default=4500, type=int)
    parser.add_argument('--encoder_start', default=0, type=int)
    parser.add_argument('--encoder_end', default=2800, type=int)
    parser.add_argument('--image_suffix', default=0, type=int)


    


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



def _generate_result_for_canvas(args, model, canvas, premask_pass_indices = None, postmask_pass_indices = None, attention_heads=None, attention_injection=None):
    """canvas is already in the right range."""

    ids_shuffle, len_keep = generate_mask_for_evaluation()
    if attention_heads is not None:
        attention_heads = torch.tensor(attention_heads).to(args.device)
        #import pdb; breakpoint()

    _, im_paste, _, latents = generate_image(canvas.unsqueeze(0).to(args.device), model, ids_shuffle.to(args.device),
                                    len_keep, device=args.device, attention_heads = attention_heads, attention_injection = attention_injection, record=False)

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

    captions_subset = ["segmentation"]
    captions = ["segmentation", "colorization", "uncolor", "lowlight enhance", "inpaint single random", "inpaint double random"]

    with open('/home/ahojel/visual_prompting_vid/holder_dict.pkl', 'rb') as f:
        injection = pickle.load(f)
            
    injeciton = [injection["encoder"]["segmentation"].to(args.device),injection["decoder"]["segmentation"].to(args.device)]

    
    with open('/home/ahojel/visual_prompting_vid/ranked_tokens.pkl', 'rb') as f:
        master = pickle.load(f)
            
    decoder_ranked_task = master["metric"]["decoder"]
    
    for i in range(len(decoder_ranked_task)):
        decoder_ranked_task[i][0] = decoder_ranked_task[i][0]+24

    encoder_ranked_task = master["metric"]["encoder"]

    decoder_start = args.decoder_start
    decoder_end = args.decoder_end
    decoder_step = decoder_end//10

    encoder_start = args.encoder_start
    encoder_end = args.encoder_end
    encoder_step = encoder_end//10
    
    decoder_group = [decoder_ranked_task[:a] for a in range(decoder_start, decoder_end,decoder_step)] #+ [decoder_ranked_neutral[:a] for a in range(decoder_start, decoder_end,10)]
    
    encoder_group = [encoder_ranked_task[:a] for a in range(encoder_start, encoder_end,encoder_step)] #+ [encoder_ranked_neutral[:a] for a in range(encoder_start, encoder_end,10)] + [encoder_mid]

    
    #import pdb; breakpoint()
    target_array = [[encoder_group[a]+decoder_group[b]] for a in range(len(encoder_group)) for b in range(len(decoder_group))]

    encoder_labels = ["e_"+str(a) for a in range(encoder_start, encoder_end,encoder_step)] #+ ["encoder_neutral_"+str(a) for a in range(encoder_start, encoder_end,10)] + ["encoder_mid"]
    decoder_labels = ["d_"+str(a) for a in range(decoder_start, decoder_end,decoder_step)] #+ ["decoder_neutral_"+str(a) for a in range(decoder_start, decoder_end,10)] 


    for idx in trange(len(ds)):
        canvas = ds[idx]['grid']

        query_name = ds[idx]['query_name']
        support_name = ds[idx]['support_name']

        
        for i in range(len(canvas)):

            if captions[i] not in captions_subset:
                continue

            gen_holder = []
            label_holder = []

            curr_canvas = (canvas[i] - imagenet_mean[:, None, None]) / imagenet_std[:, None, None]
            original_image, generated_result, latents = _generate_result_for_canvas(args, model, curr_canvas)

            original_image = np.uint8(torch.clip(torch.einsum('chw->hwc', canvas[i]) * 255, 0, 255).int().numpy())

            gen_holder.append(original_image)
            label_holder.append("Ground Truth")

            curr_canvas = (canvas[i] - imagenet_mean[:, None, None]) / imagenet_std[:, None, None]
            original_image, generated_result, latents = _generate_result_for_canvas(args, model, curr_canvas)

            gen_holder.append(generated_result)
            label_holder.append("Actual Prompt")

            copy_canvas = curr_canvas
            midpoint = copy_canvas.shape[2] // 2
            left_half = copy_canvas[:, :, :midpoint]
            copy_canvas[:, :, midpoint:] = left_half

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

            
            for encoder_index, decoder_index in itertools.product(range(len(encoder_group)), range(len(decoder_group))):
                index = encoder_index * len(decoder_group) + decoder_index
                #curr_array = []
                #for sss in target_array[index][0]:
                #    curr_array.extend(sss)
                
                og2, gen2, latents = _generate_result_for_canvas(args, model, copy_canvas, attention_heads=target_array[index][0], attention_injection = injeciton)
                
                with open(os.path.join(args.output_dir, 'log.txt'), 'a') as log:
                    current_metric = {}
                    current_metric["query_name"] = query_name
                    current_metric["support_name"] = support_name
                    current_metric["baseline"] = "False"
                    current_metric["vector_type"] = "metric"
                    current_metric["decoder_type"] = decoder_labels[decoder_index][2:]
                    current_metric["encoder_type"] = encoder_labels[encoder_index][2:]
                    #current_metric["coordinates"] = target_array[index]
                    current_metric["task"] = captions[i]
                    current_metric["metric"] = evaluate_mse(original_image, gen2, args)["mse"]
                    if i == 0:
                        h = evaluate_segmentation(original_image, gen2, args)
                        current_metric["iou"] = h["iou"]
                        current_metric["accuracy"] = h["accuracy"]
                
                    log.write(str(current_metric) + '\n')

                gen_holder.append(gen2)
                label_holder.append(encoder_labels[encoder_index]+"\n"+decoder_labels[decoder_index])
            
            if args.output_dir and args.save_images is not None and idx % args.save_images == 0:
                    
                gen_holder = [np.array(img) for img in gen_holder]

                # Determine the number of images
                num_images = len(gen_holder)
                # Adjusting rows and columns to accommodate the first two images on their own row
                num_rows = len(encoder_group) + 1  # Adding an extra row for the first two images
                num_cols = max(len(decoder_group), 2)  # Ensuring there are at least two columns

                fig_size = (num_cols * 2, num_rows * 2)
                fig, axs = plt.subplots(num_rows, num_cols, figsize=fig_size, squeeze=False)
                
                # Displaying the first two images on their own row
                for c in range(2):
                    if c < num_images:  # Check to avoid index out of range
                        axs[0, c].imshow(gen_holder[c])
                        axs[0, c].axis('off')
                        axs[0, c].set_title(label_holder[c])
                
                # Displaying the rest of the images with the actual structure
                for r in range(1, num_rows):
                    for c in range(num_cols):
                        index = (r-1) * num_cols + c + 2 # Adjusting index to skip the first two images
                        if index < num_images:  # Check to avoid index out of range
                            axs[r, c].imshow(gen_holder[index])
                            axs[r, c].axis('off')
                            axs[r, c].set_title(label_holder[index])
                        else:
                            axs[r, c].axis('off')  # Hide axis if no image
                plt.tight_layout()
                plt.subplots_adjust(bottom=0.1)  # Adjust as needed
                plt.savefig(os.path.join(args.output_dir, f'combined_{idx}_{captions[i]}_{args.image_suffix}.png'))
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

if __name__ == '__main__':
    args = get_args()

    args = args.parse_args()
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    evaluate(args)
