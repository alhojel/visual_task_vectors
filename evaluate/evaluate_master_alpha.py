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
    parser.add_argument('--split', default=2, type=int)
    parser.add_argument('--purple', default=0, type=int)
    parser.add_argument('--flip', default=0, type=int)
    parser.add_argument('--save_images', default=None, type=int, help='Save images')
    parser.add_argument('--query_support_list_file', default=None, type=str, help='Directory of query support list file')
    parser.add_argument('--iters', default=1000, type=int)
    parser.add_argument('--store_latents', default=None, type=str, help='Where to store latents')
    parser.add_argument('--decoder_start', default=0, type=int)
    parser.add_argument('--decoder_end', default=30, type=int)
    parser.add_argument('--encoder_start', default=0, type=int)
    parser.add_argument('--encoder_end', default=100, type=int)
    parser.add_argument('--image_suffix', default=2, type=float)


    


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

    captions_subset = ["segmentation"]
    captions = ["segmentation", "colorization", "uncolor", "lowlight enhance", "inpaint single random", "inpaint double random"]

    with open('/home/ahojel/visual_prompting_vid/holder_dict.pkl', 'rb') as f:
        injection_master = pickle.load(f)

    
    with open('/home/ahojel/visual_prompting_vid/ranked_tokens.pkl', 'rb') as f:
        master = pickle.load(f)
            
    #decoder_ranked_task = master["grouped_metric"]["decoder_zero"]
    #decoder_ranked_task = [[25, 2], [26, 7], [27, 8], [28, 3], [30, 7], [24, 15], [26, 10], [24, 3], [28, 9], [24, 7], [29, 15], [26, 0], [30, 0], [25, 7], [30, 3], [29, 2], [29, 7], [28, 13], [31, 13], [29, 4], [27, 3], [25, 10], [30, 11], [31, 1], [30, 8], [24, 10], [30, 5], [27, 0], [30, 14], [25, 15], [30, 12], [26, 14], [27, 9], [24, 4], [31, 0], [30, 2], [25, 8], [29, 12], [27, 4], [31, 5], [25, 1], [27, 1], [31, 2], [31, 15], [31, 12], [24, 5], [27, 14], [31, 8], [28, 5], [26, 6], [25, 13], [31, 11], [31, 10], [28, 6], [25, 12], [28, 15], [29, 8], [25, 4], [27, 13], [31, 6], [27, 6], [28, 0], [25, 5], [31, 7], [31, 4], [31, 3], [28, 11], [29, 10], [28, 10], [30, 10], [25, 11], [25, 9], [28, 14], [25, 14], [25, 0], [28, 7], [31, 14], [24, 14], [30, 4], [24, 12], [27, 7], [26, 8], [27, 5], [28, 1], [28, 2], [24, 11], [29, 0], [24, 13], [29, 6], [31, 9], [28, 12], [29, 3], [24, 8], [26, 2], [29, 9], [27, 10], [27, 2], [28, 4], [24, 0], [30, 15], [27, 12], [26, 12], [25, 6], [29, 1], [26, 9], [29, 13], [30, 6], [29, 5], [27, 15], [30, 9], [24, 1], [29, 11], [29, 14], [28, 8], [30, 13], [24, 6], [24, 2], [27, 11], [25, 3], [26, 1], [26, 15], [26, 3], [24, 9], [30, 1], [26, 5], [26, 13], [26, 11], [26, 4]]
    #decoder_ranked_task = []#[[25, 2, a+1] for a in calculate_quadrant_indices(14, 14, 2)]# + [[0, 3, a+1] for a in calculate_quadrant_indices(14, 14, 1)+calculate_quadrant_indices(14, 14, 2)] + [[0, 15, a+1] for a in calculate_quadrant_indices(14, 14, 2)] + [[0, 4, a+1] for a in calculate_quadrant_indices(14, 14, 1)+calculate_quadrant_indices(14, 14, 2)]

    #for i in range(len(decoder_ranked_task)):
        #decoder_ranked_task[i][0] = decoder_ranked_task[i][0]+24

    #encoder_ranked_task = master["grouped_metric"]["encoder_zero"]
    #encoder_ranked_task = [[10, 2], [10, 12], [21, 2], [8, 3], [3, 9], [21, 15], [3, 11], [16, 8], [8, 9], [0, 0], [20, 9], [21, 8], [22, 11], [0, 10], [6, 0], [5, 14], [13, 15], [13, 6], [7, 7], [23, 13], [23, 8], [4, 4], [23, 14], [23, 15], [11, 11], [23, 0], [22, 9], [4, 1], [4, 15], [9, 15], [22, 0], [22, 14], [23, 3], [23, 5], [1, 9], [3, 6], [13, 1], [4, 12], [12, 2], [1, 10], [16, 13], [11, 13], [7, 0], [9, 14], [8, 2], [12, 5], [1, 13], [13, 13], [3, 10], [3, 15], [10, 14], [0, 14], [2, 5], [2, 8], [22, 8], [6, 11], [4, 6], [19, 10], [2, 12], [16, 6], [2, 6], [6, 4], [14, 15], [21, 6], [3, 7], [9, 12], [12, 13], [6, 2], [1, 5], [22, 13], [15, 5], [11, 1], [5, 2], [13, 3], [2, 10], [21, 14], [7, 12], [3, 1], [11, 8], [13, 12], [9, 8], [7, 14], [17, 14], [3, 2], [7, 1], [3, 14], [3, 8], [10, 0], [11, 5], [8, 1], [5, 10], [7, 2], [18, 15], [1, 1], [18, 6], [9, 13], [23, 10], [13, 0], [8, 4], [11, 3], [22, 12], [2, 2], [19, 6], [15, 3], [21, 11], [3, 3], [9, 9], [5, 13], [13, 8], [0, 9], [2, 0], [3, 13], [0, 4], [0, 5], [2, 7], [9, 2], [22, 2], [11, 15], [1, 12], [2, 4], [3, 4], [0, 8], [20, 11], [18, 11], [20, 12], [5, 3], [0, 3], [22, 10], [12, 9], [20, 2], [10, 6], [18, 3], [11, 14], [0, 11], [5, 8], [20, 1], [4, 5], [13, 11], [15, 1], [7, 8], [8, 0], [4, 0], [13, 9], [18, 9], [4, 2], [6, 7], [22, 1], [16, 9], [5, 4], [23, 12], [2, 13], [6, 1], [10, 1], [23, 7], [3, 5], [17, 5], [3, 0], [6, 5], [16, 1], [14, 8], [2, 9], [20, 15], [2, 3], [5, 6], [10, 9], [1, 3], [22, 6], [20, 8], [4, 9], [14, 5], [19, 0], [19, 14], [19, 1], [5, 15], [0, 13], [10, 7], [8, 8], [13, 7], [15, 12], [21, 7], [13, 2], [10, 3], [15, 0], [2, 15], [23, 2], [11, 0], [3, 12], [21, 1], [0, 2], [1, 7], [7, 6], [20, 3], [5, 5], [4, 10], [17, 0], [12, 0], [21, 12], [7, 13], [0, 1], [23, 6], [16, 5], [4, 11], [10, 15], [19, 3], [21, 10], [9, 1], [6, 9], [8, 14], [16, 7], [22, 5], [9, 4], [8, 6], [19, 11], [9, 10], [11, 7], [9, 11], [7, 3], [2, 14], [7, 4], [8, 12], [5, 7], [12, 10], [21, 3], [6, 12], [20, 10], [21, 0], [5, 0], [23, 9], [10, 13], [2, 1], [4, 8], [7, 10], [20, 14], [16, 2], [22, 4], [18, 13], [8, 13], [16, 10], [15, 13], [8, 11], [12, 12], [16, 11], [15, 11], [6, 13], [8, 7], [9, 7], [10, 5], [14, 0], [11, 10], [22, 3], [11, 2], [19, 5], [14, 7], [17, 15], [10, 10], [13, 14], [23, 4], [4, 3], [23, 1], [15, 15], [10, 4], [9, 3], [7, 9], [9, 0], [13, 10], [14, 2], [17, 9], [7, 5], [21, 9], [11, 6], [17, 7], [22, 7], [12, 8], [12, 3], [16, 12], [7, 11], [16, 15], [17, 12], [10, 8], [15, 6], [12, 14], [18, 4], [17, 8], [21, 4], [0, 12], [15, 9], [14, 12], [20, 5], [19, 4], [12, 7], [14, 1], [12, 4], [20, 6], [23, 11], [7, 15], [19, 13], [16, 4], [15, 7], [21, 13], [17, 3], [6, 14], [14, 11], [19, 2], [0, 15], [20, 4], [4, 7], [12, 15], [9, 5], [14, 6], [18, 0], [12, 1], [4, 13], [18, 5], [14, 13], [17, 4], [4, 14], [18, 2], [19, 12], [22, 15], [11, 4], [5, 1], [10, 11], [14, 14], [13, 4], [14, 4], [16, 14], [8, 10], [9, 6], [12, 11], [6, 8], [15, 10], [12, 6], [6, 15], [1, 2], [1, 6], [1, 0], [20, 7], [8, 15], [19, 7], [19, 15], [14, 10], [11, 9], [17, 6], [0, 6], [1, 14], [5, 11], [21, 5], [16, 3], [13, 5], [20, 13], [19, 8], [17, 2], [14, 3], [2, 11], [1, 4], [18, 10], [15, 8], [20, 0], [17, 13], [18, 14], [6, 3], [6, 10], [16, 0], [17, 11], [14, 9], [11, 12], [17, 10], [15, 14], [17, 1], [8, 5], [18, 8], [15, 4], [1, 8], [1, 11], [15, 2], [18, 7], [18, 12], [18, 1], [6, 6], [5, 12], [19, 9], [5, 9], [0, 7], [1, 15]]
    #encoder_ranked_task = []#[[23, 13, 0]]
    decoder_start = args.decoder_start
    decoder_end = args.decoder_end
    decoder_step = decoder_end//10

    encoder_start = args.encoder_start
    encoder_end = args.encoder_end
    encoder_step = encoder_end//10

    
    #decoder_group = [decoder_ranked_task[:a] for a in range(decoder_start, decoder_end,decoder_step)] #+ [decoder_ranked_neutral[:a] for a in range(decoder_start, decoder_end,10)]
    
    #encoder_group = [encoder_ranked_task[:a] for a in range(encoder_start, encoder_end,encoder_step)] #+ [encoder_ranked_neutral[:a] for a in range(encoder_start, encoder_end,10)] + [encoder_mid]

    #target_array = [[encoder_group[a]+decoder_group[b]] for a in range(len(encoder_group)) for b in range(len(decoder_group))]

    # component_lists = [
    #     [],
    #     [[23, 13, 0]],
    #     [[25, 2, a+1] for a in calculate_quadrant_indices(14, 14, 2)],
    #     [[25, 10, a+1] for a in calculate_quadrant_indices(14, 14, 2)],
    #     [[26, 7, a+1] for a in calculate_quadrant_indices(14, 14, 2)],
    #     [[24, 15, a+1] for a in calculate_quadrant_indices(14, 14, 2)],
    #     [[25, 0, a+1] for a in calculate_quadrant_indices(14, 14, 2)],
    #     [[25, 0, a+1] for a in calculate_quadrant_indices(14, 14, 1)],
    #     [[25, 2, a+1] for a in calculate_quadrant_indices(14, 14, 1)],
    #     [[26, 7, a+1] for a in calculate_quadrant_indices(14, 14, 1)],
    #     [[29, 15, a+1] for a in calculate_quadrant_indices(14, 14, 1)],
    #     [[4, 1, a] for a in range(1,50)],
    #     [[4, 4, a] for a in range(1,50)],
    #     [[5, 14, a] for a in range(1,50)],
    #     [[10, 12, a] for a in range(1,50)],
    #     [[4, 6, a] for a in range(1,50)],
    #     [[6, 0, a] for a in range(1,50)],
    #     [[6, 11, a] for a in range(1,50)],
    #     [[3, 10, a] for a in range(1,50)],
    #     [[21, 15, a] for a in range(1,50)],
    #     [[9, 2, a] for a in range(1,50)],
    #     [[9, 10, a] for a in range(1,50)],
    #     [[10, 2, a] for a in range(1,50)],
    #     [[16, 8, a] for a in range(1,50)],
    #     [[5, 4, 0]],
    #     [[31, 7, a+1] for a in calculate_quadrant_indices(14, 14, 2)],
    #     [[24, 6, a+1] for a in calculate_quadrant_indices(14, 14, 1)],
    #     [[30, 7, a+1] for a in calculate_quadrant_indices(14, 14, 1)],
    #     [[30, 7, a+1] for a in calculate_quadrant_indices(14, 14, 2)],
    #     [[33, 3, a+1] for a in calculate_quadrant_indices(14, 14, 1)],
    #     [[33, 3, a+1] for a in calculate_quadrant_indices(14, 14, 2)],
    #     [[21, 2, a+1] for a in range(1,50)],
    # ]
   
    component_lists = [
        [],
        [[23, 13, 0]],
        [[5, 4, 0]],
        [[5, 14, a] for a in range(1,50)],
        [[6, 0, a] for a in range(1,50)],
        [[10, 12, a] for a in range(1,50)],
        [[21, 15, a] for a in range(1,50)],
        [[25, 2, a+1] for a in calculate_quadrant_indices(14, 14, 1)],
        [[25, 2, a+1] for a in calculate_quadrant_indices(14, 14, 2)],
        #[[30, 7, a+1] for a in calculate_quadrant_indices(14, 14, 1)],
        #[[30, 7, a+1] for a in calculate_quadrant_indices(14, 14, 2)],
        #[[27, 8, a+1] for a in calculate_quadrant_indices(14, 14, 2)],
        #[[29, 15, a+1] for a in calculate_quadrant_indices(14, 14, 2)],
        #[[25, 0, a+1] for a in calculate_quadrant_indices(14, 14, 2)],
        #[[24, 6, a+1] for a in calculate_quadrant_indices(14, 14, 2)],
    ]

    labels = ["None", "23-13-CLS", "5-4-CLS", "5-14-Q", "6-0-Q", "10-12-Q", "21-15-Q", "25-2-Q", "25-2-M"]#,  "0-6-M"]#,"25-0-M"]#, ]
    #labels = ["None", "+ 23-13-CLS", "+ 25-2-M", "+ 25-10-M", "+ 26-7-M", "+ 24-15-M", "+ 25-0-M", "+ 25-0-Q", "+ 25-2-Q", "+ 26-7-Q", "+ 29-15-Q", "+ 4-1-Q", "+ 4-4-Q", "+ 5-14-Q", "+ 10-12-Q", "+ 4-6-Q", "+ 6-0-Q", "+ 6-11-Q", "+ 3-10-Q", "+ 21-15-Q", "+ 9-2-Q", "+ 9-10-Q", "+ 10-2-Q", "+ 16-8-Q", "+ 5-4-CLS", "+ 31-7-M",  "+ 24-6-Q", "+ 30-7-Q",  "+ 30-7-M", "+ 33-3-Q",  "+ 33-3-M", "+ 21-2-Q"]

    target_array = []

    for i, _ in enumerate(component_lists):
        combined_components = [component for j, component in enumerate(component_lists) if j != i]
        aggregated_components = []
        for component in combined_components:
            aggregated_components += component
        target_array.append([aggregated_components])

        #= ["e_"+str(a) for a in range(encoder_start, encoder_end,encoder_step)] #+ ["encoder_neutral_"+str(a) for a in range(encoder_start, encoder_end,10)] + ["encoder_mid"]
    #decoder_labels = ["d_"+str(a) for a in range(decoder_start, decoder_end, decoder_step)] #+ ["decoder_neutral_"+str(a) for a in range(decoder_start, decoder_end,10)] 
    #encoder_labels = ["e_"+str(a) for a in range(encoder_start, encoder_end, encoder_step)] #+ ["decoder_neutral_"+str(a) for a in range(decoder_start, decoder_end,10)] 


    #target_array = [target_array[0]]
    
    for idx in trange(len(ds)):
        canvas = ds[idx]['grid']

        query_name = ds[idx]['query_name']
        support_name = ds[idx]['support_name']

        
        for i in range(len(canvas)):

            vector_type = ["segmentation","segmentation_1","segmentation_2","segmentation_3","segmentation_neutral"][-1]#[int(args.image_suffix)]
        
            if captions[i] not in captions_subset:
                continue

            gen_holder = []
            label_holder = []

            curr_canvas = (canvas[i] - imagenet_mean[:, None, None]) / imagenet_std[:, None, None]
            original_image, generated_result, latents = _generate_result_for_canvas(args, model, curr_canvas)

            original_image = np.uint8(torch.clip(torch.einsum('chw->hwc', canvas[i]) * 255, 0, 255).int().numpy())

            gen_holder.append(original_image)
            label_holder.append("Ground Truth")

            #original_image, generated_result, latents = _generate_result_for_canvas(args, model, curr_canvas)
            
            copy_canvas = curr_canvas.clone()
            midpoint = copy_canvas.shape[2] // 2
            left_half = copy_canvas[:, :, :midpoint]
            copy_canvas[:, :, midpoint:] = left_half

            original_image, generated_result, latents = _generate_result_for_canvas(args, model, copy_canvas)
            
            gen_holder.append(generated_result)
            label_holder.append("Actual Copy")


            
            print("done")
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
                
            
            #for index in range(len(target_array)):

            index = 0

            for alpha in [0,0.9,1,1.2,1.35,1.5,1.6,1.7,2,3]:    

                enc_inj = alpha*torch.tensor(injection_master["encoder"][vector_type]).to(args.device)
                dec_inj = alpha*torch.tensor(injection_master["decoder"][vector_type]).to(args.device)
                        
                injection = [enc_inj,dec_inj]
                
                #curr_array = []
                #for sss in target_array[index][0]:
                #    curr_array.extend(sss)

                #injection[0] = torch.zeros_like(injection[0])
                #injection[1] = torch.zeros_like(injection[1])

                indices_premask = []
                indices_postmask = []

                drop_indices = [1+a for a in calculate_quadrant_indices(14, 14, 1)]+[1+a for a in calculate_quadrant_indices(14, 14, 2)]
                
                indices_postmask += [1+a for a in calculate_quadrant_indices(14, 14, 3)] + [1+a for a in calculate_quadrant_indices(14, 14, 4)]
                #indices_premask = list(range(0,148))#
                indices_premask = [0]+list(range(99,148))
                
                #import pdb; breakpoint()
                if target_array[0][0] is None:
                    og2, gen2, latents = _generate_result_for_canvas(args, model, copy_canvas, premask_pass_indices = indices_premask, postmask_pass_indices = indices_postmask, drop_indices = drop_indices)#
                else:
                    og2, gen2, latents = _generate_result_for_canvas(args, model, copy_canvas, premask_pass_indices = indices_premask, attention_heads=target_array[index][0], attention_injection = injection, postmask_pass_indices = indices_postmask, drop_indices = drop_indices)#
                
                with open(os.path.join(args.output_dir, 'log.txt'), 'a') as log:
                    current_metric = {}
                    current_metric["query_name"] = query_name
                    current_metric["support_name"] = support_name
                    current_metric["baseline"] = "False"
                    current_metric["targets"] = str(labels[index])+str(alpha)
                    current_metric["vector_type"] = vector_type
                    #\current_metric["coordinates"] = target_array[index]
                    current_metric["task"] = captions[i]
                    current_metric["metric"] = evaluate_mse(original_image, gen2, args)["mse"]
                    if i == 0:
                        h = evaluate_segmentation(original_image, gen2, args)
                        #h = evaluate_segmentation(original_image, gen2, args)
                        current_metric["iou"] = h["iou"]
                        current_metric["accuracy"] = h["accuracy"]
                
                    log.write(str(current_metric) + '\n')

                gen_holder.append(gen2)
                label_holder.append(labels[index]+"\n"+str(alpha))
                
            if args.output_dir and args.save_images is not None and idx % args.save_images == 0:
                    
                gen_holder = [np.array(img) for img in gen_holder]

                # Determine the number of images
                num_images = len(gen_holder)
                # Calculate rows and columns to display all images in scan line order
                num_cols = 5  # Number of columns can be adjusted as needed
                num_rows = (num_images + num_cols - 1) // num_cols  # Calculate rows needed

                fig_size = (num_cols * 2, num_rows * 2)
                fig, axs = plt.subplots(num_rows, num_cols, figsize=fig_size, squeeze=False)
                
                # Displaying all images in scan line order
                for ele in range(num_images):
                    row = ele // num_cols
                    col = ele % num_cols
                    axs[row, col].imshow(gen_holder[ele])
                    axs[row, col].axis('off')
                    axs[row, col].set_title(label_holder[ele])
                
                # Hide any unused subplots
                for ele2 in range(ele + 1, num_rows * num_cols):
                    row = ele2 // num_cols
                    col = ele2 % num_cols
                    axs[row, col].axis('off')
                
                plt.tight_layout()
                plt.subplots_adjust(bottom=0.1)  # Adjust as needed
                plt.savefig(os.path.join(args.output_dir, f'combined_{idx}_{vector_type}.png'))
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
