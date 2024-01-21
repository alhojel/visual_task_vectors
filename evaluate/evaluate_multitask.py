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
    parser.add_argument('--store_latents', default=None, type=str, help='Where to store latents')


    return parser

def write_latent(file_path, pass_id, latent, label, metric):
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
        label_group = pass_group.require_group(f'label_{label}')

        # Create datasets for latent and metric within the label group
        label_group.create_dataset('decoder_latent', data=np.array(latent), compression="gzip", compression_opts=2)

        label_group.create_dataset('metric', data=metric)


def _generate_result_for_canvas(args, model, canvas, input_prompts=""):
    """canvas is already in the right range."""
    if args.model != "improv":
        ids_shuffle, len_keep = generate_mask_for_evaluation()
        _, im_paste, _, latents = generate_image(canvas.unsqueeze(0).to(args.device), model, ids_shuffle.to(args.device),
                                        len_keep, device=args.device)
        canvas = torch.einsum('chw->hwc', canvas)
        canvas = torch.clip((canvas.cpu().detach() * imagenet_std + imagenet_mean) * 255, 0, 255).int().numpy()
        assert canvas.shape == im_paste.shape, (canvas.shape, im_paste.shape)
        return np.uint8(canvas), np.uint8(im_paste), latents
    else:
        input_mask = torch.zeros(1, 224, 224).to(args.device)
        input_mask[:, 113:224, 113:224] = 1

        generator = torch.Generator(device=model.device).manual_seed(42)
        init_image = canvas.unsqueeze(0).to(args.device)
        with ExitStack() as stack:
            stack.enter_context(torch.no_grad())

            

            raw_inpaint, latents = model(
                input_prompts,
                image=init_image,
                mask_image=input_mask,
                generator=generator,
                height=init_image.shape[-2],
                width=init_image.shape[-1],
                guidance_scale=1.0,
                num_inference_steps=1,
                choice_temperature=0.0,
                output_type="torch",
            )
            raw_inpaint = raw_inpaint.images

        im_paste = raw_inpaint * input_mask.unsqueeze(1) + init_image * (
            1 - input_mask.unsqueeze(1)
        )[0]

        canvas = torch.einsum('chw->hwc', canvas)
        im_paste = torch.einsum('chw->hwc', im_paste[0])
        im_paste = torch.clip((im_paste.cpu().detach()) * 255, 0, 255).int().numpy()
        canvas = torch.clip((canvas.cpu().detach()) * 255, 0, 255).int().numpy()
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
    
    if args.model == "improv":
        model = IMProvPipeline.from_pretrained(pretrained_model_name_or_path="xvjiarui/IMProv-v1-0")
        _ = model.to(args.device)
    else:
        model = prepare_model(args.ckpt, arch=args.model)
        
    _ = model.to(args.device)


    captions = ["segmentation", "colorization", "uncolor", "lowlight enhance", "inpaint single random", "inpaint double random"]
    
    for idx in trange(len(ds)):
        canvas = ds[idx]['grid']

        query_name = ds[idx]['query_name']
        support_name = ds[idx]['support_name']

        
        
        for i in range(len(canvas)):

            og_holder = []
            gen_holder = []

            
            curr_canvas = (canvas[i] - imagenet_mean[:, None, None]) / imagenet_std[:, None, None]

            if args.model == "improv":
                curr_canvas = canvas[i]

            original_image, generated_result, latents = _generate_result_for_canvas(args, model, curr_canvas)


            curr_canvas = canvas[i].clone().detach()
            midpoint = curr_canvas.shape[2] // 2
            left_half = curr_canvas[:, :, :midpoint]
            curr_canvas[:, :, midpoint:] = left_half

            if args.model != "improv":
                curr_canvas = (curr_canvas - imagenet_mean[:, None, None]) / imagenet_std[:, None, None]

            og2, gen2, latents_neutral = _generate_result_for_canvas(args, model, curr_canvas)

            #import pdb; breakpoint()

            for index in range(len(latents)):
                latents[index] = latents[index] - latents_neutral[index]
                
            og_holder.append(original_image)
            gen_holder.append(generated_result)

            og_holder.append(og2)
            gen_holder.append(gen2)

                
            if i == 0:
                metric = segmentation_iou = evaluate_mse(original_image, generated_result, args)["mse"]
            if i == 1:
                metric = colorization_mse = evaluate_mse(original_image, generated_result, args)["mse"]
            if i == 2:
                metric = neutral_copy_mse = evaluate_mse(original_image, generated_result, args)["mse"]
            if i == 3:
                metric = bw_mse = evaluate_mse(original_image, generated_result, args)["mse"]
            if i == 4:
                metric = lowlight_mse = evaluate_mse(original_image, generated_result, args)["mse"]
            if i == 5:
                metric = inpaint1_mse = evaluate_mse(original_image, generated_result, args)["mse"]
            if i == 6:
                metric = inpaint2_mse = evaluate_mse(original_image, generated_result, args)["mse"]

            if args.store_latents:
                try:
                    write_latent(args.store_latents, f'{query_name}___{support_name}', latents, captions[i], metric)
                except Exception as e:
                    print(f"Failed to write latent for {query_name}___{support_name}. Error: {e}")
            
            if args.output_dir and args.save_images is not None and idx % args.save_images == 0:
            
                og_holder = [np.array(img) for img in og_holder]
                gen_holder = [np.array(img) for img in gen_holder]

                fig, axs = plt.subplots(2, 2, figsize=(8, 8))

                for index in range(2):
                    axs[0, index].imshow(og_holder[index])
                    axs[0, index].axis('off')  # Turn off axis

                    axs[1, index].imshow(gen_holder[index])
                    axs[1, index].axis('off')  # Turn off axis

                plt.tight_layout()
                plt.subplots_adjust(bottom=0.1)  # Adjust as needed
                plt.savefig(os.path.join(args.output_dir, f'combined_{idx}_{captions[i]}.png'))
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
    if args.model != "improv":
        ours = (np.transpose(ours/255., [2, 0, 1]) - imagenet_mean[:, None, None]) / imagenet_std[:, None, None]
        target = (np.transpose(target/255., [2, 0, 1]) - imagenet_mean[:, None, None]) / imagenet_std[:, None, None]
    else:
        ours = (np.transpose(ours/255., [2, 0, 1]))
        target = (np.transpose(target/255., [2, 0, 1]))

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
