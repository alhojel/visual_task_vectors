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

def write_latent(file_path, pass_id, latent, label, metric, model_type="improv"):
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
        
        if model_type == "improv":
            label_group.create_dataset('decoder_latent', data=np.array(latent[:-1]), compression="gzip", compression_opts=2)
            label_group.create_dataset('text', data=np.array(latent[-1]), compression="gzip", compression_opts=2)
        else:
            if len(latent) > 8:
                latent = latent[-8:]
            label_group.create_dataset('latent', data=np.array(latent), compression="gzip", compression_opts=2)


        label_group.create_dataset('metric', data=metric)


def _generate_result_for_canvas(args, model, canvas, input_prompts="", model_type="improv"):
    """canvas is already in the right range."""
    if model_type != "improv":
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
    with open(os.path.join(args.output_dir, 'log.txt'), 'a') as log:
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
    
    # """if args.model == "improv":
    #     model = IMProvPipeline.from_pretrained(pretrained_model_name_or_path="xvjiarui/IMProv-v1-0")
    #     _ = model.to(args.device)
    # else:
    #     model = prepare_model(args.ckpt, arch=args.model)"""
        
    improv = IMProvPipeline.from_pretrained(pretrained_model_name_or_path="xvjiarui/IMProv-v1-0")
    model = prepare_model(args.ckpt, arch=args.model)

    _ = improv.to(args.device)
    _ = model.to(args.device)


    captions = ["segmentation", "colorization", "uncolor", "lowlight enhance", "inpaint single random", "inpaint double random"]
    
    for idx in trange(len(ds)):
        canvas = ds[idx]['grid']

        query_name = ds[idx]['query_name']
        support_name = ds[idx]['support_name']

        
        for i in range(len(canvas)):

            if captions[i] != "segmentation":
                continue

            image_holder = []
            label_holder = []

            
            model_canvas = (canvas[i] - imagenet_mean[:, None, None]) / imagenet_std[:, None, None]
            improv_canvas = canvas[i]

            text_prompt = best_prompts[i]

            improv_original_image, improv_generated_result, improv_latents = _generate_result_for_canvas(args, improv, improv_canvas, input_prompts = text_prompt, model_type="improv")
            model_original_image, model_generated_result, model_latents = _generate_result_for_canvas(args, model, model_canvas, model_type="model")

            improv_canvas = canvas[i].clone().detach()
            midpoint = improv_canvas.shape[2] // 2
            left_half = improv_canvas[:, :, :midpoint]
            improv_canvas[:, :, midpoint:] = left_half

            model_canvas = (improv_canvas - imagenet_mean[:, None, None]) / imagenet_std[:, None, None]

            improv_og2, improv_gen2, improv_latents_neutral = _generate_result_for_canvas(args, improv, improv_canvas,  input_prompts = nc_best_prompt, model_type="improv")
            model_og2, model_gen2, model_latents_neutral = _generate_result_for_canvas(args, model, model_canvas, model_type="model")

            for latent_i in range(len(improv_latents)):
                improv_latents[latent_i] = improv_latents[latent_i] - improv_latents_neutral[latent_i]

            for latent_i in range(len(model_latents)):
                model_latents[latent_i] = model_latents[latent_i] - model_latents_neutral[latent_i]
            
            image_holder.append(improv_original_image)
            label_holder.append("Ground Truth Task")
            image_holder.append(improv_og2)
            label_holder.append("Ground Truth Neutral")

            image_holder.append(improv_generated_result)
            label_holder.append("Improv Task Genereation")
            image_holder.append(improv_gen2)
            label_holder.append("Improv Neutral Genereation")

            image_holder.append(model_generated_result)
            label_holder.append("MAE-VQGAN Task Genereation")
            image_holder.append(model_gen2)
            label_holder.append("MAE-VQGAN Neutral Genereation")

            with open(os.path.join(args.output_dir, 'log.txt'), 'a') as log:
                current_metric = {}
                current_metric["query_name"] = query_name
                current_metric["support_name"] = support_name
                current_metric["task"] = captions[i]
                current_metric["prompt"] = text_prompt
                current_metric["split"] = args.split

                current_metric["improv_metric"] = evaluate_mse(improv_original_image, improv_generated_result, args)["mse"]
                current_metric["improv_copymetric"] = evaluate_mse(improv_og2, improv_gen2, args)["mse"]\
                
                current_metric["model_metric"] = evaluate_mse(model_original_image, model_generated_result, args)["mse"]
                current_metric["model_copymetric"] = evaluate_mse(model_og2, model_gen2, args)["mse"]
                
                improv_h = evaluate_segmentation(improv_original_image, improv_generated_result, args)
                current_metric["improv_iou"] = improv_h["iou"]
                current_metric["improv_accuracy"] = improv_h["accuracy"]

                model_h = evaluate_segmentation(model_original_image, model_generated_result, args)
                current_metric["model_iou"] = model_h["iou"]
                current_metric["model_accuracy"] = model_h["accuracy"]

                log.write(str(current_metric) + '\n')

            if args.store_latents:
                try:
                    write_latent(args.store_latents, f'{query_name}___{support_name}', improv_latents, captions[i]+"_improv", current_metric["improv_iou"], model_type="improv")
                except Exception as e:
                    import pdb; breakpoint()
                    print(f"Failed to write latent for {query_name}___{support_name}. Error: {e}")
                
                try:
                    write_latent(args.store_latents, f'{query_name}___{support_name}', model_latents, captions[i]+"_model", current_metric["model_iou"], model_type="model")
                except Exception as e:
                    import pdb; breakpoint()
                    print(f"Failed to write latent for {query_name}___{support_name}. Error: {e}")
            
            if args.output_dir and args.save_images is not None and idx % args.save_images == 0:
            
                image_holder = [np.array(img) for img in image_holder]

                fig, axs = plt.subplots(3, 2, figsize=(8, 12))  # Adjusted to 3 rows and 2 columns

                for index in range(6):  # Adjusted to loop over 6 images
                    row = index // 2  # Determine row for subplot
                    col = index % 2  # Determine column for subplot
                    axs[row, col].imshow(image_holder[index])
                    axs[row, col].axis('off')  # Turn off axis
                    axs[row, col].set_title(label_holder[index])  # Add label under each image

                plt.tight_layout()
                plt.subplots_adjust(bottom=0.1)  # Adjust as needed
                plt.savefig(os.path.join(args.output_dir, f'{args.split}combined_{idx}_{captions[i]}.png'))
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
    
    """ if args.model != "improv":
        ours = (np.transpose(ours/255., [2, 0, 1]) - imagenet_mean[:, None, None]) / imagenet_std[:, None, None]
        target = (np.transpose(target/255., [2, 0, 1]) - imagenet_mean[:, None, None]) / imagenet_std[:, None, None]
    else:
        ours = (np.transpose(ours/255., [2, 0, 1]))
        target = (np.transpose(target/255., [2, 0, 1])) """
    
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
