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



def _generate_result_for_canvas(args, model, canvas, encoder_task_vector=None, decoder_task_vector=None, only_cls=True, input_prompts=""):
    """canvas is already in the right range."""
    if args.model != "improv":
        ids_shuffle, len_keep = generate_mask_for_evaluation()
        if decoder_task_vector is not None:
            _, im_paste, _, latents = generate_image(canvas.unsqueeze(0).to(args.device), model, ids_shuffle.to(args.device),
                                        len_keep = len_keep, e_vec = None, d_vec = decoder_task_vector.to(args.device), device=args.device, only_cls=False)
        else:
            _, im_paste, _, latents = generate_image(canvas.unsqueeze(0).to(args.device), model, ids_shuffle.to(args.device),
                                        len_keep = len_keep, e_vec = None, d_vec = None, device=args.device, only_cls=False)
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
                d_vec = decoder_task_vector.to(args.device) if decoder_task_vector is not None else None,
                text_vec = encoder_task_vector.to(args.device) if encoder_task_vector is not None else None,
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
    
    if args.model == "improv":
        model = IMProvPipeline.from_pretrained(pretrained_model_name_or_path="xvjiarui/IMProv-v1-0")
        _ = model.to(args.device)
    else:
        model = prepare_model(args.ckpt, arch=args.model)
        
    _ = model.to(args.device)

    captions = ['label_segmentation', 'label_colorization', "label_uncolor", "label_lowlight enhance", 'label_inpaint single random', 'label_inpaint double random']


    with open(args.task_vector, "rb") as file:
        data = pickle.load(file)
        
    vector_task_labels = list(data.keys())

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

            text_prompt = ""

            if args.model == "improv":
                curr_canvas = canvas[i]
                text_prompt = best_prompts[i]

            
            original_image, generated_result, latents = _generate_result_for_canvas(args, model, curr_canvas, input_prompts = text_prompt)

            og_holder.append(generated_result)
            label_holder.append("Actual Prompt")
                    
            
            vector_model = ['label_segmentation_improv', 'label_segmentation_model']
            
            with open(os.path.join(args.output_dir, 'log.txt'), 'a') as log:
                current_metric = {}
                current_metric["query_name"] = query_name
                current_metric["support_name"] = support_name
                current_metric["task"] = captions[i]
                current_metric["model"] = args.model
                current_metric["metric"] = evaluate_mse(original_image, generated_result, args)["mse"]
                if i == 0:
                    h = evaluate_segmentation(original_image, generated_result, args)
                    current_metric["iou"] = h["iou"]
                    current_metric["accuracy"] = h["accuracy"]
            
                log.write(str(current_metric) + '\n')
                
            if args.model == "improv":
                text_task_vector = torch.tensor(data[vector_model[0]]["text"])
                number = 0
            else:
                number = 1
            
            decoder_task_vector = torch.tensor(data[vector_model[number]]["decoder"])

            curr_canvas = canvas[i].clone().detach()
            midpoint = curr_canvas.shape[2] // 2
            left_half = curr_canvas[:, :, :midpoint]
            curr_canvas[:, :, midpoint:] = left_half

            if args.model != "improv":
                curr_canvas = (curr_canvas - imagenet_mean[:, None, None]) / imagenet_std[:, None, None]

            coeff_array = np.append(np.array(0),np.arange(0.8,1.3,0.1))

            for coeff_first in coeff_array:
                    
                if args.model == "improv":
                    for text_coeff in [0, 0.7, 0.9, 1.1]:
                    
                        zero_tensor = torch.zeros_like(decoder_task_vector)

                        zero_tensor[0] = coeff_first*decoder_task_vector[0]
                        
                        original_image, generated_result, latents = _generate_result_for_canvas(args, model, curr_canvas, text_coeff*text_task_vector, zero_tensor, input_prompts = nc_best_prompt)

                        original_image = np.uint8(torch.clip(torch.einsum('chw->hwc', canvas[i]) * 255, 0, 255).int().numpy())

                        gen_holder.append(generated_result)
                        label = "Lambda:"+("{:.6f}".format(coeff_first)).rstrip('0').rstrip('.')+", "+"{:.6f}".format(text_coeff).rstrip('0').rstrip('.')
                        label_holder.append(label)

                        with open(os.path.join(args.output_dir, 'log.txt'), 'a') as log:
                            current_metric = {}
                            current_metric["query_name"] = query_name
                            current_metric["support_name"] = support_name
                            current_metric["task"] = captions[i]
                            current_metric["model"] = args.model
                            current_metric["coeff"] = coeff_first
                            current_metric["text_coeff"] = text_coeff
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

                else:
                    for tokens in ["all", "prompt_only", "excluding_query"]:
                    
                        zero_tensor = torch.zeros_like(decoder_task_vector)
                        if tokens == "all":
                            zero_tensor[0] = coeff_first*decoder_task_vector[0]
                        if tokens == "prompt_only":
                            zero_tensor[0][:,:99] = coeff_first*decoder_task_vector[0][:,:99]
                        if tokens == "excluding_query":
                            zero_tensor[0] = coeff_first*decoder_task_vector[0]
                            zero_tensor[0][:,99:148] = 0

                        original_image, generated_result, latents = _generate_result_for_canvas(args, model, curr_canvas,decoder_task_vector=zero_tensor)

                        original_image = np.uint8(torch.clip(torch.einsum('chw->hwc', canvas[i]) * 255, 0, 255).int().numpy())

                        gen_holder.append(generated_result)
                        label = "Lambda:"+("{:.6f}".format(coeff_first)).rstrip('0').rstrip('.')+", "+tokens
                        label_holder.append(label)

                        with open(os.path.join(args.output_dir, 'log.txt'), 'a') as log:
                            current_metric = {}
                            current_metric["query_name"] = query_name
                            current_metric["support_name"] = support_name
                            current_metric["task"] = captions[i]
                            current_metric["coeff"] = coeff_first
                            current_metric["model"] = args.model
                            current_metric["tokens"] = tokens
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
                num_rows = 6
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
                plt.savefig(os.path.join(args.output_dir, f'combined_{idx}_{captions[i]}_{args.model}.png'))
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


""" if args.output_dir and args.save_images is not None and idx % args.save_images == 0:
                
                og_holder = [np.array(img) for img in og_holder]
                
                # Sort the metric_holder by the metric value and select the top 5
                metric_holder.sort(key=lambda x: x[0])
                top_gen_images = metric_holder[:5]

                # If the task is segmentation, also select the top 5 images with the highest IOU
                if captions[i] == "label_segmentation":
                    top_iou_images = sorted(metric_holder, key=lambda x: x[1], reverse=True)[:5]
                else:
                    top_iou_images = []

                # Extract the images and labels for the top 5 by metric and IOU
                gen_holder = [x[3] if len(x) > 3 else x[2] for x in top_gen_images + top_iou_images]  # Adjusted to prevent index out of range
                label_holder = [x[2] for x in top_gen_images + top_iou_images]

                # Determine the number of images
                num_rows = max(len(og_holder), len(gen_holder))  # Adjusted to use the maximum of og_holder and gen_holder lengths
                num_cols = 2  # One for og_holder and one for top 5 gen_holder by metric and IOU

                fig_size = (num_cols * 3, num_rows * 3)
                fig, axs = plt.subplots(num_rows, num_cols, figsize=fig_size, squeeze=False)  # Added squeeze=False to always return a 2D array
                
                for r in range(len(og_holder)):
                    axs[r, 0].imshow(og_holder[r])
                    axs[r, 0].axis('off') 
                    axs[r, 0].set_title("Ground Truth")

                for r in range(len(gen_holder)):
                    axs[r % num_rows, 1].imshow(gen_holder[r])  # Adjusted to prevent index out of bounds
                    axs[r % num_rows, 1].axis('off') 
                    axs[r % num_rows, 1].set_title(label_holder[r])  # Adjusted to prevent index out of bounds

                plt.tight_layout()
                plt.subplots_adjust(bottom=0.1)  # Adjust as needed
                plt.savefig(os.path.join(args.output_dir, f'new_combined_{idx}_{captions[i]}.png'))
                plt.show() 
 """