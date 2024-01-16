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
    parser.add_argument('--task_vector', default=None, type=str, help='What task vector to use')


    return parser


def _generate_result_for_canvas(args, model, canvas, encoder_task_vector=None, decoder_task_vector=None):
    """canvas is already in the right range."""
    ids_shuffle, len_keep = generate_mask_for_evaluation()
    if encoder_task_vector is not None and decoder_task_vector is not None:

        _, im_paste, _, latents = generate_image(canvas.unsqueeze(0).to(args.device), model, ids_shuffle.to(args.device),
                                    len_keep = len_keep, e_vec = encoder_task_vector.to(args.device), d_vec = decoder_task_vector.to(args.device), device=args.device)
    else:
        _, im_paste, _, latents = generate_image(canvas.unsqueeze(0).to(args.device), model, ids_shuffle.to(args.device),
                                    len_keep, device=args.device)
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
                         flipped_order=args.flip, purple=args.purple, query_support_list_file=args.query_support_list_file, iters=args.iters)
    
    model = prepare_model(args.ckpt, arch=args.model)
    _ = model.to(args.device)

    captions = ['label_segmentation', 'label_colorization', "neutral copy", "label_uncolor", "label_lowlight enhance", 'label_inpaint single random', 'label_inpaint double random']


    with open(args.task_vector, "rb") as file:
        data = pickle.load(file)
        
    vector_task_labels = data.keys()
    vector_type_labels = data[vector_task_labels[0]].keys()

    for idx in trange(len(ds)):
        canvas = ds[idx]['grid']

        query_name = ds[idx]['query_name']
        support_name = ds[idx]['support_name']

        gen_holder = []
        og_holder = []
        
        for i in range(len(canvas)):
            if i == 2:
                continue

            original_image = np.uint8(torch.clip(torch.einsum('chw->hwc', canvas[i]) * 255, 0, 255).int().numpy())

            og_holder.append(original_image)

            assert captions[i] in vector_task_labels

            for vector_type in vector_type_labels:

                encoder_task_vector = data[captions[i]][vector_type]["encoder"]
                decoder_task_vector = data[captions[i]][vector_type]["decoder"]
        
                coeff_array = np.arange(0, 2, 0.25)
                coeff_array.append(None)

                for coeff in coeff_array:
                    if coeff is None:
                        curr_canvas = (canvas[i] - imagenet_mean[:, None, None]) / imagenet_std[:, None, None]
                        original_image, generated_result, latents = _generate_result_for_canvas(args, model, curr_canvas)
                    else:
                        curr_canvas = (canvas[2] - imagenet_mean[:, None, None]) / imagenet_std[:, None, None]
                        original_image, generated_result, latents = _generate_result_for_canvas(args, model, curr_canvas, coeff*encoder_task_vector, coeff*decoder_task_vector)
                    

                    original_image = np.uint8(torch.clip(torch.einsum('chw->hwc', canvas[i]) * 255, 0, 255).int().numpy())

                    gen_holder.append(generated_result)

                    if i == 0:
                        metric = segmentation_iou = evaluate_segmentation(original_image, generated_result, args)["iou"]
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

                    
                    with open(os.path.join(args.output_dir, 'log.txt'), 'a') as log:
                        current_metric = {}
                        current_metric["query_name"] = query_name
                        current_metric["support_name"] = support_name
                        current_metric["task"] = captions[i]
                        current_metric["lambda"] = coeff
                        current_metric["metric"] = metric
                        current_metric["vector"] = vector_type
                    
                        log.write(str(current_metric) + '\n')
            
            """ if args.output_dir and args.save_images is not None and idx % args.save_images == 0:
                
                og_holder = [np.array(img) for img in og_holder]
                gen_holder = [np.array(img) for img in gen_holder]

                # Determine the number of images
                num_images = len(og_holder)+len(gen_holder)

                # Handling the case where there is only one image
                if num_images == 1:
                    fig, axs = plt.subplots(2, 1, figsize=(3.5, 8))
                    axs = axs.reshape(2, -1)  # Reshape axs to 2D array for consistency
                else:
                    fig, axs = plt.subplots(1, num_images, figsize=(3.5*num_images, 8))

                for i in range(num_images):
                    if i == 0:
                        axs[i].imshow(og_holder[0])
                        axs[i].axis('off') 
                    else:
                        axs[i].imshow(gen_holder[i-1])
                        axs[i].axis('off') 

                plt.tight_layout()
                plt.subplots_adjust(bottom=0.1)  # Adjust as needed
                plt.savefig(os.path.join(args.output_dir, f'combined_{idx}.png'))
                plt.show() """
 
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
