import os.path
from tqdm import trange
import multitask_dataloader
from reasoning_dataloader import *
import torchvision
from mae_utils import *
import argparse
from pathlib import Path
from segmentation_utils import *
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import os
import torch
import pickle
import torch.nn as nn
from intervention_utils import *


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
    parser.add_argument('--split', default=0 , type=int)
    parser.add_argument('--purple', default=0, type=int)
    parser.add_argument('--flip', default=0, type=int)
    parser.add_argument('--save_images', default=0, type=int, help='Save images')
    parser.add_argument('--eval_iters', default=1000, type=int)
    parser.add_argument('--zero_shot', default=1, type=int)
    parser.add_argument('--task', default=None, type=int)
    parser.add_argument('--granularity', default=1, type=int)
    parser.add_argument('--setup', default="None", type=str)
    parser.add_argument('--load_model', default=None, type=str, help='Where to load model from')

    return parser


class JointModel(nn.Module):
    def __init__(self, args, prompting_model, num_variables, eval_ds, task_tensor, load_model=None):
        super().__init__()
        self.prompting_model = prompting_model
        self.num_variables = num_variables
        self.eps = 1e-6

        self.eval_ds = eval_ds

        self.task_tensor = task_tensor

        if load_model is not None:
            self.bernoullis = pickle.load(open(load_model, 'rb'))
            self.bernoullis = [torch.tensor(bernoulli, requires_grad=True) for bernoulli in self.bernoullis]
        else:
            self.bernoullis = None


    def loss_mse(self, target, ours):
        ours = (torch.permute(ours / 255., (2, 0, 1)) - torch.tensor(imagenet_mean, dtype=torch.float32).to(ours.device)[:, None, None]) / torch.tensor(imagenet_std, dtype=torch.float32).to(ours.device)[:, None, None]
        target = (torch.permute(target.to(ours.device) / 255., (2, 0, 1)) - torch.tensor(imagenet_mean, dtype=torch.float32).to(ours.device)[:, None, None]) / torch.tensor(imagenet_std, dtype=torch.float32).to(ours.device)[:, None, None]

        target = target[:, 113:, 113:]
        ours = ours[:, 113:, 113:]
        mse = torch.mean((target - ours) ** 2)
        return mse.item()

    def loss_iou(self, original_image, generated_result):
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
    
    def run_eval(self, args):

        eval_iou, eval_indices = self.eval(args)

        with open(os.path.join(args.output_dir,'eval_log.txt'), 'a') as log:
            current_line = {}
            current_line["eval_loss"] = eval_iou
            current_line["eval_patch_count"] = eval_indices
            current_line["setup"] = args.setup
            current_line["granularity"] = args.granularity
            current_line["zero_shot"] = args.zero_shot
            current_line["task"] = args.task
            current_line["split"] = args.split
            current_line["load_model"] = args.load_model

            log.write(str(current_line) + '\n')
                
        return None
    
    def eval(self, args):
        if self.bernoullis is not None:
            sigmoid_tensor = torch.stack([torch.sigmoid(bernoulli).clamp(min=self.eps, max=1-self.eps) for bernoulli in self.bernoullis])
            sigmoid_tensor = sigmoid_tensor
            prob_dist = torch.distributions.Bernoulli(sigmoid_tensor)
            sampled_patches = prob_dist.sample()
            indices = construct_indices(sampled_patches, args.granularity, args.zero_shot)
        else:
            indices = []

        if args.task is None:
            curr_injection = self.task_tensor[0]
        else:
            curr_injection = self.task_tensor
        
        loss_holder = []
        for idx in trange(len(self.eval_ds)):

            canvas = self.eval_ds[idx]['grid']
            canvas = (canvas - imagenet_mean[:, None, None]) / imagenet_std[:, None, None]

            with torch.no_grad():        
                if args.zero_shot:
                    indices_premask = []

                    drop_indices = [1+a for a in q1]+[1+a for a in q2]
                    
                    indices_premask = [0]+list(range(99,148))

                    original_image, generated_result, _ = _generate_result_for_canvas(args, self.prompting_model, canvas, premask_pass_indices = indices_premask, attention_heads=indices, attention_injection=curr_injection, drop_indices = drop_indices)
                
                else:
                    original_image, generated_result, _ = _generate_result_for_canvas(args, self.prompting_model, canvas, attention_heads=indices, attention_injection=self.task_tensor)
                
            if args.task is None:
                loss = self.loss_iou(original_image, generated_result).item()
            elif args.task == 0 or args.task == 6:
                loss = self.loss_iou(original_image, generated_result).item()
            else:
                loss = self.loss_mse(original_image, generated_result)
            loss_holder.append(loss)

            image = generated_result.detach().cpu().numpy()

            if args.setup=="GT":
                image = original_image.detach().cpu().numpy()
            
            if args.save_images:
                plt.figure()
                plt.imshow(image)
                plt.axis('off')  # Turn off axis numbers and ticks
                image_save_dir = os.path.join(args.output_dir, 'eval_images')
                if not os.path.exists(image_save_dir):
                    os.makedirs(image_save_dir)
                image_save_path = os.path.join(image_save_dir, f'{args.task}_{args.split}_{idx}_{args.setup}.png')
                plt.savefig(image_save_path, bbox_inches='tight', pad_inches=0)  # Save the image without padding and no axis
                plt.close()  # Close the plot to prevent it from displaying in the notebook or script output

            with open(os.path.join(args.output_dir,'log-images.txt'), 'a') as log:
                current_line = {}
                current_line["file_name"] = f"{args.task}_{args.split}_{idx}_{args.setup}"
                current_line["task"] = args.task
                current_line["split"] = args.split
                current_line["idx"] = idx
                current_line["setup"] = args.setup
                current_line["metric"] = loss

                log.write(str(current_line) + '\n')

        eval_mean_iou = np.mean(loss_holder)

        return eval_mean_iou, len(indices)

def _generate_result_for_canvas(args, model, canvas, premask_pass_indices = None, attention_heads=None, attention_injection=None, drop_indices = None):
    """canvas is already in the right range."""

    ids_shuffle, len_keep = generate_mask_for_evaluation()
    if attention_heads is not None:
        attention_heads = torch.tensor(attention_heads, dtype=torch.int64).to(args.device)
        

    _, im_paste, _, latents, celoss = generate_image(canvas.unsqueeze(0).to(args.device), model, ids_shuffle.to(args.device),
                                    len_keep, device=args.device, premask_pass_indices = premask_pass_indices, attention_heads = attention_heads, attention_injection = attention_injection, drop_indices = drop_indices)

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

    if args.task is not None:
        task = tasks[args.task]
        mean_activations_file = args.output_dir + '/' + task + '_mean_activations.pkl'
        with open(mean_activations_file, 'rb') as file:
            content = pickle.load(file)
            mean_activations_encoder = content[1]
            mean_activations_decoder = content[2]
    
        enc_inj = mean_activations_encoder.to(args.device)
        dec_inj = mean_activations_decoder.to(args.device)
                
        injection = [enc_inj,dec_inj]
    else:
        injection = []
        for task_element in tasks:
            mean_activations_file = args.output_dir + '/' + task_element + '_mean_activations.pkl'
            with open(mean_activations_file, 'rb') as file:
                content = pickle.load(file)
                mean_activations_encoder = content[1]
                mean_activations_decoder = content[2]
        
            enc_inj = mean_activations_encoder.to(args.device)
            dec_inj = mean_activations_decoder.to(args.device)
            injection.append([enc_inj,dec_inj])
    
    if args.granularity==0:
        params = 24*16+8*16
    elif args.granularity==1:
        params = 24*16*2+8*16*3
    elif args.granularity==2:
        params = 24*16*50+8*16*99
    
    for split in [0,1,2,3]:
        args.split = split
        eval_ds = multitask_dataloader.DatasetPASCAL(args.base_dir, fold=split, image_transform=image_transform, mask_transform=mask_transform,
                         flipped_order=args.flip, purple=args.purple, iters=args.eval_iters, type="val", task= args.task if args.task is not None else 0)

        rl_model = JointModel(args, model, params, eval_ds, injection, args.load_model)
        rl_model = rl_model.to(args.device)

        rl_model.run_eval(args)
        
if __name__ == '__main__':
    args = get_args()

    args = args.parse_args()
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    args.eval_task = args.task
    evaluate(args)