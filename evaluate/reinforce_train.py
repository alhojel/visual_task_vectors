import os.path
from tqdm import trange
import multitask_dataloader
from reasoning_dataloader import *
import torchvision
from mae_utils import *
import argparse
from pathlib import Path
from segmentation_utils import *
import numpy as np
import numpy as np
import os
import torch
import torchvision.transforms as T
import pickle
import torch.optim as optim
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
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--t', default=[0, 0, 0], type=float, nargs='+')
    parser.add_argument('--ckpt', help='model checkpoint')
    parser.add_argument('--split', default=0 , type=int)
    parser.add_argument('--purple', default=0, type=int)
    parser.add_argument('--flip', default=0, type=int)
    parser.add_argument('--save_images', default=None, type=int, help='Save images')
    parser.add_argument('--train_images', default=10, type=int)
    parser.add_argument('--train_iters', default=500, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--init', default=-1.0, type=float)
    parser.add_argument('--zero_shot', default=1, type=int)
    parser.add_argument('--restrict_area', default=0, type=int)
    parser.add_argument('--task', default=None, type=int)
    parser.add_argument('--granularity', default=1, type=int)
    parser.add_argument('--regularization_strength', default=0, type=float)

    return parser


class JointModel(nn.Module):
    def __init__(self, args, prompting_model, num_variables, train_ds, task_tensor):
        super().__init__()
        self.prompting_model = prompting_model
        self.num_variables = num_variables
        self.bernoullis = [torch.tensor(args.init, requires_grad=True) for _ in range(self.num_variables)]

        if args.restrict_area==1:
            for i in range(24*16*2):
                self.bernoullis[i] = torch.tensor(-1000, requires_grad=True)
        elif args.restrict_area==2:
            for i in range(24*16*2, len(self.bernoullis)):
                self.bernoullis[i] = torch.tensor(-1000, requires_grad=True)

        self.optim = optim.Adam(self.bernoullis, lr=args.lr)
        self.eps = 1e-6
        self.batch_size = 64*5
        self.regularization_strength = args.regularization_strength

        self.train_ds = train_ds

        self.task_tensor = task_tensor

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
    
    def forward(self, args, canvases):

        sigmoid_tensor = torch.stack([torch.sigmoid(bernoulli).clamp(min=self.eps, max=1-self.eps) for bernoulli in self.bernoullis])
        sigmoid_tensor = sigmoid_tensor.unsqueeze(0).expand(self.batch_size, self.num_variables)
        prob_dist = torch.distributions.Bernoulli(sigmoid_tensor)
        sampled_patches = prob_dist.sample()

        loss_list = []
        for i in range(0,self.batch_size):
            
            indices = construct_indices(sampled_patches[i], args.granularity, args.zero_shot)
            canvas = canvases[i%len(canvases)]

            if args.task is not None:
                current_injection = self.task_tensor
            else:
                current_injection = self.task_tensor[i%len(self.task_tensor)]
        
            with torch.no_grad():        
                if args.zero_shot:
                    indices_premask = []
                    drop_indices = [1+a for a in q1]+[1+a for a in q2]
                    
                    indices_premask = [0]+list(range(99,148))

                    original_image, generated_result, _ = _generate_result_for_canvas(args, self.prompting_model, canvas, premask_pass_indices = indices_premask, attention_heads=indices, attention_injection=current_injection, drop_indices = drop_indices)
                else:
                    original_image, generated_result, _ = _generate_result_for_canvas(args, self.prompting_model, canvas, attention_heads=indices, attention_injection=current_injection)
                
            if args.task == 0:
                loss = -1*self.loss_iou(generated_result, original_image)
            else:
                if args.task is None:
                    if i%len(self.task_tensor) == 0:
                        loss = -1*self.loss_iou(generated_result, original_image)
                    else:
                        loss = self.loss_mse(original_image, generated_result)
                else:
                    loss = self.loss_mse(original_image, generated_result)

            loss_list.append(loss)

        loss_list = torch.tensor(loss_list)
        average_loss = loss_list.mean()
        log_prob = prob_dist.log_prob(sampled_patches).mean(-1)

        if args.task is  None:
            for index in range(len(self.task_tensor)):
                current = loss_list[index::len(self.task_tensor)]
                loss_list[index::len(self.task_tensor)] = (current-current.mean())/(current.std() + self.eps)

        minus_r = (loss_list - loss_list.mean())/(loss_list.std() + self.eps)
                
        minus_r = minus_r.detach()
        loss = (log_prob*minus_r).mean() + self.regularization_strength*torch.mean(torch.stack([torch.sigmoid(bernoulli).clamp(min=self.eps, max=1-self.eps) for bernoulli in self.bernoullis]))
        return loss, generated_result, average_loss

    def train(self, args, num_itr):

        canvases = []
        for idx in range(len(self.train_ds)):
            if args.task is not None:
                canvas = self.train_ds[idx]['grid']
                canvas = (canvas - imagenet_mean[:, None, None]) / imagenet_std[:, None, None]

                canvases.append(canvas)
            else:
                canvas = self.train_ds[idx]['grid']
                for element in canvas:
                    element = (element - imagenet_mean[:, None, None]) / imagenet_std[:, None, None]
                    canvases.append(element)

        best_checkpoint = float('-inf') if args.task == 0 or args.task is None else float('inf')
        
        for i in trange(num_itr):
            self.optim.zero_grad()
            loss, res, average_loss = self.forward(args, canvases[:args.train_images])
            loss.backward()
            self.optim.step()

            if i % 10 == 0:
                print("Metric:", i, average_loss)

                with open(os.path.join(args.output_dir,'log.txt'), 'a') as log:
                    current_metric = {}
                    current_metric["iter"] = i
                    current_metric["train_loss"] = average_loss.item()
                    current_metric["lr"] = args.lr
                    current_metric["init"] = args.init
                    current_metric["reg_strength"] = self.regularization_strength
                    current_metric["restrict_area"] = args.restrict_area
                    current_metric["batch_size"] = self.batch_size
                    current_metric["granularity"] = args.granularity
                    current_metric["total_train_images"] = args.train_images
                    current_metric["task"] = args.task

                    log.write(str(current_metric) + '\n')
            
            if i % 50 == 0:

                eval_loss, eval_indices = self.eval(args)

                with open(os.path.join(args.output_dir,'log.txt'), 'a') as log:
                    current_metric = {}
                    current_metric["iter"] = i
                    current_metric["eval_loss"] = eval_loss
                    current_metric["eval_patch_count"] = eval_indices
                    current_metric["lr"] = args.lr
                    current_metric["init"] = args.init
                    current_metric["granularity"] = args.granularity
                    current_metric["batch_size"] = self.batch_size
                    current_metric["reg_strength"] = self.regularization_strength
                    current_metric["reg_strength"] = args.restrict_area
                    current_metric["images_per_batch"] = args.train_images
                    current_metric["task"] = args.task
                    current_metric["split"] = args.split

                    log.write(str(current_metric) + '\n')

                # Save self.bernoullis to a pickle file with suffix {args.lr}_{args.init}_{i}
                bernoullis_save_path = os.path.join(args.output_dir, f'bernoullis_{args.task}_{args.granularity}_{self.regularization_strength}_{args.restrict_area}_{args.train_images}_{args.lr}_{args.init}_{i}.pkl')
                with open(bernoullis_save_path, 'wb') as f:
                    pickle.dump([bernoulli.detach().cpu().numpy() for bernoulli in self.bernoullis], f)

                if eval_loss < best_checkpoint and (args.task != 0 and args.task is not None):
                    best_checkpoint = eval_loss
                    best_bernoullis_save_path = os.path.join(args.output_dir, f'bernoullis_{args.task}_{args.granularity}_{self.regularization_strength}_{args.restrict_area}_{args.train_images}_{args.lr}_{args.init}_best.pkl')
                    with open(best_bernoullis_save_path, 'wb') as f:
                        pickle.dump([bernoulli.detach().cpu().numpy() for bernoulli in self.bernoullis], f)
                elif eval_loss > best_checkpoint and (args.task == 0 or args.task is None):
                    best_checkpoint = eval_loss
                    best_bernoullis_save_path = os.path.join(args.output_dir, f'bernoullis_{args.task}_{args.granularity}_{self.regularization_strength}_{args.restrict_area}_{args.train_images}_{args.lr}_{args.init}_best.pkl')
                    with open(best_bernoullis_save_path, 'wb') as f:
                        pickle.dump([bernoulli.detach().cpu().numpy() for bernoulli in self.bernoullis], f)
                
        return
    
    def eval(self, args):

        sigmoid_tensor = torch.stack([torch.sigmoid(bernoulli).clamp(min=self.eps, max=1-self.eps) for bernoulli in self.bernoullis])
        sigmoid_tensor = sigmoid_tensor
        prob_dist = torch.distributions.Bernoulli(sigmoid_tensor)
        sampled_patches = prob_dist.sample()
        indices = construct_indices(sampled_patches, args.granularity, args.zero_shot)

        if args.task is None:
            curr_injection = self.task_tensor[0]
        else:
            curr_injection = self.task_tensor
        
        loss_holder = []
        offset = args.train_images if args.task is not None else args.train_images // 5

        for idx in trange(offset, min(len(self.train_ds), 500)):
            if args.task is not None:
                canvas = self.train_ds[idx]['grid']
            else:
                canvas = self.train_ds[idx]['grid'][0]

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
            elif args.task == 0:
                loss = self.loss_iou(original_image, generated_result).item()
            else:
                loss = self.loss_mse(original_image, generated_result)
            loss_holder.append(loss)

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

    ds = multitask_dataloader.DatasetPASCAL(args.base_dir, fold=args.split, image_transform=image_transform, mask_transform=mask_transform,
                         flipped_order=args.flip, purple=args.purple, iters=None, type="trn", task=args.task)
    
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

    rl_model = JointModel(args, model, params, ds, injection)
    rl_model = rl_model.to(args.device)

    rl_model.train(args, num_itr=args.train_iters)

if __name__ == '__main__':
    args = get_args()

    args = args.parse_args()
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    evaluate(args)



