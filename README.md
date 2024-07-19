# Finding Visual Task Vectors (ECCV 2024)
### [Alberto Hojel](https://alhojel.github.io/), [Yutong Bai](https://yutongbai.com/), [Trevor Darrell](https://people.eecs.berkeley.edu/~trevor/), [Amir Globerson](http://www.cs.tau.ac.il/~gamir/), [Amir Bar](https://amirbar.net)

Welcome to the official repository for our paper: ["Finding Visual Task Vectors."](https://arxiv.org/abs/2404.05729)


## Abstract:

Visual Prompting is a technique for teaching models to perform a visual task via
in-context examples, and without any additional training. In this work, we analyze
the activations of MAE-VQGAN, a recent Visual Prompting model, and find
task vectors, activations that encode task specific information. Equipped with this
insight, we demonstrate that it is possible to identify the task vectors and use them
to guide the network towards performing different tasks without providing any
input-output examples. To find task vectors, we compute the average intermediate
activations per task and use the REINFORCE algorithm to search for the subset
of task vectors. The resulting task vectors guide the model towards performing a
task better than the original model without the need for input-output examples

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=alhojel/visual_task_vectors&type=Date)](https://star-history.com/#alhojel/visual_task_vectors&Date)

### Dataset preparation:

Our evaluation pipeline is based on [Volumetric Aggregation Transformer](https://github.com/Seokju-Cho/Volumetric-Aggregation-Transformer). Please follow the dataset preparation steps for PASCAL-5i dataset in this repository. 

### Collect mean activations for each task:
```
cd evaluate && python collect_attention_heads.py \
    --model mae_vit_large_patch16 \
    --base_dir <pascal_5i_basedir> \
    --output_dir <outputs_dir> \
    --ckpt <model_ckp_path> \
    --device <device> \ 
    --num_collections 100 \ 
```

The script will save a the mean activations as pkl files.

### Start REINFORCE algorithm to find optimal patching positions:
```
cd evaluate && python reinforce_train.py \
    --model mae_vit_large_patch16 \
    --base_dir <pascal_5i_basedir> \
    --output_dir <outputs_dir> \
    --ckpt <model_ckp_path> \
    --split <split> \
    --device <device> \ 
    --task [0,1,2,3,4, None] \
```

The script will run the REINFORCE algorithm using the mean activations computed previously and store the optimal patching positions as pkl files.

### Evaluate the patching positions:
```
cd evaluate && python reinforce_evaluate.py \
    --model mae_vit_large_patch16 \
    --base_dir <pascal_5i_basedir> \
    --output_dir <outputs_dir> \
    --ckpt <model_ckp_path> \
    --split <split> \
    --device <device> \ 
    --setup <setup name for identification purposes> \ 
    --task [0,1,2,3,4] \
    --load_model <path to the pkl file> \
```

The script will evaluate the patching positions and store the results to a log file.

# Pretrained Models
| Model             | Pretraining | Epochs | Link |
|-------------------|-------------|--------|------|
| MAE-VQGAN (ViT-L) | CVF + IN    | 3400   |   [link](https://github.com/amirbar/visual_prompting)   |
