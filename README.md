# Long-Horizon Embodied Agent with Perceptual-Cognitive Memory
[Enhua Cheng]

## Contents
This repository is built upon the official implementation of MemoryVLA [Link](https://github.com/shihao1895/MemoryVLA/tree/openvla-codebase?tab=readme-ov-file#Training). Thanks to the original authors for their open-source work.

 * [**Install**](#Install)
 * [**Training**](#Training)
 * [**Evaluation in SimplerEnv**](#Evaluation-in-SimplerEnv)
 * [**Evaluation in LIBERO**](#Evaluation-in-LIBERO)


## Install

The code is built using Python 3.10, and we use PyTorch == 2.2.0 and CUDA == 12.1 (It may run with lower versions, but we have not tested it).

We recommend using [Miniconda](https://docs.conda.io/en/latest/miniconda.html) and setting up an environment:
```bash
conda create --name memvla python=3.10
conda activate memvla

pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
conda install -c nvidia cuda-nvcc=12.1 cuda-toolkit=12.1 -y
```
If you need to use the traning code, please also install the [Flash Attention](https://github.com/Dao-AILab/flash-attention), we use flash-attn==2.5.5:

```bash
# Install Flash Attention 2.5.5, this is an example for pytorch2.2-cuda12.1
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.5/flash_attn-2.5.5+cu122torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install flash_attn-2.5.5+cu122torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

Next, clone our repo and install the required packages:

```bash
git clone https://github.com/shihao1895/MemoryVLA
cd MemoryVLA
pip install -e .
```
If you are using an NVIDIA Hopper GPU (e.g., H20) and encounter the error  
“Floating point exception (core dumped)”, try reinstalling the specific cuBLAS version below:

```bash
# Fix for NVIDIA H20: "Floating point exception (core dumped)"
pip install nvidia-cublas-cu12==12.4.5.8
```

## Training

1. Prepare training dataset with [RLDS](https://github.com/google-research/rlds) format:

   - [LIBERO](https://libero-project.github.io/intro.html) (including Spatial, Object, Goal, Long-10, Long-90 suites)
   - Bridge from [Open X-Embodiment (OXE)](https://robotics-transformer-x.github.io/)
   - Fractal from [Open X-Embodiment (OXE)](https://robotics-transformer-x.github.io/)

   ```bash
   # Make sure you have git-lfs installed (https://git-lfs.com)
   git lfs install
   # Download the LIBERO dataset (processed, ~22 GB)
   git clone https://huggingface.co/datasets/shihao1895/libero-rlds
   # Download the Bridge dataset (processed, ~157 GB)
   git clone https://huggingface.co/datasets/shihao1895/bridge-rlds
   # Download the Fractal dataset (processed)
   git clone https://huggingface.co/datasets/shihao1895/fractal-rlds
   ```

2. Download pretrained model, we use [OpenVLA Pretrained Model](https://huggingface.co/openvla/openvla-7b-prismatic) for LIBERO training, and [CogACT Pretrained Model](https://huggingface.co/CogACT/CogACT-Large) for Bridge and Fractal training.

   ```bash
   # Download OpenVLA pretrained checkpoint (~30 GB)
   git clone https://huggingface.co/openvla/openvla-7b-prismatic
   
   # Download CogACT pretrained checkpoint (~31 GB)
   git clone https://huggingface.co/CogACT/CogACT-Large
   ```

3. Train the model on different datasets

   Before training, modify several parameters in the corresponding scripts, such as `hf_token`, `wandb_entity`, checkpoint paths, dataset paths, and log directories.

   We train on a single node with 8× NVIDIA A100 GPUs.

   ```bash
   # Train on the Bridge dataset
   bash script/train/bridge/train_bridge.sh
   # Train on the LIBERO-Spatial dataset
   bash script/train/libero/train_libero_spatial.sh
   # Train on the LIBERO-Object dataset
   bash script/train/libero/train_libero_object.sh
   # Train on the LIBERO-Goal dataset
   bash script/train/libero/train_libero_goal.sh
   # Train on the LIBERO-100 dataset
   bash script/train/libero/train_libero_100.sh
   # Train on the Fractal dataset
   bash script/train/fractal/train_fractal.sh
   # Train on real-world data
   bash script/train/real_world/train_real.sh
   ```

   To finetune on your own customized data, please follow the instruction [(rlds_dataset_builder)](https://github.com/kpertsch/rlds_dataset_builder) for converting your data to RLDS format. The actions should be the deltas of end effector ``EEF Delta XYZ (3) + Roll-Pitch-Yaw (3) + Gripper Open/Close (1)``. Once your customized data is ready, place the customized data directly under the ``<data_root_dir>/custom_finetuning/1.0.0`` directory. Then set ``vla.data_mix="custom_finetuning"``.

## Evaluation in SimplerEnv

We provide evaluation interfaces and scripts based on [SimplerEnv](https://simpler-env.github.io/).

1. Please follow the installation guide in the [SimplerEnv Repo](https://github.com/simpler-env/SimplerEnv) to set up the simulation environment, and make sure to place the repo under: `./third_libs/SimplerEnv`

2. Evaluation Example.

   ```bash
   # Run evaluation
   bash script/eval/bridge/eval_bridge.sh
   # Summarize results
   python script/eval/bridge/extract_bridge_results.py
   ```

   > **NOTE**: Due to the instability of the SimplerEnv benchmark and diffusion process, the performance scores across different iterations can vary significantly. Please evaluate multiple checkpoints and report the best result.

## Evaluation in LIBERO

We also provide evaluation interfaces and scripts based on [LIBERO](https://libero-project.github.io/intro.html).

1. Please follow the installation guide in the [LIBERO Repo](https://github.com/Lifelong-Robot-Learning/LIBERO) to set up the simulation environment, and make sure to place the repo under: `./third_libs/LIBERO`

2. Evaluation Example.

   ```bash
   # Run evaluation
   bash script/eval/libero/eval_libero.sh
   # Summarize results
   python script/eval/libero/extract_libero_results.py
   ```

   > **NOTE:** The evaluation mechanism here is different from SimplerEnv. The process first loads the model using `develop.py`, then waits for a period before running `evaluation/libero/eval_libero.py` for testing. In addition, since performance may vary across iterations, please evaluate multiple checkpoints and report the best result.



