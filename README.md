# BabyLLaVA-V2: Generative VLM Trained-from-scratch on SAYCam

📄 [Paper](https://arxiv.org/abs/2512.10932) | 🌐 [Project Page](https://shawnking98.github.io/BabyVLM-v2/) 

<a href="https://shawnking98.github.io/">Shengao Wang</a><sup>*†1</sup>,
<a href="https://wqwang.me/">Wenqi Wang</a><sup>*1</sup>,
<a href="https://victor-wang-902.github.io/">Zecheng Wang</a><sup>*1</sup>,
<a href="https://www.linkedin.com/in/max-whitton-9b648a222/">Max Whitton</a><sup>*1</sup>,
<br>
<a href="https://www.linkedin.com/in/mikewakeham/">Michael Wakeham</a><sup>1</sup>,
<a href="https://arjunchandra2.github.io/">Arjun Chandra</a><sup>1</sup>,
<a href="https://www.linkedin.com/in/joey-t-huang/">Joey Huang</a><sup>1</sup>,
<a href="https://www.linkedin.com/in/pengyuez/">Pengyue Zhu</a><sup>1</sup>,
<br>
Helen Chen<sup>‡1</sup>,
David Li<sup>‡1</sup>,
Jeffrey Li<sup>‡1</sup>,
Shawn Li<sup>‡1</sup>,
Andrew Zagula<sup>‡1</sup>,
Amy Zhao<sup>‡1</sup>,
Andrew Zhu<sup>‡1</sup>,
<br>
Sayaka Nakamura<sup>2</sup>,
Yuki Yamamoto<sup>2</sup>,
Jerry Jun Yokono<sup>2</sup>,
<br>
<a href="https://aaronmueller.github.io/">Aaron Mueller</a><sup>1</sup>,
<a href="https://bryanplummer.com/">Bryan A. Plummer</a><sup>1</sup>,
<a href="https://ai.bu.edu/ksaenko.html">Kate Saenko</a><sup>1</sup>,
<a href="https://venkatesh-saligrama.github.io/">Venkatesh Saligrama</a><sup>1</sup>,
<a href="https://boqinggong.github.io/">Boqing Gong</a><sup>1</sup>


<span class="author-block"><sup>1</sup>Boston University,</span>
<span class="author-block"><sup>2</sup>Sony Group Corporation</span>

<div class="is-size-7 publication-authors">
<span><sup>*</sup>Equal contribution. <sup>†</sup>Project lead. <sup>‡</sup>Equal contribution; work done as interns at Boston University.</span>
</div>

## Overview

This is the codebase of the "Baby model" introduced in BabyVLM-V2, adapted from the original [LLaVA repository](https://github.com/haotian-liu/LLaVA). In this repo, we'll refer to the model as BabyLLaVA-V2.

## Environment Setup

Install this package by cloning the repository and running the following command:

```bash
git clone git@github.com:ShawnKing98/BabyLLaVA-V2.git
cd BabyLLaVA-V2
pip install -e ".[train]"
pip install flash-attn==2.6.3 --no-build-isolation
```

## Data Preparation

The training data uses post-processed data from the SAYCam dataset, which is hosted on the Databrary platform. According to SAYCam's regulation terms, we cannot publicly share the dataset here. We are currently working towards putting our dataset on Databrary as well, please stay tuned for future updates. In the meantime, If you are interested in an early access to the dataset, feel free to email the author team with your IRB approval / proof of access to SAYCam, we are happy to assist you with access to the training dataset.

## Download Checkpoints

The checkpoints of our 1.1B BabyLLaVA-V2 model can be downloaded from Hugging Face. We provide several checkpoints after different phase of training, see below:

| Phase | Checkpoint |
|-------|------------|
| Vision Pretrain (phase 0 - vision) | [Link](https://huggingface.co/wsashawn/babyllava_v2_vision_backbone) |
| Joint Pretrain (phase 2) | [Link](https://huggingface.co/wsashawn/babyllava_v2_phase2) |
| Instruction Finetuning (phase 3) | [Link](https://huggingface.co/wsashawn/babyllava_v2_instruction_ft) |

## Usage
Please refer to the original [LLaVA repository](https://github.com/haotian-liu/LLaVA) for the usage of the BabyLLaVA model, most of the APIs should be the same.

## Training
Please see the figure below for an overview of the whole training pipeline. More details can be found in appendix A of the paper.

<img width="100%" src="images/babyllava_recipe.png">

### Phase 0 - Language:
Run the following command to train a compact language model (TinyLLaMA-1.1B) from scratch on the transcribed SAYCam. Please note that you need to change the data path in the script to point to your local SAYCam transcription data.

```bash
bash scripts/babyLLaVA_train/sweep_phase0.sh
```

### Phase 0 - Vision:

The vision backbone is trained from scratch separately, using Orhan et al.'s [training pipeline](https://github.com/eminorhan/dino). Please check [their paper](https://arxiv.org/abs/2305.15372) for more details about the vision backbone. 


### Phase 1:
Run the following command for phase 1 training. You need to change the data path / language backbone checkpoint / vision backbone checkpoint in the script to point to your local files.
```bash
bash scripts/babyLLaVA_train/sweep_phase1.sh
```

### Phase 2:
Run the following command for phase 2 training. In addition to the modification needed for phase 1, you also need to update the MLP connector to point to the checkpoint of phase 1 training.
```bash
bash scripts/babyLLaVA_train/sweep_phase2.sh
```

### Phase 3:

Run the following command for phase 3 training (instruction finetuning). This script kicks off instruction finetuning for several different tasks sequentially, you can modify the tasks being tuned by editing the corresponding fields.

```bash
bash scripts/babyLLaVA_train/sweep_instruction_ft.sh
```

## Citation

Please cite us if you use this repository in your work.

```bibtex
@misc{wang2026babyvlmv2developmentallygroundedpretraining,
      title={BabyVLM-V2: Toward Developmentally Grounded Pretraining and Benchmarking of Vision Foundation Models}, 
      author={Shengao Wang and Wenqi Wang and Zecheng Wang and Max Whitton and Michael Wakeham and Arjun Chandra and Joey Huang and Pengyue Zhu and Helen Chen and David Li and Jeffrey Li and Shawn Li and Andrew Zagula and Amy Zhao and Andrew Zhu and Sayaka Nakamura and Yuki Yamamoto and Jerry Jun Yokono and Aaron Mueller and Bryan A. Plummer and Kate Saenko and Venkatesh Saligrama and Boqing Gong},
      year={2026},
      eprint={2512.10932},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2512.10932}, 
}
```