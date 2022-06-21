## Winning the CVPR'22 LOVEU-AQTC challenge 

### A Two-stage Function-centric Approach

This repo provides the **1st place** solution(code and checkpoint) of the CVPR'22 LOVEU-AQTC challenge.

[[Challenge Page]](https://showlab.github.io/assistq/)  [[Challenge Paper]](https://arxiv.org/abs/2203.04203)  [[Our paper]]() [[LOVEU@CVPR'22 Challenge]](https://sites.google.com/view/loveucvpr22/track-3?authuser=0) [[CodaLab Leaderboard]](https://codalab.lisn.upsaclay.fr/competitions/4642#results)

Click to know the task:

[![Click to see the demo](https://img.youtube.com/vi/3v8ceel9Mos/0.jpg)](https://www.youtube.com/watch?v=3v8ceel9Mos)

Model Architecture (see [[Our Paper]]() for details):

![image-20220619201014430](https://raw.githubusercontent.com/starsholic/pic/main/image-20220619201014430.png)


## Install

(1) PyTorch. See https://pytorch.org/ for instruction. For example,

```
conda install pytorch torchvision torchtext cudatoolkit=11.3 -c pytorch
```

(2) PyTorch Lightning. See https://www.pytorchlightning.ai/ for instruction. For example,

```
pip install pytorch-lightning
```

## Data

Download training set and testing set (without ground-truth labels) by filling in the [[AssistQ Downloading Agreement]](https://forms.gle/h9A8GxHksWJfPByf7).

Then carefully set your data path in the config file ;)

## Encoding

Before starting, you should encode the instructional videos, scripts, function-paras, QAs. See [encoder.md](https://github.com/starsholic/LOVEU-CVPR22-AQTC/tree/main/encoder).

## Training & Evaluation

Select the config file and simply train, e.g.,

```
CUDA_VISIBLE_DEVICES=0 python train.py --cfg configs/q2a_vit_xlnet.yaml
```

To inference a model, e.g.,

```
CUDA_VISIBLE_DEVICES=0 python inference.py --cfg configs/q2a_vit_xlnet.yaml CKPT "outputs/q2a_vit_xlnet/lightning_logs/version_0/checkpoints/epoch=5-step=155.ckpt"
```

The evaluation will be performed after each epoch. You can use Tensorboard, or just terminal outputs to record evaluation results.

To inference out best model, please download the checkpoint [here](https://drive.google.com/file/d/1nfgQLFE3ehiPeazSCMLavlUsAxoykH6C/view?usp=sharing), then

```
CUDA_VISIBLE_DEVICES=0 python inference.py --cfg configs/q2a_vit_xlnet.yaml CKPT ""
```

## Function-centric Approach Performance for LOVEU@CVPR2022 Challenge: 80 videos' QA samples for training, 20 videos' QA samples for testing

![image-20220619193605180](https://raw.githubusercontent.com/starsholic/pic/main/image-20220619193605180.png)

We obtained better results after the competition deadline as shown below.

| Model                                                        | Recall@1 ↑  | Recall@3 ↑  | MR (Mean Rank) ↓ | MRR (Mean Reciprocal Rank) ↑ |
| ------------------------------------------------------------ | ----------- | ----------- | ---------------- | ---------------------------- |
| Q2A(baseline) ([configs/q2a_gru+fps1+maskx-1_vit_b16+bert_b.yaml](https://github.com/showlab/Q2A/blob/master/configs/q2a_gru+fps1+maskx-1_vit_b16+bert_b.yaml)) | 30.2        | 62.4        | 3.2              | 3.2                          |
| Function-centric ([configs/q2a_vit_xlnet.yaml](configs/q2a_vit_xlnet.yaml)) | 44.8(+14.6) | 75.4(+13.0) | 2.60(+0.6)        | 3.87(+0.67)                    |

## Contact

Feel free to contact us if you have any problems: dwustc@mail.ustc.edu.cn, or leave an issue in this repo.