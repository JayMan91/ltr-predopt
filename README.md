# Predict and Optimize: Through the Lens of Learning to Rank
This repository contains the official PyTorch implementation of [Predict and Optimize: Through the Lens of Learning to Rank](https://arxiv.org/abs/2112.03609) (accepted for publication in ICML,2022).


## Overview
This paper generalizes the work of [Contrastive Losses and Solution Caching for Predict-and-Optimize](https://doi.org/10.24963/ijcai.2021/390). This paper starts with the idea of caching solutions and frames the predict-and-optimze as a learning to rank problem, where models are trained to learn the partial ordering of the pool of feasible solutions. The proposed loss functions are motivated from the learning to rank literature and these loss functions, do not include the optimization problem menaing they are end-to-end differentiable. 
This repository contains pointwise, pairwise and listwise ranking loss functions for 3 predict-and-optimze problems.

### Requirements
Contents of requirements.txt

```
  pandas==1.2.3
  ortools==9.1.9490
  numpy==1.21.5
  pytorch_lightning==1.4.9
  networkx==2.3
  gurobipy==9.5.0
  tqdm==4.62.3
  torch==1.10.0
  torchmetrics==0.5.1
  tensorboard==2.8.0
  scikit_learn==1.1.1
 ```
