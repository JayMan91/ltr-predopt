# Predict and Optimize: Through the Lens of Learning to Rank
This repository contains the official PyTorch implementation of [Decision-Focused Learning: Through the Lens of Learning to Rank](https://proceedings.mlr.press/v162/mandi22a.html) (accepted for publication in ICML,2022).


## Overview
![Alt text](AbstractFig.png?raw=true "Optional Title")
This paper generalizes our previous work- [Contrastive Losses and Solution Caching for Predict-and-Optimize](https://doi.org/10.24963/ijcai.2021/390). Here we start with the idea of caching solutions explored in our previous work. Here we frame the predict-and-optimze as a learning to rank problem, where models are trained to learn the partial ordering of the pool of feasible solutions. The proposed loss functions are motivated from the learning to rank literature and these loss functions, do not include the optimization problem menaing they are end-to-end differentiable. 
This repository contains pointwise, pairwise and listwise ranking loss functions for 3 predict-and-optimze problems.

### Requirements 
Recommended python version: 3.6.10+, 3.7.6+ and 3.8.1+

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
  protobuf==3.20.1
 ```
To run the codes and reproduce the results, go inside the folder and run the python scipts whose name starts with *test*.
