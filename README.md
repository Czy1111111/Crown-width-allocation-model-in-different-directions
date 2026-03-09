# Crown-width-allocation-model-in-different-directions
# Crown Asymmetry Modelling using Interpretable Deep Learning

## Overview
This repository contains the code used in the study:
"Neighbourhood and species competition drive crown asymmetry in Chinese fir".

The project develops an interpretable deep-learning framework to model directional crown radius (CR) using forest inventory variables and neighbour information.

## Model
The model integrates:
- Gradient boosting feature extraction (LightGBM, CatBoost, HistGB)
- Embedding layers for categorical variables
- Attention-enhanced multilayer perceptron
- Hyperparameter optimisation with Optuna

## Requirements
Python 3.9+

Main packages:
- pytorch
- scikit-learn
- optuna
- numpy
- pandas

## Data
The dataset used in this study is not publicly available due to project restrictions.

Example input format is provided.

## Usage
Run the training script:

```bash
python train_model.py
