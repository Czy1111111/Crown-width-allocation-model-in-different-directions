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

## Workflow
The modelling workflow includes three main stages:
1. **Feature pre-training**
   - Extract nonlinear interaction features using gradient boosting models.
   - Output: feature matrices used by the neural network.
2. **Model training**
   - Train the attention-enhanced MLP model.
   - Hyperparameters are optimised using Optuna.
3. **Model interpretation**
   - Calculate SHAP values to quantify the contribution of candidate drivers.

## Usage
Run the scripts in the following order:
```bash
python Pre-training.py
python Train.py
python SHAP.py
