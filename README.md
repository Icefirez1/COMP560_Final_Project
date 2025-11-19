# COMP560 Final Project

This repository contains our final project for COMP 560. Our aim is to train a classifier model to be able to detect the rank of a player based on their stats of a game. It includes the data, analyses, model development, and an application that integrates the selected model with best performance.

## Contents

- `data/` — Raw and processed datasets used in the analysis and model training.  
- `league_data_analysis.ipynb` — Jupyter notebook for exploratory data analysis and visualizations.  
- `model_training.ipynb` — Jupyter notebook covering feature engineering, model selection, training, and evaluation.  
- `mainapp.py` — Streamlit application to show the usage of the selected model. 

## Usage

1. Clone the repository to your local machine.  
2. Install required dependencies
3. Run `league_data_analysis.ipynb` to explore the data and understand feature relationships.  
4. Run `model_training.ipynb` to reproduce model training and evaluation results.  
5. Execute `mainapp.py` to launch the streamlit application using the trained model.

## Machine Learning Content Utilized
We utilize various classifier algorithms to explore which kind of model works best for this kind of data. The models we use are as follows: 
- Logistic Regression 
- Ridge Classification with Cross Validated Parameters (sklearn's Ridge Regression for Classification)
- Decision Tree utilizing Gini Impurity (default for sklearn)
- Random Forest Tree Ensemble 
- Adaboost Tree Ensemble
- Gradient Boosting Tree Ensemble
