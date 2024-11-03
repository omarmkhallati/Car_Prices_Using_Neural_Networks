# Sport Car Price Prediction using Feed-Forward Neural Network

This project is a deep learning model that predicts the price of sports cars using a feed-forward neural network built with PyTorch. The model utilizes several features, such as engine size and other specifications, to estimate car prices in USD. The dataset is preprocessed by standardizing both the features and the target variable. The model’s architecture consists of six fully connected layers with batch normalization and Leaky ReLU activation functions.

## Table of Contents
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training the Model](#training-the-model)
- [Evaluation](#evaluation)

## Project Overview
This project uses a deep learning model to predict car prices based on features such as engine size and other numeric specifications. It demonstrates data preprocessing, model training, and evaluation using mean squared error (MSE) as a performance metric.

## Installation
To run this project, you'll need the following dependencies:

`pip install pandas torch scikit-learn`

## Usage
### Data Preparation :
Place the dataset CSV file (Car_Price_Data.csv) in your preferred directory.
### Running The Code :
Modify the file path in the code.
`data = pd.read_csv ('YourPathTo/Car_Price_Data.csv')`
### Training The Model :
Run the code to train the model. The training process will output the training loss per epoch, and will provide the validation and test set mean squared errors (MSE).

## Model Architecture
The neural network consists of:
. An input layer of five features
. Five fully connected hidden layers with batch normalization and Leaky ReLU activation functions for better gradient flow and regularization.
. A single output layer to predict the car price in USD.

## Training The Model
The model is trained for 150 epochs with the Stochastic Gradient Descent (SGD) optimizer and a learning rate of 0.0001. The training process includes evaluation on a validation set to monitor the model’s generalization ability.

## Evaluation
Some sample outputs :  
`Epoch [1/150], Training Loss: 0.1453`  
`...`  
`Validation MSE: 0.0321`  
`Test MSE: 0.0314`
