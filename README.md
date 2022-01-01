# Challenge14 
Improve the existing algorithmic trading systems and maintain the firm’s competitive advantage in the market. 

# User Story

Role: Financial advisor at one of the top five financial advisory firms in the world

Goal: To enhance the existing trading signals with machine learning algorithms that can adapt to new data.

Reason: To improve the existing algorithmic trading systems and maintain the firm’s competitive advantage in the market. 

## Libraries and dependencies
The following libraries and dependencies are used:
import pandas as pd
import numpy as np
from pathlib import Path
import hvplot.pandas
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from pandas.tseries.offsets import DateOffset
from sklearn.metrics import classification_report


## General information
The steps for this Challenge are divided into the following sections:
• Establish a Baseline Performance
• Tune the Baseline Trading Algorithm
• Evaluate a New Machine Learning Classifier
• Create an Evaluation Report


## Technology
Jupyter notebook that contains data preparation, analysis, and visualizations %matplotlib inline Python


## Analysis and Model optimization
Tune the Baseline Trading Algorithm
Step 6: Use an Alternative ML Model and Evaluate Strategy Returns
In this section, you’ll tune, or adjust, the model’s input features to find the parameters that result in the best trading outcomes. You’ll choose the best by comparing the cumulative products of the strategy returns.

Step 1: Tune the training algorithm by adjusting the size of the training dataset.
To do so, slice your data into different periods. Rerun the notebook with the updated parameters, and record the results in your README.md file.

Answer the following question: What impact resulted from increasing or decreasing the training window?

Step 2: Tune the trading algorithm by adjusting the SMA input features.
Adjust one or both of the windows for the algorithm. Rerun the notebook with the updated parameters, and record the results in your README.md file.

Answer the following question: What impact resulted from increasing or decreasing either or both of the SMA windows?

Step 3: Choose the set of parameters that best improved the trading algorithm returns.
Save a PNG image of the cumulative product of the actual returns vs. the strategy returns, and document your conclusion in your README.md file.


## Deliverables
This project consists of three technical deliverables as follows:
• Preprocess data for use on a neural network model.
• Use the model-fit-predict pattern to compile and evaluate a binary classification model using a neural network.
• Optimize the neural network model.
. After finishing your models, display the accuracy scores achieved by each model, and compare the results. 
. Save each models as an HDF5 file, labeled as:
   Original model as -  AlphabetSoup.h5
   Alernative model 1 as - A1_AlphabetSoup.h5
   Alternative model 2 as - A2_AlphabetSoup.h5
   
## Results
The accuracy and loss of the three neural network models did not change significantly despite trying different activation functions for the hidden layers. The original model has the best result with accuracy of 0.537 and a loss of 0.903.

#### Original Model nn with training data


####  Review of the classification report to evaluate the model using the SVC model predictions and testing data
# svm_testing classification report
print(svm_testing_report)
              precision    recall  f1-score   support

        -1.0       0.43      0.04      0.07      1804
         1.0       0.56      0.96      0.71      2288

    accuracy                           0.55      4092
   macro avg       0.49      0.50      0.39      4092
weighted avg       0.50      0.55      0.43      4092



#### Backtest the new model to evaluate its performance.
Save a PNG image of the cumulative product of the actual returns vs. the strategy returns for this updated trading algorithm, and write your conclusions in your README.md file.



# Use a classification report to evaluate the model using the predictions and testing data
# Logistic_regression classification report
              precision    recall  f1-score   support

        -1.0       0.44      0.33      0.38      1804
         1.0       0.56      0.66      0.61      2288

    accuracy                           0.52      4092
   macro avg       0.50      0.50      0.49      4092
weighted avg       0.51      0.52      0.51      4092


Answer the following questions: Did this new model perform better or worse than the provided baseline model? Did this new model perform better or worse than your tuned trading algorithm?