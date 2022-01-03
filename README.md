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

   
## Results

## Tune the Baseline Trading Algorithm

### Step 1: Tune the training algorithm by adjusting the size of the training dataset. 
Answer the following question: What impact resulted from increasing or decreasing the training window?
The higher the size of the training window the more the departure between the actual and the predicted returns. The precision, recall and accuracy improves with shorter size of window.

### Step 2: Tune the trading algorithm by adjusting the SMA input features. 
Answer the following question: What impact resulted from increasing or decreasing either or both of the SMA windows?

### Set the short window and long window
short_window = 5
long_window = 80

### svm_testing_report
                  precision    recall  f1-score   support

        -1.0       0.39      0.03      0.06      1806
         1.0       0.56      0.96      0.70      2289

    accuracy                           0.55      4095

### logistic_regression_report
                  precision    recall  f1-score   support

        -1.0       0.43      0.27      0.33      1806
         1.0       0.56      0.72      0.63      2289

    accuracy                           0.52      4095

### Set the short window and long window
short_window = 4
long_window = 50

### svm_testing_report
                  precision    recall  f1-score   support

        -1.0       0.42      0.13      0.19      1826
         1.0       0.56      0.86      0.68      2321

    accuracy                           0.54      4147

### logistic_regression_report
              precision    recall  f1-score   support

        -1.0       0.44      0.21      0.28      1826
         1.0       0.56      0.79      0.66      2321

    accuracy                           0.53      4147

### Set the short window and long window
short_window = 10
long_window = 120

### svm_testing_report
                precision    recall  f1-score   support

        -1.0       0.46      0.02      0.04      1793
         1.0       0.56      0.98      0.71      2284

    accuracy                           0.56      4077

### logistic_regression_report
             precision    recall  f1-score   support

        -1.0       0.44      0.54      0.48      1793
         1.0       0.55      0.45      0.50      2284

    accuracy                           0.49      4077

   Considering the classification reports of both the SVM and the logistics regression with the short and long windows slices above, we can conclude that the precision, recall and accuracy improves or deterioriates depending on the size of the short or long windows. 

## Step 3: Choose the set of parameters that best improved the trading algorithm returns.
Save a PNG image of the cumulative product of the actual returns vs. the strategy returns, and document your conclusion in your README.md file.

   ### Original Model nn with training data
![cumulative return plot that shows the actual returns vs  the strategy returns](https://user-images.githubusercontent.com/88909565/147861181-08e0c682-d6a7-4ab4-bdf1-5fedf8714181.png)

The SVM model performed well from the beginning of the period until end of 2019. That’s when the actual and predicted returns start to differ. To truly find out how well this model works, the model was fitted with a different sets of pricing data, have it make predictions, backtest it, and then evaluate it against the actual performance of the asset with that trading strategy.

###  Review of the classification report to evaluate the model using the SVC model predictions and testing data
# svm_testing classification report
print(svm_testing_report)
              precision    recall  f1-score   support

        -1.0       0.43      0.04      0.07      1804
         1.0       0.56      0.96      0.71      2288

    accuracy                           0.55      4092
   macro avg       0.49      0.50      0.39      4092
weighted avg       0.50      0.55      0.43      4092

### Backtest the new model to evaluate its performance.
Save a PNG image of the cumulative product of the actual returns vs. the strategy returns for this updated trading algorithm, and write your conclusions in your README.md file.

![actual_vs_strategy1_cum_ plot](https://user-images.githubusercontent.com/88909565/147861193-2663773f-b1d2-4452-b059-8375f4528487.png)

Backtesting the SMA long-short trading strategy using the SVM model, the SVM model made trading decisions that strongly outperformed the actual returns in some market scenarios, as illustrated by the steep rise in the trading_algorithm_returns plot line. However, sometimes the SVM model also made decisions that underperformed relative to the actual returns, as illustrated by the steep decline in the trading_algorithm_returns plot line in Q42018 & Q1 2019. Overall, the SVM model's trading decisions, though more volatile, produced a higher cumulative return value than the original trading strategy's actual returns.

## Use a classification report to evaluate the model using the predictions and testing data
### Logistic_regression classification report
              precision    recall  f1-score   support

        -1.0       0.44      0.33      0.38      1804
         1.0       0.56      0.66      0.61      2288

    accuracy                           0.52      4092
   macro avg       0.50      0.50      0.49      4092
weighted avg       0.51      0.52      0.51      4092


## Answer the following questions: 
### 1. Did this new model perform better or worse than the provided baseline model? 

For the classification report using the SVC model using the original training data, the precision is 0.43 for the −1 class and 0.56 for the 1 class. The recall is 0.04 for the −1 class and 0.96 for the 1 class.
For the classification report using the Backtest new model prediction and testing data the precision is 0.44 for the −1 class and 0.56 for the 1 class. The recall is 0.33 for the −1 class and 0.66 for the 1 class.
## 2. Did this new model perform better or worse than your tuned trading algorithm?
We can conclude that the precision is similar for the training data and the testing data. But, the recall is much lower for the −1 class of the training data (0.04 vs. 0.33) and much higher for the 1 class (0.96 vs. 0.66). Overall, the accuracy score for the recall is only slightly better for the training data, at 0.55, than for the testing data, at 0.52.