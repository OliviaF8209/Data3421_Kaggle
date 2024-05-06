![](UTA-DataScience-Logo.png)



# Project Title

* **One Sentence Summary** In this challenge, participants predict what new products Santander bank customers will buy by June 28, 2016, based on their past purchases by May 28, 2016. https://www.kaggle.com/competitions/santander-product-recommendation/data  

## Overview

* This section could contain a short paragraph which include the following:
  * The goal of this challenge is to guess which new products Santander bank customers will buy by June 28, 2016, compared to what they already had by May 28, 2016. Participants use past customer behavior data to make these predictions accurately. It's like trying to predict what someone might add to their shopping cart based on what they've bought before.
  * The data includes demographics, account status, and owned financial products. It tracks customer codes, age, sex, residence country, and product ownership, aiding analysis of customer behavior and preferences
  * For the Kaggle challenge, I utilized historical data to forecast new product purchases for Santander customers. Additionally, I engineered a feature to count the number of products each consumer bought, aiding in understanding customer behavior.
  * The model received a Kaggle score of 0.00421

## Summary of Workdone

### Data

* Data:
  * Type: Tabular
    * santander_recommendation file was 240.06 MB and the train.csv had over 900,000+ rows.
    * For adjusted train was 500,000 and the test was 200,000 split
    * Input: CSV file of features, output: signal/background flag in 1st column.
 
#### Preprocessing / Clean up

* Used the missing percentage of each column and deleted ones with over 99%
* A good amount of numerical features were marked as categorical
* Making a tabulate out of the categorical features helped with the added column that counts all the products a consumer buys.

#### Data Visualization
Categorical Features after encoding: 
![image](https://github.com/OliviaF8209/Data3421_Kaggle/assets/143292953/5cc3941a-8a2d-4e2c-b721-80d744ae9fff)
![image](https://github.com/OliviaF8209/Data3421_Kaggle/assets/143292953/d578fafd-9f3f-4ca9-911a-40d139f5ece6)
![image](https://github.com/OliviaF8209/Data3421_Kaggle/assets/143292953/4a2db5aa-3c02-496b-9769-34851dfd33cc)
![image](https://github.com/OliviaF8209/Data3421_Kaggle/assets/143292953/0f16e551-a779-422d-a0aa-1f780b77d7db)
![image](https://github.com/OliviaF8209/Data3421_Kaggle/assets/143292953/918c33a1-3683-41da-8207-8bc317076446)
![image](https://github.com/OliviaF8209/Data3421_Kaggle/assets/143292953/46161e7f-2d93-433e-8a8c-dde01994db3a)
![image](https://github.com/OliviaF8209/Data3421_Kaggle/assets/143292953/87053d51-5fc0-446d-9962-f1c69c660663)

key = {
    'ind_empleado': {'A': 0, 'B': 1, 'F': 2, 'N': 3, 'P': 4},
    
    'pais_residencia': {'ES': 0, 'CH': 1, 'DE': 2, 'GB': 3, 'BE': 4, 'DJ': 5, 'IE': 6, 'QA': 7, 'US': 8, 'VE': 9,
                       'DO': 10, 'SE': 11, 'AR': 12, 'CA': 13, 'PL': 14, 'CN': 15, 'CM': 16, 'FR': 17, 'AT': 18,
                       'RO': 19, 'LU': 20, 'PT': 21, 'CL': 22, 'IT': 23, 'MR': 24, 'MX': 25, 'SN': 26, 'BR': 27,
                       'CO': 28, 'PE': 29, 'RU': 30, 'LT': 31, 'EE': 32, 'MA': 33, 'HN': 34, 'BG': 35, 'NO': 36,
                       'GT': 37, 'UA': 38, 'NL': 39, 'GA': 40, 'IL': 41, 'JP': 42, 'EC': 43, 'IN': 44},
                       
    'sexo': {'V': 0, 'H': 1},
    
    'tiprel_1mes': {'A': 0, 'I': 1, 'P': 2, 'R': 3},
    
    'indresi': {'S': 0, 'N': 1},
    
    'indext': {'N': 0, 'S': 1},
    
    'canal_entrada': {'KAT': 0, 'KHE': 1, 'KFC': 2, 'KHN': 3, 'KFA': 4, 'KHM': 5, 'KHL': 6, 'RED': 7, 'KHQ': 8,
                      'KHO': 9, 'KHK': 10, 'KAZ': 11, 'KEH': 12, 'KBG': 13, 'KHF': 14, 'KHC': 15, 'KHD': 16,
                      'KAK': 17, 'KAD': 18, 'KDH': 19, 'KGC': 20},
                      
    'indfall': {'N': 0, 'S': 1},
    
    'segmento': {'01 - TOP': 0, '02 - PARTICULARES': 1, '03 - UNIVERSITARIO': 2}
}

### Problem Formulation

*  Inputs include customer demographics and product ownership, while outputs are the predicted additional products. Various models like XGBoost were experimented with due to their ability to handle tabular data. Loss functions, optimizers, and hyperparameters such as learning rate and max depth were adjusted to optimize model performance.

### Training

* Training was conducted using XGBoost on a CPU-based system.
* Despite identical preprocessing, a feature shape mismatch error occurred between train and test sets.
* Training time varied based on data size and complexity.
* Training curves were monitored to assess model convergence.
* Training was stopped when validation loss plateaued or started to increase.
* Difficulties arose from feature inconsistencies, resolved by cadjusting preprocessing steps and data structures.

### Performance Comparison

* Key Performance Metric: Mean Squared Error (MSE) and Mean Absolute Error (MAE) were used to evaluate model performance.
 * Mean Squared Error (MSE): 5.354332437543275
 * Mean Absolute Error (MAE): 2.1902204738622957
   
* Feature importance plot visually represents the significance of each feature in model predictions.
* It highlights key features like `indrel_1mes` and `ind_empleado`, indicating their importance in predicting customer behavior and product purchases.
![Feature Importance Plot](https://github.com/OliviaF8209/Data3421_Kaggle/assets/143292953/4b7ff5c0-f0a4-4db2-a079-d97294a9beb6)

### Conclusions

* The XGBoost model's performance in the Santander Kaggle challenge fell short, indicated by its low Kaggle score (0.004) and high MSE/MAE
* Next time, I would like to add more features and experiment on those. The preprocessing steps will go through cross-validation more thoroughly aswell.

### Overview of files in repository

* Kaggle_3402(done).ipynb: Shows the code behind the preprocessing and visualizations. Also includes the training and test sets.
* ind_empleado_plot.png, indext_plot.png, indfall_plot.png, indresi_plot.png, segmento_plot.png, sexo_plot.png, tiprel_1mes_plot.png
*  Images of the visuals, specifically the categorical features

### Software Setup
Essential libraries (type as seen to import):
* import pandas as pd
* import numpy as np
* import matplotlib.pyplot as plt
* from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2

* from sklearn.model_selection import RandomizedSearchCV
* from xgboost import XGBRegressor

### Data

* Use the link above and download the file in the Data tab aka where the Dataset Description is located.
* For missing values, use XGBoost to fill them in
* Check which features are numerical or categorical

### Training

* How to train the dataset:
  - Ensure your dataset is preprocessed and split into training and testing sets.
  - Consider using XGBoost, a suitable algorithm for predicting customer behavior.
  - Initialize the XGBoost model with appropriate hyperparameters for your dataset.
  - Train the model using the training data to learn from customer behavior patterns.
  - Evaluate model performance using metrics like MSE and MAE on the test set.
  - Fine-tune hyperparameters if necessary to improve performance.
  - Validate results by comparing predicted product purchases with actual purchases from the test set (make sure the rows and users match).

#### Performance Evaluation

1. Prepare trained model and test dataset.
2. Use appropriate evaluation metrics such as MSE and MAE to assess model performance.
3. Make predictions on the test dataset using the trained model.
4. Compare the predicted values with the actual values to calculate the evaluation metrics.
5. Interpret the results to understand how well the model performs in predicting customer behavior and product purchases.


## Citations

* @misc{santander-product-recommendation,
    author = {Meg Risdal, Mercedes Piedra, Wendy Kan},
    title = {Santander Product Recommendation},
    publisher = {Kaggle},
    year = {2016},
    url = {https://kaggle.com/competitions/santander-product-recommendation}

"BLACKBOXAI. (2023). BLACKBOXAI: A World-Class AI Assistant. https://www.blackboxai.com/"







