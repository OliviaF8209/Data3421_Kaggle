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

Include only the sections that are relevant an appropriate.

### Data

* Data:
  * Type: Tabular
    * santander_recommendation file was 240.06 MB and the train.csv had over 900,000+ rows.
    * For adjusted train was 500,000 and the test was 200,000 split
    * Input: CSV file of features, output: signal/background flag in 1st column.
 
#### Preprocessing / Clean up

* Used the missing percentage of each column and deleted ones with over 99%
* A good amount of numerical features were marked as categorical

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
* Difficulties arose from feature inconsistencies, resolved by careful examination of preprocessing steps and data structures.

### Performance Comparison

* Key Performance Metric: Mean Squared Error (MSE) and Mean Absolute Error (MAE) were used to evaluate model performance.
 * Mean Squared Error (MSE): 5.354332437543275
 * Mean Absolute Error (MAE): 2.1902204738622957
* 
![Feature Importance Plot](https://github.com/OliviaF8209/Data3421_Kaggle/assets/143292953/4b7ff5c0-f0a4-4db2-a079-d97294a9beb6)

### Conclusions

* State any conclusions you can infer from your work. Example: LSTM work better than GRU.

### Future Work

* What would be the next thing that you would try.
* What are some other studies that can be done starting from here.

## How to reproduce results

* In this section, provide instructions at least one of the following:
   * Reproduce your results fully, including training.
   * Apply this package to other data. For example, how to use the model you trained.
   * Use this package to perform their own study.
* Also describe what resources to use for this package, if appropirate. For example, point them to Collab and TPUs.

### Overview of files in repository

* Describe the directory structure, if any.
* List all relavent files and describe their role in the package.
* An example:
  * utils.py: various functions that are used in cleaning and visualizing data.
  * preprocess.ipynb: Takes input data in CSV and writes out data frame after cleanup.
  * visualization.ipynb: Creates various visualizations of the data.
  * models.py: Contains functions that build the various models.
  * training-model-1.ipynb: Trains the first model and saves model during training.
  * training-model-2.ipynb: Trains the second model and saves model during training.
  * training-model-3.ipynb: Trains the third model and saves model during training.
  * performance.ipynb: loads multiple trained models and compares results.
  * inference.ipynb: loads a trained model and applies it to test data to create kaggle submission.

* Note that all of these notebooks should contain enough text for someone to understand what is happening.

### Software Setup
* List all of the required packages.
* If not standard, provide or point to instruction for installing the packages.
* Describe how to install your package.

### Data

* Point to where they can download the data.
* Lead them through preprocessing steps, if necessary.

### Training

* Describe how to train the model

#### Performance Evaluation

* Describe how to run the performance evaluation.


## Citations

* Provide any references.







