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



### Problem Formulation

* Define:
  * Input / Output
  * Models
    * Describe the different models you tried and why.
  * Loss, Optimizer, other Hyperparameters.

### Training

* Describe the training:
  * How you trained: software and hardware.
  * How did training take.
  * Training curves (loss vs epoch for test/train).
  * How did you decide to stop training.
  * Any difficulties? How did you resolve them?

### Performance Comparison

* Clearly define the key performance metric(s).
* Show/compare results in one table.
* Show one (or few) visualization(s) of results, for example ROC curves.

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






