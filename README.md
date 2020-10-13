# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary

### Problem Statement
In this project we utilize Azure ML services to carry out a classification task on the [Bank Marketing Data Set]9https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv) (more details [here](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)) in order to predict if the client will subcribe to the term deposit or not.

**In 1-2 sentences, explain the solution: e.g. "The best performing model was a ..."**
### Solution
Azure's Hyperdrive and AutoML services were utilized for this task. The best solution using each were -

- Hyperdrive: Logistic Regression with a ~95% accuracy. The parameters for the best model were - ['--C', '0.5401408285696929', '--max_iter', '50']
- AutoML: Voting Ensemble with a 91.67% accuracy.

## Scikit-learn Pipeline

These were the steps involved in the pipeline (which can be found in the `train.py` script) -
- The dataset was loaded in using the `TabularDatasetFactory`. 
- The data was then preprocessed and cleaned. Some of the steps performed were (these can also be found in the `clean_data` function in `train.py`) -
  - Dropping null values
  - One-hot encode some of the features.
  - Separate the target values into its own dataframe
- The preprocessed data was then, using Scikit-learn's `train_test_split()` method, was converted into training and test set data with a 20% split.
- Scikit-Learn's `LogisticRegression` model was then fit upon the training data. The hyperparameters utilized for the model, `C` and `number of iterations`, were obtained from Hyperdrive.

After launching a workspace and cpu compute cluster in Azure ML, Hyperdrive was utilized in order to identify the best possible hyperparameter values for our model. The configuration for this can be found in `udacity-project.ipynb`, but here are the details as well -

The `RandomParameterSampling` was utilized to randomly select hyperparameter values. For `C`, I chose a uniform distribution (0.1, 0.9), and for `max_iter` I gave it a choice of 4 values - (25, 50, 75, 100). Random sampling, as the name suggests, randomly selects hyperparameter values from a search space defined by us. This can often help in doing a sort of preliminary search and then refining the values to try and improve the model.

**Benefits** - `RandomParameterSampling` allows us to define hyperparameters from either a discrete or continuous distribution. As can be seen from the hyperparameters required for this particular model, this was important - continuous range for `C` and discrete values for `max_iter`.

For the Early Stopping policy I selected the `BanditPolicy` with an `evaluation_interval` of 2 and a `slack_factor` of 0.5.

**Benefits** - The `BanditPolicy`is useful for early termination when the primary metric (accuracy, in our case) is not within the `slack_factor` value in comparison to the best performing model/run. This helps make the process more efficent and cost-effective. 


## AutoML
For the AutoML approach, the process was similar to some extent. I loaded in the dataset, cleaned it, and then split it into training and test set. This was similar to the previous pipeline. I had to concatenate the training features with the previously removed target labels to feed the training data into AutoML as we have to specific the targett column in the configuration. Then I utilized AutoML on the dataset.

I kept the configuration simple. I only utilized cross-validation set to 3.

AutoML trained through a lot of different models, and those runs can be found in the `udacity-project.ipynb` Notebook. The best performing model was the Voting Ensemble as mentioned previously.

The Notebook currently displays the run parameters as well as the metrics. I am sharing some of those parameters below -

- n_estimators=800
- num_leaves=53
- reg_lambda=0.631578947368421
- weights=[0.2, 0.1, 0.1, 0.1, 0.1, 0.2, 0.1, 0.1]
- ensemble_iterations = 15


## Pipeline comparison
As mentioned above, following were the outcomes of the two pipelines - 

- Hyperdrive: Logistic Regression with a ~95% accuracy.
- AutoML: Voting Ensemble with a 91.67% accuracy.

Hyperdrive in this case performed better, but I also didn't explore much of AutoML's capabilities. Given the latter, the result was still very impressive.

Additionally, AutoMl has multiple automated checks in place such as for - 

- Class balancing detection
- Missing feature values imputation
- High cardinality feature detection

Thanks to AutoML (and the previous reviewer pointing it out), I was made aware that the data we're working with is imbalanced. I can use that information to my benefit to improve upon the model and results next time. Hyperdrive, unfortunately, is limited in this regards. 

## Future work
I would like to experiment with more early stopping policies as that is not something I am familiar with, but I think that has an impact on making the process more efficient to be able to explore more. 

However, given what I learned, I think a reasonable approach is to first use AutoML to identify a model and then narrow down on hyperparameters using Hyperdrive and refine from there. 

## Proof of cluster clean up
- I did include the code in the Notebook and it ran without issues.

## Other issues (These have now been resolved)

There were just one main issue I faced with the project -

- Sometimes, it gave me messages that it couldn't find the primary metric even though the metric matched (with no spelling errors). There was no way to effectively debug this

Other than that, you will notice an error in the Notebook - 

> NameError: name 'RunDetails' is not defined

I had to restart the Kernel for the AutoML part, and I didn't run the cell that imported `RunDetails`. I didn't want to run AutoML again, so I have left the error as is for now. I have run it in the next cell, however.
