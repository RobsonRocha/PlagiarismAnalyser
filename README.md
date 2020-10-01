# PlagiarismAnalyser

Second project from [Machine Learning Engineer Nanodegree Program Udacity course](https://www.udacity.com/course/machine-learning-engineer-nanodegree--nd009t)

## Motivation

This project consists in implement two algorithms that are capable to predict if a text is a plagiarized from its source.
The dataset used to train and test is a slightly modified version of a dataset created by Paul Clough (Information Studies) and Mark Stevenson (Computer Science), at the University of Sheffield. You can read all about the data collection and corpus, at their university [webpage.at](https://ir.shef.ac.uk/cloughie/resources/plagiarism_corpus.html).
The porpose is compare the perfomance of that algorithms and show how to use AWS Sagemaker features to do that.
This project still contains 3 notebooks that explain how to prepare the data, train, test and creation of an endpoint to access the trained model.

## Training and testing

All process to get the data file, transform it, train, test create the endpoint and use it is explain in notebook file [1_Data_Exploration.ipynb](./1_Data_Exploration.ipynb)

Everything that it's necessery to create feature engineering is explained in notebook file [2_Plagiarism_Feature_Engineering.ipynb](./2_Plagiarism_Feature_Engineering.ipynb)

And the efective training and tests are implemented here [3_Training_a_Model.ipynb](./3_Training_a_Model.ipynb)
