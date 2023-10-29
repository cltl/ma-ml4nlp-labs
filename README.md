# ma-ml4nlp-labs

## overview

This repository provides notebooks and scripts for the course 'Machine Learning in NLP'.

It is structured as follows:

/code

is the main directory for finding notebooks and scripts. This is the only directory you should submit when submitting your code. 

It currently has three subfolders:
/settings : for configuration files or other files with fixed definitions that support your code 
/assignment1 : where you will find a notebook that support Assignment 1 (basic_system.ipynb) as well as a notebook from an old assignment that may contain useful functions (evaluation_assignment_2019_2022.ipynb)
/assignment2: where you will find a notebook supporting Assignment 2
/assignment3: where you will find a notebook supporting feature ablation analysis as well as code for two advanced systems
/bert4ner: a repository with code to train Bert for NERC

/data

is the default location to put the data you are using in your local directory. It currently contains three small testfiles.
You can place training and evaluation data in this directory on your local machine. Do *not* submit the data as part of your assignments.
If you manipulate data (add things manually or through a script), please provide instructions on this in the README.txt that you include with your code.

/models

is the default location for placing language models that you may use. Here it holds as well: do *not* include language models in your submissions: they are huge and you will not be able to upload your code in Canvas if you include them in your zip.

## submitting your code

1. Include a README.txt in your /code directory. This README should specify everything that someone else needs to know to run your code. 
For instance:
- which arguments must and which can be used when running your program
- if you use fixed paths in your code, where which files should be placed and how they should be called. This also applies if you use a fixed path to call a language model
-  any changes you made to the standard setup

2. Rename your code directory to your student identification number

3. Create a .zip or tar.gz file of this directory.

4. Upload under "code submission" in Canvas

