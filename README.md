# Machine Learning for NLP Course code

This repository provides (assignment) notebooks and scripts for the course *Machine Learning in NLP*.

## Submitting your code

### What to submit
* You should only submit the `code/` folder of your assignment. In particular, you should not upload datasets or model checkpoints (see [Hints](#hints) below).
* Your code folder should contain, beside your code:
  * a README.md file explaining what your code does and how to run it:
    * What your code does: this does not need to be long, just a one or two sentence summary
    * Installation and setup instructions if needed, including paths to input data
    * For scripts, the command call(s) to the code 
  * a `requirements.txt` file if your code depends on more libraries than specified for the given assignment

### How to submit
1. Rename your code directory to your student identification number
2. Create a .zip or tar.gz file of this directory.
3. Upload under "code submission" in Canvas

## Hints

* You may want to place your data in a `data/` subfolder and models into a `models/` subfolder, and put these outside of the `code/` folder. There are no fixed rules in such structure, but it will help making your projects more approachable and reusable. It is also good practice to separate heavy from light files on one hand, and version-controlled from input and temporary files on the other hand. You can use the `.gitignore` file to specify exclusion patterns.
* Do not sow hard paths and other hard-coded variables through your code. They will make your **code break** and they are **hard to find** once it does.   
  * You can use an argument parser like [argparse](https://docs.python.org/3/library/argparse.html) or [click](https://click.palletsprojects.com/en/8.1.x/) to set defaults for variables for scripts;
  * Setup variables and variables that are unlikely to change can best be put in a config file (`json` and `yaml` are common formats for this; they can be parsed with [json](https://docs.python.org/3/library/json.html) and [pyyaml](https://pypi.org/project/PyYAML/)). They can be input as an argument to your scripts or as data into notebooks. This is practical if there are a lot of them, and you would like to separate the code from specific uses of it;
  * For single-use scripts and notebooks, it is still good practice to isolate usage-specific variables. Declare them as constants (all caps) in the same place of your script/notebook (and make them easy to find, i.e. all at the top or at the bottom).

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

