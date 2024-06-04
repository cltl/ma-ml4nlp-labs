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



