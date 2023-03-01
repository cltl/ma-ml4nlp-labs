
To run in a local machine, create a new conda environment and install the needed packages:
```
conda create -n bert4ner python=3.8
conda activate bert4ner
pip install -r requirements.txt
```

To run in Google Colab:

1. Upload ONLY the notebook `bert_finetuner.ipynb` to your google drive.
2. Open the `bert_finetuner.ipynb` noteboook (it might take half a minute to load the resources).
3. To enable the GPU, you should locate the image on the top right that contains the text "RAM / Disk", click on it and at the bottom of the pane click on _Change runtime type_. Select the option _GPU_ from the dropdown _Hardware accelerator_ and save.
4. On the left pane, look for the folder icon (Files) and upload there the `bert_utils.py` using the _Upload to session storage_ button.
5. Create a `data` folder there (right click > New folder) and upload the files that you need for training, validation and testing. Make sure you move the files inside the data folder.
6. Return to the notebook. 
7. To install the packages needed, un-comment the first cell (delete the hashtags, and make sure each line starts with the `%` symbol, without any leading space)
8. Run the cell to install. This step can take a while...
9. Now you are all set to run the experiments! 
10. Make sure to select the desired train and development files (as well as the hyperparameters) before running. For now there is a mini _dummy_ file `data/trial_mini_data.conll` just to test that your code runs. You should replace them to train/validate/test properly.


