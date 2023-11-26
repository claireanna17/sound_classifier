# sound_classifier

Final project for McGill AI Society Intro to ML Bootcamp (MAIS 202 - Winter 2023). 

Training data retrieved from [zenodo](https://zenodo.org/records/2552860#.XFD05fwo-V4).

Source code inspiration/references from [kaggle](https://www.kaggle.com/code/blackjacl/pytorch-audio).

## Project description

This Sound Classifier project uses a web app that allows the user to input an audio clip and outputs the sound's label. 
We finetuned a CNN model based off of Kaggle source code and Torchvision model “efficientnet_b0”. The training and testing data is from Zenodo.

## Running the app

1. Download (into a new project/folder):
   * FSDKaggle2018.meta/train_post_competition.csv
   * models/modelV1.2.pth
   * prediction.py
   * app.py
   * templates/index.html
2. Install necessary packges (might need to do 'pip install opencv-python')
3. Run following code:


```
streamlit run app.py
```


## Repository organization

This repository contains the scripts used to both train the model and build the web app.

1. deliverables/
	* Deliverables submitted to the MAIS 202 TPM's.
2. MAISproject.ipynb
	* The notebook that contains the code used for training the model.
3. MAISproject.py
	* The notebook code exported as a .py file.
4. FSDKaggle2018.meta/
	* contains the metadata (labels) of the training and testing data.
5. templates/
	* HTML template for landing page
6. prediction.py
	* The model's prediction code used by the webapp
7. app.py
	* The webapp code

