# sound_classifier

Final project for McGill AI Society Intro to ML Bootcamp (MAIS 202 - Winter 2023). 

Training data retrieved from [zenodo](https://zenodo.org/records/2552860#.XFD05fwo-V4).

## Project description

This Sound Classifier project using a web app that allows the user to input an audio clip and outputs the sound's label. 
We built the model based off of Kaggle source code and Torchvision model “efficientnet_b0”. The training and testing datat is from Zenodo. Our model's architecture utilizes CNN.

## Running the app



```
python app.py
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

