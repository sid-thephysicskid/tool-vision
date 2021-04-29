# tool-vision
With massive labor shortage in construction industry, there's a massive opportunity in US market alone to standardize and automate all the low level tasks involved in construction proecess. This app is a basic MVP for a potential robot that can use machine vision to classify different tools. For right now I have included 8 most common ones, namely : flashlight, hammer, handsaw, level, pliers, screwdriver, tape measure, toolbox

The first version (current) is simple image classification using Keras/Tensorflow. (using CNNs)
The second iteration will be multiple object detection and localization.(YOLO most likely)

# Summary
The model training and deployment is performed on Google's Cloud Platform.
The data ingestion and training pipeline is dockerized and pushed to Google Container Registry. The input data is goes in Cloud Storage buckets and Google's AI platform is leveraged to train the model. config.yaml helps with running multiple parallel trials while testing different values of the hyperparameters. The best model is saved in the cloud storage after the training is complete.
For more detaiils on how to train custom containers on Google Cloud Platform, please refer to this guide: https://cloud.google.com/ai-platform/training/docs/custom-containers-training


All the dependencies are saved in requirements.txt and can be installed using 'pip install -r requirements.txt'. Using a virtual environment is highly advised. Please uncomment the tensorflow and tensorboard libraries in the requirements file if you need to test or train the model in local environment.

The frontend app development is underway.

# Data Collection
The raw dataset used for this project can be downloaded directly from here: https://drive.google.com/file/d/1VH3jqREFB3rFrlbmhkwH4hIyNBGR1byX/view?usp=sharing

There are 2149 images more or less evenly distributed among the 8 classes, and sourced directly from reddit, amazon and lowe's product pages.
In my experience, a simple script to scrape images from websites only works (generally) for <100 images at a time. Often websites block requests after a threshold. To source more images, there are 3 solutions (amongst others I am sure):

1) Use a vpn and send requests from different IP addresses. 

2) Download selenium, webdriver, and chrome. This has its limitations, but here's a good writeup if you want to go down that path: https://radiant-brushlands-42789.herokuapp.com/levelup.gitconnected.com/how-to-download-google-images-using-python-2021-82e69c637d59

3) Use an extension like Fatkun batch image downloader (https://chrome.google.com/webstore/detail/fatkun-batch-download-ima/nnjjahlikiabnchcpehcpkdeckfgnohf?hl=en) and ripme: https://github.com/RipMeApp/ripme

Since I wanted the images to come from a variety of sources, including Reddit, to source real life pics taken by everyday people. I went with approach #3.


# Data pipeline

handler.py creates the pipeline for feeding the data to the model by downloading it from the Cloud Storage. A storage client is created that uses the credentials file which can be sourced from the service account under IAM roles. Please be sure to add the folder in .gitignore so you don't accidentally share that publicly.

To split data into training, validation, and training, you can write a custom function (which is what I started with), but there's also a python package called 'split_folders' that can be implemented in one line. More details: https://pypi.org/project/split-folders/


# Training

trainer.py trains the MobileNetV2 model. 
Currently only the top layer is removed and a dense layer with dropout is added to finally output to the appropriate number of classes (8, in this case). Potential improvements in accuracy and F1 scores can be seen if we train more layers,those tests haven't been performed.
2 callbacks, namely EarlyStopping and ModelCheckpoint are employed. ReducingLR on plateau is perhaps on utility as well for further optimization.

.. to be continued
