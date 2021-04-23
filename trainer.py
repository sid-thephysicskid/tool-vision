from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.client import device_lib

from sklearn.metrics import classification_report, confusion_matrix

import os
import numpy as np
import argparse
import hypertune
import shutil
from datetime import datetime
from datahandler import download_data_to_local_dir, upload_data_to_bucket


print("Tensorflow is running on the following devices: ")
print(device_lib.list_local_devices())

def build_model(nbr_classes):

    base_model = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

    head_model = base_model.output
    head_model = Flatten()(head_model)
    head_model = Dense(512, activation='relu')(head_model)
    head_model = Dropout(0.5)(head_model)
    head_model = Dense(nbr_classes, activation="softmax")(head_model)

    model = Model(inputs=base_model.input, outputs=head_model)

    for layer in base_model.layers:
        layer.trainable = False

    return model

def build_data_pipelines(batch_size, train_data_path, val_data_path, test_data_path):

    train_augmentor = ImageDataGenerator(
        rescale = 1. / 255,
        rotation_range=25,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    val_augmentor = ImageDataGenerator(
        rescale = 1. / 255
    )

    train_generator = train_augmentor.flow_from_directory(
        train_data_path,
        class_mode="categorical",
        target_size=(224,224),
        color_mode="rgb",
        shuffle=True,
        batch_size=batch_size
    )

    val_generator = val_augmentor.flow_from_directory(
        val_data_path,
        class_mode="categorical",
        target_size=(224,224),
        color_mode="rgb",
        shuffle=False,
        batch_size=batch_size
    )

    test_generator = val_augmentor.flow_from_directory(
        test_data_path,
        class_mode="categorical",
        target_size=(224,224),
        color_mode="rgb",
        shuffle=False,
        batch_size=batch_size
    )


    return train_generator, val_generator, test_generator


def get_number_of_imgs_inside_folder(directory):

    totalcount = 0

    for r, d, f in os.walk(directory):
        for file in f:
            _, ext = os.path.splitext(file)
            if ext in [".png", ".jpg", ".JPEG"]:
                totalcount = totalcount + 1

    return totalcount

    
def train(path_to_data, batch_size, epochs, learning_rate, models_bucket_name):
    #Path to folders
    train_data_path = os.path.join(path_to_data, 'training')
    val_data_path = os.path.join(path_to_data, 'validation')
    test_data_path = os.path.join(path_to_data, 'test')

    total_train_imgs = get_number_of_imgs_inside_folder(train_data_path)
    total_val_imgs = get_number_of_imgs_inside_folder(val_data_path)
    total_test_imgs = get_number_of_imgs_inside_folder(test_data_path)

    print(total_train_imgs, total_val_imgs, total_test_imgs)


    train_generator, val_generator, test_generator = build_data_pipelines(
        batch_size=batch_size,
        train_data_path=train_data_path,
        val_data_path=val_data_path,
        test_data_path=test_data_path
    )
    
    #count the number of classes automatically
    classes_dict=train_generator.class_indices
    model = build_model(nbr_classes=len(classes_dict.keys()))

    optimizer = Adam(lr=learning_rate)#1e-5

    early_stopping = EarlyStopping(monitor='val_loss', patience=8)
    
    path_to_save_model = './tmp'
    if not os.path.isdir(path_to_save_model):
        os.makedirs(path_to_save_model)

    
    
    ckpt_saver = ModelCheckpoint(
        path_to_save_model,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        save_freq='epoch',
        verbose=1
    )
    
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    model.fit_generator(
        train_generator,
        steps_per_epoch=total_train_imgs // batch_size,
        validation_data=val_generator,
        validation_steps=total_val_imgs // batch_size,
        epochs=epochs,
        callbacks=[early_stopping, ckpt_saver]
    )

    print("[INFO] Test phase...")

    predictions = model.predict(test_generator)
    predictions_idxs = np.argmax(predictions, axis=1)

    my_classification_report = classification_report(test_generator.classes, predictions_idxs, target_names=test_generator.class_indices.keys())
    
    my_confusion_matrix = confusion_matrix(test_generator.classes, predictions_idxs)

    print("[INFO] Classification report : ")
    print(my_classification_report)

    print("[INFO] Confusion matrix : ")
    print(my_confusion_matrix)
    print("Starting testing using model.evaluate_generator")
    scores = model.evaluate_generator(test_generator)
    print("Done evaluating!")
    loss = scores[0]
    print(f"loss for hyptertune = {loss}")

    now = datetime.now().strftime("%Y%m%d%H%M%S")
    zipped_folder_name = f'trained_model_{now}_loss_{loss}'
    path_to_zipped_folder = '/usr/src/app/' + zipped_folder_name + '.zip'
    shutil.make_archive(zipped_folder_name, 'zip', '/usr/src/app/tmp')

    upload_data_to_bucket(models_bucket_name, path_to_zipped_folder, zipped_folder_name)


    hpt = hypertune.HyperTune()
    hpt.report_hyperparameter_tuning_metric(hyperparameter_metric_tag='loss', 
                                            metric_value=loss, global_step=epochs)


if __name__=='__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--bucket-name", type=str, help="Bucket name on google cloud storage",
                        default= "tools-data-bucket")
    parser.add_argument("--models_bucket_name", type=str, help="Bucket name on google cloud storage for saving trained models",
                        default= "trained-models-tool-vision")
    parser.add_argument("--batch_size", type=int, help="Batch size used by model",
                        default=2)
    parser.add_argument("--learning_rate", type=float, help="Batch size used by the deep learning model", 
                        default=1e-5)
    args = parser.parse_args()

    print("Downloading data started...")
    download_data_to_local_dir(args.bucket_name, "./data")
    print("Download finished")

    path_to_data = './data'
    train(path_to_data, args.batch_size, 20, args.learning_rate, args.models_bucket_name)