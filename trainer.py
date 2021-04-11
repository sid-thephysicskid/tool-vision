

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam

from sklearn.metrics import classification_report, confusion_matrix

import os
import numpy as np
import argparse

from datahandler import download_data_to_local_dir

from tensorflow.python.client import device_lib

print("Tensorflow is running on the following devices: ")
print(device_lib.list_local_devices())

def build_model(nbr_classes):

    base_model = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

    head_model = base_model.output
    head_model = Flatten()(head_model)
    head_model = Dense(512)(head_model)
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

    
def train(path_to_data, batch_size, epochs):
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

    optimizer = Adam(lr=1e-5)

    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    model.fit_generator(
        train_generator,
        steps_per_epoch=total_train_imgs // batch_size,
        validation_data=val_generator,
        validation_steps=total_val_imgs // batch_size,
        epochs=epochs
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

if __name__=='__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--bucket-name", type=str, help="Bucket name on google cloud storage",
                        default= "tools-data-bucket")
    parser.add_argument("--batch_size", type=int, help="Batch size used by model",
                        default=8)

    args = parser.parse_args()

    print("Downloading data started...")
    download_data_to_local_dir(args.bucket_name, "./gcp_data")
    print("Download finished")

    path_to_data = './gcp_data/dummy'
    train(path_to_data, args.batch_size, 1)