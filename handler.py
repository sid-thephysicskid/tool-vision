import glob
import shutil
import os
import random
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

#import split_folders

from google.cloud import storage

path_to_credentials = './credentials/train-tool-vision-72affcbebda0.json'

tool_classes = ['flashlight', 'hammer', 'handsaw', 'level', 'pliers', 'screwdriver', 
                'tape measure', 'toolbox']


# def split_data_into_class_folders(path_to_data, class_id):

#     imgs_paths = glob.glob(path_to_data + '*.jpg')

#     for path in imgs_paths:

#         basename = os.path.basename(path)

#         if basename.startswith(str(class_id) + '_'):

#             path_to_save = os.path.join(path_to_data, food_classes[class_id])

#             if not os.path.isdir(path_to_save):
#                 os.makedirs(path_to_save)

#             shutil.move(path, path_to_save)



def visualize_images(path_to_data):

    imgs_paths = []
    labels = []

    for r, d, f in os.walk(path_to_data):
        for file in f:
            if file.endswith(".jpg"):
                imgs_paths.append(os.path.join(r, file))
                labels.append(os.path.basename(r))

    fig = plt.figure()

    for i in range(16):
        chosen_index = random.randint(0, len(imgs_paths)-1)
        chosen_img = imgs_paths[chosen_index]
        chosen_label = labels[chosen_index]

        ax = fig.add_subplot(4,4, i+1)
        ax.title.set_text(chosen_label)
        ax.imshow(Image.open(chosen_img))

    fig.tight_layout(pad=0.05)
    plt.show()


def get_images_sizes(path_to_data):

    imgs_paths = []
    widths = []
    heights = []

    for r, d, f in os.walk(path_to_data):
        for file in f:
            if file.endswith(".jpg"):

                img = Image.open(os.path.join(r, file))
                widths.append(img.size[0])
                heights.append(img.size[1])
                img.close()

    mean_width = sum(widths) / len(widths)
    mean_height = sum(heights) / len(heights)
    median_width = np.median(widths)
    median_height = np.median(heights)

    return mean_width, mean_height, median_width, median_height


def download_data_to_local_directory(bucket_name, local_directory):

    storage_client = storage.Client.from_service_account_json(path_to_credentials)
    blobs = storage_client.list_blobs(bucket_name)

    if not os.path.isdir(local_directory):
        os.makedirs(local_directory)

    for blob in blobs:

        joined_path = os.path.join(local_directory, blob.name)

        if os.path.basename(joined_path) == '':
            if not os.path.isdir(joined_path):
                os.makedirs(joined_path)

        else:
            if not os.path.isfile(joined_path):
                if not os.path.isdir(os.path.dirname(joined_path)):
                    os.makedirs(os.path.dirname(joined_path))
                    
                blob.download_to_filename(joined_path)

def upload_data_to_bucket(bucket_name, path_to_data, bucket_blob_name):

    storage_client = storage.Client.from_service_account_json(path_to_credentials)
    bucket = storage_client.get_bucket(bucket_name)

    blob = bucket.blob(bucket_blob_name)
    blob.upload_from_filename(path_to_data)

if __name__ == '__main__':

    split_data_switch = False
    visualize_data_switch = False
    print_insights_switch = False
    list_blob_switch = False
    download_data_switch = True

    path_train_data = './data_dummy/train/'
    path_val_data = './data_dummy/val/'
    path_test_data = './data_dummy/test/'

    if split_data_switch :
        split_folders.ratio("./tools-images-dataset", output="data", seed=1337, ratio=(0.8,0.1,0.1))

    if visualize_data_switch:
        visualize_images(path_train_data)


    if print_insights_switch:
        mean_width, mean_height, median_width, median_height = get_images_sizes(path_train_data)

        print(f"mean width = {mean_width}")
        print(f"mean height = {mean_height}")
        print(f"median width = {median_width}")
        print(f"median height = {median_height}")

    if list_blob_switch:
        blobs = list_blobs('tool-vision-data')

        for blob in blobs:
            print(blob.name)

    if download_data_switch:
        download_data_to_local_directory("tool-vision-data", "./data")



