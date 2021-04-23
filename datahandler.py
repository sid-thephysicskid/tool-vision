import os
import random
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from google.cloud import storage

path_to_credentials = './credentials/object-detection-310321-790e8cdb90b9.json'

tool_classes = ['hammer', 'screwdriver','wrench']

def visualize_data(path_to_data):

    img_path = []
    labels = []

    for root,dir,file in os.walk(path_to_data):
        for f in file:
            if f.endswith(".jpg"):
                img_path.append(os.path.join(root,f))
                labels.append(os.path.basename(root))

    fig = plt.figure()

    for i in range(16):
        chosen_index = random.randint(0, len(img_path)-1)
        chosen_img = img_path[chosen_index]
        chosen_label = labels[chosen_index]

        ax = fig.add_subplot(4,4,i+1)
        ax.title.set_text(chosen_label)
        ax.imshow(Image.open(chosen_img))
    
    fig.tight_layout(pad=0.05)
    plt.show()
def get_images_sizes(path_to_data):

    img_path = []
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

def list_blobs(bucket_name):

    storage_client = storage.Client.from_service_account_json(path_to_credentials)
    blobs = storage_client.list_blobs(bucket_name)

    return blobs

def download_data_to_local_dir(bucket_name, local_dir):
    storage_client = storage.Client.from_service_account_json(path_to_credentials)
    blobs = storage_client.list_blobs(bucket_name)

    if not os.path.isdir(local_dir):
        os.makedirs(local_dir)

    for blob in blobs:
        joined_path = os.path.join(local_dir, blob.name)

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

    visualize_data_switch = False
    print_insights_switch = False
    list_blobs_switch = False
    download_data_switch = True

    data_path = '/home/sid/MEGAsync/Projects/object-detection/data'
    train_data_path = data_path + '/training'
    val_data_path = data_path + '/validation'
    test_data_path = data_path + '/test'

    if visualize_data_switch:
        visualize_data(train_data_path)

    if print_insights_switch:
        mean_width, mean_height, median_width, median_height = get_images_sizes(train_data_path)

        print(f"mean width = {mean_width}")
        print(f"mean height = {mean_height}")
        print(f"median width = {median_width}")
        print(f"median height = {median_height}")

    if list_blobs_switch:
        blobs = list_blobs('tools-data-bucket')

        for blob in blobs:
            print(blob.name)

    if download_data_switch:
        download_data_to_local_dir("tools-data-bucket", "./data")



