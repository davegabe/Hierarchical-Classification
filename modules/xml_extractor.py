import os
import pandas as pd


val_path = "ILSVRC2012_img_val"
csv = "val_mapping.csv"
data = pd.read_csv(csv)


for row in data.iterrows():
    # For each row in the csv file, create a folder with the object name and move the image to that folder
    filename = row[1][0]
    object = row[1][1]

    # Source and destination paths
    source = os.path.join(val_path, filename+".JPEG")
    path = os.path.join(val_path, object)
    destination = os.path.join(path, filename+".JPEG")

    # Create the folder if it does not exist
    if not os.path.exists(path):
        os.mkdir(path)

    # Move the image to the folder
    os.rename(source, destination)
