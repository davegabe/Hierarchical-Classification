import os 
from bs4 import BeautifulSoup
import pandas as pd


# data = []


# for filename in os.listdir("dataset/val_xmls"):
#     with open(f"dataset/val_xmls/{filename}", "r") as f:
#         soup = BeautifulSoup(f, "xml")
#         name = soup.find("filename").text
#         object = soup.find('name').text
#         data.append([name, object])
      
# # create a dataframe
# data = pd.DataFrame(data, columns=["filename", "object"])

# # save the dataframe to a csv file
# data.to_csv("dataset/val.csv", index=False)

data = pd.read_csv("dataset/val.csv")

try:
    os.mkdir("dataset/validation")
except:
    pass

for row in data.iterrows():
    filename = row[1][0]
    object = row[1][1]
    try:
        os.mkdir(f"dataset/validation/{object}")
    except:
        pass
    os.rename(f"dataset/val/{filename}.JPEG", f"dataset/validation/{object}/{filename}.JPEG")
    
   