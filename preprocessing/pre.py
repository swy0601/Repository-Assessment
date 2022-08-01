import os
import pickle

import numpy as np

DATASET_DIR="D:\Project\Graph\dataset\Readability Data"


list_java_file = {
    "Neutral": [i for i in os.listdir(os.path.join(DATASET_DIR,"Java Code\\Neutral")) if ".java" in i],
    "Readable": [i for i in os.listdir(os.path.join(DATASET_DIR,"Java Code\\Readable")) if ".java" in i],
    "Unreadable": [i for i in os.listdir(os.path.join(DATASET_DIR,"Java Code\\Unreadable")) if ".java" in i]}

list_java_data = {
    "Neutral": [],
    "Readable": [],
    "Unreadable": []
}





for i, file in enumerate(list_java_file["Neutral"]):
    with open(os.path.join(DATASET_DIR,"Java Code\\Neutral",file) , "r") as f:
        list_java_data['Neutral'] .append( f.read())
for file in list_java_file["Readable"]:
    with open(os.path.join(DATASET_DIR,"Java Code\\Readable",file), "r") as f:
        list_java_data['Readable'].append( f.read())
for file in list_java_file["Unreadable"]:
    with open(os.path.join(DATASET_DIR,"Java Code\\Unreadable",file), "r") as f:
        list_java_data['Unreadable'].append( f.read())


folder1={}
folder2={}
folder3={}

folder1["Neutral"]=list_java_data["Neutral"][0:int(list_java_data["Neutral"].__len__()/3)]
folder2["Neutral"]=list_java_data["Neutral"][int(list_java_data["Neutral"].__len__()/3):int(2*list_java_data["Neutral"].__len__()/3)]
folder3["Neutral"]=list_java_data["Neutral"][int(2*list_java_data["Neutral"].__len__()/3):]


folder1["Readable"]=list_java_data["Readable"][0:int(list_java_data["Readable"].__len__()/3)]
folder2["Readable"]=list_java_data["Readable"][int(list_java_data["Readable"].__len__()/3):int(2*list_java_data["Readable"].__len__()/3)]
folder3["Readable"]=list_java_data["Readable"][int(2*list_java_data["Readable"].__len__()/3):]


folder1["Unreadable"]=list_java_data["Unreadable"][0:int(list_java_data["Unreadable"].__len__()/3)]
folder2["Unreadable"]=list_java_data["Unreadable"][int(list_java_data["Unreadable"].__len__()/3):int(2*list_java_data["Unreadable"].__len__()/3)]
folder3["Unreadable"]=list_java_data["Unreadable"][int(2*list_java_data["Unreadable"].__len__()/3):]


with open("java_data.pkl", "wb") as f:
    pickle.dump((folder1,folder2,folder3), f)
