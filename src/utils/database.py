import pandas as pd
from src.utils.path import read_file

def get_databases():
    dataset_huge = read_file("dataset_huge.raw")
    dataset_tiny = read_file("dataset_tiny.raw")
    return dataset_huge, dataset_tiny
    labels = list()
    messages = list()
    for line in file_content:
        label, message = line.split("\t")
        labels.append(label)
        messages.append(message)
    dataframe = pd.DataFrame({"x": messages, "y": labels})
    return dataframe