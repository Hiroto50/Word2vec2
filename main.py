
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from src.utils.database import get_databases
from src.classes.PreProcess import PreProcess
from src.classes.Tokenize import Tokenize
from src.classes.Word2Vec import Word2Vec

import os


def main():

    
    dataset_huge, dataset_tiny = get_databases()

    pre_process = PreProcess()
    tokenize = Tokenize(2)


    df1_preprocess = [pre_process.preproses(text) for text in dataset_tiny]
    tokenize.tokenize(df1_preprocess)
    train_examples = tokenize.trainexam(df1_preprocess)

    model = Word2Vec(len(tokenize.wordtoindex), 100)
    learning_rate = 0.001
    num_epochs = 10
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    target_tensors = torch.tensor([target for target, _ in train_examples])
    context_tensors = torch.tensor([context for _, context in train_examples])

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        scores = model(target_tensors, context_tensors)
        loss = criterion(scores.flatten(), context_tensors.flatten())
        loss.backwards()
        optimizer.step()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")
    

    word_embeddings = model.embeddings.weight.data

    num_clusters = 10

    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(word_embeddings)


    word_labels = kmeans.labels_

    for word, label in zip(tokenize.wordtoindex.keys(), word_labels):
        print(f"Word: {word}, Cluster Label: {label}")


if __name__ == '__main__':
    main()
        

