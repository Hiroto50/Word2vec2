import torch
import torch.nn as nn


class Word2Vec(nn.Module):
    def __init__(self, vocabulary_size, embedding_dim):
        super(Word2Vec, self).__init__()
        self.embeddings = nn.Embedding(vocabulary_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocabulary_size)

    def forward(self, target, context):
        target_embed = self.embeddings(target)
        context_embed = self.embeddings(context)
        scores = torch.matmul(target_embed, context_embed.t())
        return scores
