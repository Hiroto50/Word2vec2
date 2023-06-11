import torch

class Tokenize:
    def __init__(self, window_size):
        self.window_size = window_size
        self.wordtoindex = None

    def tokenize(self, dataset):
        wordtoindex = {}
        index = 0
        for sentence in dataset:
            for word in sentence.split():
                if word not in wordtoindex:
                    wordtoindex[word] = index
                    index += 1
        self.wordtoindex = wordtoindex

    def trainexam(self, dataset):
        train_examples = []
        for sentence in dataset:
            for index, target_word in enumerate(sentence.split()):
                context_words = []
                for j in range(
                    max(0, index - self.window_size),
                    min(index + self.window_size + 1, len(sentence.split()))
                ):
                    if j != index:
                        context_words.append(self.wordtoindex[sentence.split()[j]])
                if context_words:
                    train_examples.append((self.wordtoindex[target_word], context_words))
        target_tensors = torch.tensor([target for target, _ in train_examples])
        context_tensors = torch.tensor([context for _, context in train_examples])
        return list(zip(target_tensors, context_tensors))
