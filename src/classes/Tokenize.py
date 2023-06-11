class Tokenize():
    def __init__(self, window_size):
        self.window_size = window_size

    def tokenize(self, dataset):
        wordtoindex = dict()
        index = 0
        for sentence in dataset:
            for word in sentence.split():
                if word not in wordtoindex:
                    wordtoindex[word] = index
                    index += 1
        self.wordtoindex = wordtoindex

    def trainexam(self, dataset):
        train_examples = list()
        for sentence in dataset:
            for index, target_word in enumerate(sentence.split()):
                contextwords = list()
                for j in range(
                    max(0, index - self.window_size),
                    min(index + self.window_size + 1, len(sentence.split()))
                ):
                    if j != index:
                        contextwords.append(self.wordtoindex[sentence.split()[j]])
                train_examples.append((self.wordtoindex[target_word], contextwords))
        return train_examples        
        