import string

from src.constants.prepros import NORMALIZATION_RULES, STOP_WORDS

class PreProcess():

    def lowercase(self, text):
        return text.lower()

    def punctuation(self, text):
        text_transform = text.translate(str.maketrans('', '', string.punctuation))
        text = text_transform
        return text
    
    def stopwordsremove(self, text):
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in STOP_WORDS]
        filtered_text = " ".join(filtered_words)
        return filtered_text
    
    def numbers(self, text):
        words = text.split()
        filtered_words = [word for word in words if not word.isdigit()]
        filtered_text = " ".join(filtered_words)
        return filtered_text
    
    def lemmatize(self, text):
        words = text.split()
        lemmatized_words = [self.lemmatized_words(word) for word in words]
        lemmatized_text = " ".join(lemmatized_words)
        return lemmatized_text
    
    def lemmatizes(self, word):
        for suffix, replacement in NORMALIZATION_RULES.items():
            if word.endswith(suffix):
                word = word[:-len(suffix)] + replacement
                break
        return word
    
    def preproses(self, text):
        text = self.lowercase(text)
        text = self.punctuation(text)
        text = self.numbers(text)
        text = self.stopwordsremove(text)
        text = self.lemmatizes(text)
        return text
