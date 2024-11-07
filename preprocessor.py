import string
from nltk.corpus import stopwords as sw
from nltk.corpus import wordnet as wn
from nltk import wordpunct_tokenize, sent_tokenize, WordNetLemmatizer, pos_tag

class NltkPreprocessor:
    def __init__(self, stopwords=None, punct=None, lower=True, strip=True):
        self.lower = lower
        self.strip = strip
        self.stopwords = stopwords or set(sw.words('english'))
        self.punct = punct or set(string.punctuation)
        self.lemmatizer = WordNetLemmatizer()
        self.tag_map = {'N': wn.NOUN, 'V': wn.VERB, 'R': wn.ADV, 'J': wn.ADJ}

    def tokenize(self, document):
        tokenized_doc = []

        for sent in sent_tokenize(document):
            for token, tag in pos_tag(wordpunct_tokenize(sent)):
                if self.lower:
                    token = token.lower()
                if self.strip:
                    token = token.strip('_0123456789')

                if token in self.stopwords or all(char in self.punct for char in token):
                    continue

                lemma = self.lemmatize(token, tag)
                tokenized_doc.append(lemma)

        return ' '.join(tokenized_doc)

    def lemmatize(self, token, tag):
        return self.lemmatizer.lemmatize(token, self.tag_map.get(tag[0], wn.NOUN))
