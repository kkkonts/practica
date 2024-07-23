import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from tokenizer import Tokenizer
import tools
import ast
from output import output

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Transformer:
    def __init__(self, threshold):
        self.__model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.__tokenizer = Tokenizer()
        self.__indices = pd.read_excel(tools.FACT_PATH)
        self.__docs = pd.read_excel(tools.TOKENIZED_PATH)
        self.__sentences = self.__docs['факт'].apply(ast.literal_eval).apply(' '.join)
        self.__sentence_embeddings = self.__model.encode(self.__sentences)
        self.__threshold = threshold

    def extract_keywords(self, query, threshold=0.5):
        words = self.__tokenizer.clean_text(query)

        word_embeddings = self.__model.encode(words)
        query_embedding = self.__model.encode([query])
        similarities = cosine_similarity(word_embeddings, query_embedding).flatten()

        keywords = [words[i] for i in range(len(similarities)) if similarities[i] >= threshold]

        return keywords

    def search(self, target_sentence):
        print(target_sentence)
        target_keywords = self.extract_keywords(target_sentence)
        print(target_keywords)
        if not target_keywords:
            return []
        sentences = self.__docs
        target_embedding = self.__model.encode([' '.join(target_keywords)])
        similarities = cosine_similarity(self.__sentence_embeddings, target_embedding).flatten()
        doc_ids = [i for i in range(len(sentences)) if similarities[i] > self.__threshold]
        print(self.__indices.loc[doc_ids])
        return self.__indices.loc[doc_ids]


trf = Transformer(0.6)
output(trf, "trf_0.6")
