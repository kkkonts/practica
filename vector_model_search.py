import json
import operator
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tokenizer import Tokenizer
from tf_idf_calculator import TF_IDF_Calculator
import tools


def dist_cosine(vec_a, vec_b):
    cosine = cosine_similarity([vec_a], [vec_b])
    return cosine[0][0]


class VectorModelSearch:

    def __init__(self, accuracy, index_path=None, tf_idf_path=None):
        self.__tokenizer = Tokenizer()
        self.__all_docs_count = 0
        self.__tf_idf_calculations = {}
        with open(tools.TF_IDF_PATH) as json_file:
            self.__tf_idf_calculations = json.load(json_file)

        self.__indices = pd.read_excel(tools.FACT_PATH)

        self.__all_docs_count = self.__indices.shape[0]
        self.accuracy = accuracy

    def search(self, query):
        print("SEARCHING: {}".format(query))
        tokens = self.__tokenizer.clean_text(query)

        if len(tokens) == 0:
            print("Empty query")
            return

        query_vector = []

        for token in tokens:
            if token in self.__tf_idf_calculations:
                doc_with_terms_count = len(self.__tf_idf_calculations[token])
            else:
                doc_with_terms_count = 0

            _, _, tf_idf = TF_IDF_Calculator.calculate(token,
                                                       tokens,
                                                       self.__all_docs_count,
                                                       doc_with_terms_count)
            query_vector.append(tf_idf)

        distances = {}

        for index in range(self.__all_docs_count):
            document_vector = []

            for token in tokens:
                try:
                    tf_idf = self.__tf_idf_calculations[token][str(index)]["TF-IDF"]
                    document_vector.append(tf_idf)
                except KeyError:
                    document_vector.append(0.0)

            distances[index] = dist_cosine(query_vector, document_vector)

        searched_indices = sorted(distances.items(), key=operator.itemgetter(1), reverse=True)

        doc_ids = [doc_id for doc_id, tf_idf in searched_indices if tf_idf >= self.accuracy]

        results_df = self.__indices.loc[doc_ids]

        return results_df
