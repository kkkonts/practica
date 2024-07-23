import json
import os
import pandas as pd
import numpy as np
import ast

import tools
from tokenizer import Tokenizer
from inverted_index import InvertedIndexFactory
from tf_idf_calculator import TF_IDF_Calculator
from vector_model_search import VectorModelSearch
from output import output


def run_tokenizer(df):
    print("\n-> starting tokenizer...")

    tokenizer = Tokenizer()
    df['факт'] = df['факт'].apply(tokenizer.clean_text)
    df.to_excel(tools.TOKENIZED_PATH, index=False)

    print("tokenized")

    return df



def run_inverted_index():
    print("\n-> starting inverted index...")
    df = pd.read_excel(tools.TOKENIZED_PATH)
    factory = InvertedIndexFactory(tools.INVERTED_INDEX_PATH)
    factory.make_inverted_index_from_dataframe(df)
    print('inverted index was created!')


def run_tf_idf_calculator(df):
    print("\n-> starting TF-IDF calculator...")

    with open(tools.INVERTED_INDEX_PATH) as json_file:
        inverted_index = json.load(json_file)

    result = {}

    for term in inverted_index.keys():
        docs_with_term = inverted_index[term]
        for doc_index in docs_with_term:
            row = df['факт'].loc[doc_index]

            tokens = ast.literal_eval(row)

            TF, IDF, TF_IDF = TF_IDF_Calculator.calculate(term,
                                                          tokens,
                                                          df.shape[0],
                                                          len(docs_with_term))

            try:
                result[term][doc_index] = {"TF": TF, "IDF": IDF, "TF-IDF": TF_IDF}
            except KeyError:
                result[term] = {doc_index: {"TF": TF, "IDF": IDF, "TF-IDF": TF_IDF}}

    dump = json.dumps(result,
                      sort_keys=False,
                      indent=4,
                      ensure_ascii=False,
                      separators=(',', ': '))

    tools.save_text_in_file(tools.TF_IDF_PATH, dump)
    print('TF-IDF was created!')


if __name__ == '__main__':
    # run_tokenizer()
    # df = pd.read_excel(tools.TOKENIZED_PATH)

    # factory = InvertedIndexFactory(tools.INVERTED_INDEX_PATH)
    # factory.make_inverted_index_from_dataframe(df)

    # run_tf_idf_calculator(df)


    dff = pd.read_excel(tools.NATIONAL_PROJECTS_PATH)
    #
    # from sentence_transformers import SentenceTransformer
    # from sklearn.metrics.pairwise import cosine_similarity
    # import numpy as np
    # import pandas as pd
    #
    # model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    #
    #
    # def extract_keywords(text, n=5):
    #     # Разбиваем текст на слова
    #     words = text.split()
    #     # Создаем эмбеддинги для слов и всего текста
    #     word_embeddings = model.encode(words)
    #     text_embedding = model.encode([text])
    #     # Вычисляем косинусное сходство
    #     similarities = cosine_similarity(word_embeddings, text_embedding)
    #     # Выбираем топ-N слов
    #     top_n = np.argsort(similarities.flatten())[-n:]
    #     return [words[i] for i in top_n]
    #
    #
    # def find_similar_sentences(target_sentence, sentences, threshold=0.5):
    #     # Извлекаем ключевые слова из целевого предложения
    #     target_keywords = extract_keywords(target_sentence)
    #     # Создаем эмбеддинги для предложений
    #     sentence_embeddings = model.encode(sentences)
    #     target_embedding = model.encode([" ".join(target_keywords)])
    #     # Вычисляем косинусное сходство между целевым предложением и другими предложениями
    #     similarities = cosine_similarity(sentence_embeddings, target_embedding).flatten()
    #     # Находим предложения, сходство которых превышает пороговое значение
    #     similar_sentences = [sentences[i] for i in range(len(sentences)) if similarities[i] > threshold]
    #     return similar_sentences
    #
    #
    # # Пример использования
    # data = {
    #     'Наименование мероприятия': [
    #         'Конференция по искусственному интеллекту',
    #         'Семинар по машинному обучению',
    #         'Вебинар по обработке естественного языка',
    #         'Курс по глубокому обучению',
    #         'Лекция по нейронным сетям'
    #     ]
    # }

    # dff = pd.DataFrame(data)
    #
    # for index, row in df.iterrows():
    #     event_name = row['Наименование мероприятия']
    #     similar_sentences = find_similar_sentences(event_name, df['Наименование мероприятия'].tolist())
    #     print(f"Строка {index}: {event_name}")
    #     print(f"Похожие предложения: {similar_sentences}")

    # Далее можно использовать полученные эмбеддинги для выделения ключевых слов,
    # например, путем кластеризации или сравнения с эмбеддингом всего текста
    # dff = pd.read_excel(tools.NATIONAL_PROJECTS_PATH)
    # grouped_data = dff.groupby(by=['Краткое наименование национального проекта'])

    # df_res = pd.DataFrame(
    #     columns=['№ п/п', 'id категории', 'категория', 'id подкатегории', 'Подкатегория', 'id факта', 'Факт'])

    # grouped_data = dff.groupby(by=['Краткое наименование национального проекта'])

    accuracy = 0.5
    vms = VectorModelSearch(accuracy)
    output(vms, 'vms_vanilla')

    # df_res = pd.DataFrame(
    #     columns=['№ п/п', 'id категории', 'Категория', 'НП', 'id подкатегории', 'Подкатегория', 'id факта', 'Факт'])
    #
    # for name, group in grouped_data:
    #     df_res = pd.DataFrame(
    #         columns=['№ п/п', 'id категории', 'Категория', 'НП', 'id подкатегории', 'Подкатегория', 'id факта', 'Факт'])
    #     i = 1
    #     print(name)
    #     OUTPUT = f"output/res_{name[0]+str(accuracy)}.xlsx"
    #     for idx, row in group.iterrows():
    #         search_result = vms.search(row['Наименование мероприятия'])
    #
    #         row = pd.DataFrame({
    #             '№ п/п': [i],
    #             'id категории': [np.nan],
    #             'Категория': [np.nan],
    #             'НП': [row['Наименование мероприятия']],
    #             'id подкатегории': [np.nan],
    #             'Подкатегория': [np.nan],
    #             'id факта': [np.nan],
    #             'Факт': [np.nan]
    #         })
    #
    #         search_result = pd.DataFrame({
    #             '№ п/п': np.nan,
    #             'id категории': search_result['id категории'],
    #             'Категория': search_result['категория'],
    #             'НП': np.nan,
    #             'id подкатегории': search_result['id подкатегории'],
    #             'Подкатегория': search_result['подкатегория'],
    #             'id факта': search_result['id факта'],
    #             'Факт': search_result['факт']
    #         })
    #         print(search_result)
    #         df_res = pd.concat([df_res, row, search_result], ignore_index=True, axis=0)
    #
    #         i += 1
    #
    #     df_res.to_excel(OUTPUT, index=False)
    output(vms, "tf_if_0.5")




