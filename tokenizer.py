import nltk
import re

from nltk.corpus import stopwords
from pymystem3 import Mystem
import nltk
import numpy as np

# import ssl
#
# try:
#     _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     pass
# else:
#     ssl._create_default_https_context = _create_unverified_https_context
#
# nltk.download('punkt')
# nltk.download('stopwords')

#TODO удалить и не позориться
#малышка для чека прогресса токенизации
i = 0


class Tokenizer:

    def clean_text(self, text):
        global i
        print(i, text)
        i += 1
        if isinstance(text, (float, np.floating)) and np.isnan(text):
            text = ""
        tokens = self.__tokenize(text)
        tokens = self.__lemmatize(tokens)
        tokens = self.__remove_stop_words(tokens)

        return tokens

    @staticmethod
    def __tokenize(text):
        """ Делит текст на токены """
        tokens = nltk.word_tokenize(text)

        return tokens

    @staticmethod
    def __lemmatize(tokens):
        """ С помощью Mystem лемматизирует токены """
        mystem = Mystem()

        tokens = [token.replace(token, ''.join(mystem.lemmatize(token))) for token in tokens]

        return tokens

    @staticmethod
    def __remove_stop_words(tokens):
        """ Удаляет лишние символы """
        tokens = [re.sub(r"\W", "", token, flags=re.I) for token in tokens]

        stop_words = stopwords.words('russian')
        only_cyrillic_letters = re.compile('[а-яА-Я]')

        tokens = [token.lower() for token in tokens if (token not in stop_words)
                  and only_cyrillic_letters.match(token)
                  and not token.isdigit()
                  and token != '']

        return tokens
