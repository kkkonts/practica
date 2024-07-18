from datetime import datetime
import pandas as pd

FACT_PATH = "input/Классификатор_обращений.xlsx"
NATIONAL_PROJECTS_PATH = "input/Перечень_мероприятий.xlsx"

TOKENIZED_PATH = "output/Классификатор_обращений_tokenized.xlsx"
INVERTED_INDEX_PATH = "output/inverted_index.json"
TF_IDF_PATH = "output/td-idf-calculation.json"



def save_text_in_file(text_file_path, text):
    text_file = open(text_file_path, "w")
    text_file.write(text)
    text_file.close()


def serialize_datetime(obj):
    if isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()
    raise TypeError(f'Type {type(obj)} is not JSON serializable')
