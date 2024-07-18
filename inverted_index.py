import json
import tools
import ast


class SetEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return json.JSONEncoder.default(self, obj)


class InvertedIndexFactory:

    def __init__(self, output_filename):
        self.__output_filename = output_filename
        self.__inverted_index = {}

    def make_inverted_index_from_dataframe(self, df):
        """ Создает инвертированный список на основе столбца ['факт'] из DataFrame """

        cleaned_texts = df['факт']

        for index, cleaned_activity_name in cleaned_texts.items():
            print(index, cleaned_activity_name)
            for word in ast.literal_eval(cleaned_activity_name):
                print(word)
                self.__inverted_index.setdefault(word, set()).add(index)

        self.__save_inverted_index_to_json()

    def __save_inverted_index_to_json(self):
        dump = json.dumps(self.__inverted_index,
                          sort_keys=False,
                          indent=4,
                          ensure_ascii=False,
                          separators=(',', ': '),
                          cls=SetEncoder)

        tools.save_text_in_file(self.__output_filename, dump)
