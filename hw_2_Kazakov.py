import os
import re
from nltk.corpus import stopwords
import string
import pymorphy2
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from nltk.tokenize import word_tokenize
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from IPython.display import display
from tqdm import tqdm


def get_texts(main_dir):
    texts = []
    all_files = []
    for root, dirs, files in os.walk(main_dir):
        all_files.extend(files)
        for name in files:
            with open(os.path.join(root, name), 'r', encoding='utf-8-sig') as f:
                text = f.read().splitlines()
            text = ' '.join(text)
            text = re.sub(r' 9999 00:00:0,500 --> 00:00:2,00 www.tvsubtitles.net', '', text)
            text = re.sub(r'\d', '', text)
            text = re.sub(r'[A-Za-z]', '', text)
            texts.append(text)
    return texts, all_files


def preprocess(texts):
    stops = stopwords.words('russian')
    stops.extend(['это', 'весь'])
    morph = pymorphy2.MorphAnalyzer()

    preprocessed_texts = []
    for t in tqdm(texts):
        t = t.translate(str.maketrans('', '', string.punctuation)).lower()
        t = word_tokenize(t)
        t = [morph.parse(word)[0].normal_form for word in t if morph.parse(word)[0].normal_form not in stops]
        preprocessed_texts.append(' '.join(t))

    return preprocessed_texts


def inverted_index(preprocessed_texts, names):
    vectorizer = TfidfVectorizer(analyzer='word')
    X = vectorizer.fit_transform(preprocessed_texts)
    df = pd.DataFrame(X.todense(), index=names, columns=vectorizer.get_feature_names())
    print('matrix Document-Term:')
    display(df)

    return df, vectorizer


def query_transform(query, vectorizer):
    return vectorizer.transform(preprocess([query])).toarray()


def similarity(query_vector, df):
    return cosine_similarity(query_vector, df.values)


# у функции нет return, чтобы она могла повторять поиск
def main():
    main_dir = './friends-data'
    texts, names = get_texts(main_dir)
    preprocessed = preprocess(texts)
    df, vectorizer = inverted_index(preprocessed, names)
    query = ''
    while query != 'stop':
        if query != '':
            query_vector = query_transform(query, vectorizer)
            sim = similarity(query_vector, df)
            if sim.sum() > 0:
                ranked = np.argsort(-sim)
                print('Поисковая выдача:')
                for r in ranked[0]:
                    if sim[0][r] > 0:
                        print(df.iloc[r].name)
            else:
                print('К сожалению, ничего не найдено.')

        query = input('Если хотите закончить, введите "stop". '
                      'Введите поисковой запрос (не стоит вводить числа, латиницу, знаки препинания): ')
    print('Спасибо за использование моего поиска!')


if __name__ == "__main__":
    main()
