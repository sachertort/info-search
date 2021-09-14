import os
import re
import string
from nltk.corpus import stopwords
import pymorphy2
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
from IPython.display import display


def get_texts(main_dir):
    texts = []

    for root, dirs, files in os.walk(main_dir):
        for name in files:
            with open(os.path.join(root, name), 'r', encoding='utf-8-sig') as f:
                text = f.read().splitlines()
            text = ' '.join(text)
            text = re.sub(r' 9999 00:00:0,500 --> 00:00:2,00 www.tvsubtitles.net', '', text)
            text = re.sub(r'\d', '', text)
            text = re.sub(r'[A-Za-z]', '', text)
            texts.append(text)
    return texts


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


def inverted_index(preprocessed_texts):
    vectorizer = CountVectorizer(analyzer='word')
    X = vectorizer.fit_transform(preprocessed_texts)
    df = pd.DataFrame(X.todense(), columns=vectorizer.get_feature_names())
    print('matrix Term-Document:')
    display(df)

    return df


def tasks(df):
    morph = pymorphy2.MorphAnalyzer()

    print('#1: the most frequent word in the collection:', df.sum(axis=0).idxmax(axis="columns"),
          '(frequency: %s)\n' % df.sum(axis=0).max())

    print('#2: one of the rarest word in the collection:', df.sum(axis=0).idxmin(axis="columns"),
          '(frequency: %s)\n' % df.sum(axis=0).min())

    new_df = df.replace(0, np.nan)
    print('#3: words that are in all documents:',
          ', '.join(list(new_df.dropna(axis='columns').columns)) + '\n')

    characters = [['Моника', 'Мон'], ['Рэйчел', 'Рейч'], ['Чендлер', 'Чэндлер', 'Чен'],
                  ['Фиби', 'Фибс'], ['Росс'], ['Джоуи', 'Джои', 'Джо']]
    char_freq = {}
    for ch in characters:
        normal = [morph.parse(name)[0].normal_form for name in ch]
        char_freq[ch[0]] = df[normal].sum(axis=0).sum()

    print('#4: how many times was each character used in subtitles '
          '(all variants of the name are taken into account):',
          char_freq, '\nthe most frequent character:', max(char_freq))


def main():
    main_dir = './friends-data'
    texts = get_texts(main_dir)
    preprocessed = preprocess(texts)
    main_df = inverted_index(preprocessed)
    tasks(main_df)


if __name__ == "__main__":
    main()
