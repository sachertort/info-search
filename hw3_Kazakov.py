import pymorphy2
import string
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from scipy.sparse import csr_matrix
import json
from tqdm import tqdm


def data(file):
    with open(file, 'r') as f:
        corpus = list(f)[:55000]  # я взял 55000 вместо 50000, потому что там есть пустые, позже я отрезаю до 50000

    answers = []
    for c in tqdm(corpus):
        one_quest_ans = json.loads(c)['answers']
        try:
            rates = [(i, int(a['author_rating']['value'])) for i, a in enumerate(one_quest_ans)]
        except ValueError:
            continue
        if not rates:
            continue
        answer = one_quest_ans[sorted(rates, key=lambda k: k[1], reverse=True)[0][0]]['text']
        answers.append(answer)
    return answers[:50000]


def preprocess(texts):
    stops = stopwords.words('russian')
    stops.extend(['это', 'весь'])
    morph = pymorphy2.MorphAnalyzer()

    preprocessed_texts = []
    for t in texts:
        t = t.translate(str.maketrans('', '', string.punctuation)).lower()
        t = word_tokenize(t)
        t = [morph.parse(word)[0].normal_form for word in t if morph.parse(word)[0].normal_form not in stops]
        preprocessed_texts.append(' '.join(t))
    return preprocessed_texts


def document_term(data):
    cv = CountVectorizer()
    index = cv.fit_transform(data)

    Tf_vec = TfidfVectorizer(use_idf=False, norm='l2')
    TF = Tf_vec.fit_transform(data)

    TfIdf_vec = TfidfVectorizer(use_idf=True, norm='l2')
    TfIdf_vec.fit(data)
    IDF = np.expand_dims(TfIdf_vec.idf_, axis=0)

    ld = index.sum(axis=1)

    avgdl = ld.mean()

    k = 2
    b = 0.75

    num = TF.multiply(csr_matrix(IDF)).dot(k + 1).toarray()
    coef = (k * (1 - b + b * ld / avgdl))
    rows = []
    columns = []
    BM25 = []
    for i, j in zip(*num.nonzero()):
        den = TF[i, j] + coef[i]
        BM25.append(float(num[i, j]/den))
        rows.append(i)
        columns.append(j)
    DT = csr_matrix((BM25, (rows, columns)))
    return DT, cv


def query(q, cv):
    return cv.transform(preprocess([q]))


def similarity(DT, q_vec):
    return np.dot(DT, q_vec.T).toarray()


def main():
    file = 'questions_about_love.jsonl'
    dataset = data(file)
    preprocessed = preprocess(dataset)
    DT, cv = document_term(preprocessed)
    q = ''
    print('Это поиск на основе BM25 для Ответов Mail.ru.')
    while q != 'stop':
        if q != '':
            q = query(q, cv)
            ids_docs = similarity(DT, q)
            ids_docs_sorted = np.argsort(np.squeeze(ids_docs), axis=0)
            ids_docs_sorted = ids_docs_sorted.tolist()[::-1]
            if sum(ids_docs) > 0:
                print('\nВот какие ответы (топ-10) нашлись по вашемему запросу:')
                for i in ids_docs_sorted[:10]:
                    if ids_docs[i] > 0:
                        print(dataset[i])
            else:
                print('К сожалению, ничего не найдено.')
        q = input('\nВведите запрос или введите "stop", если больше не хотите пользоваться поиском: ')
    print('Спасибо за использование этого поиска!')


if __name__ == "__main__":
    main()
