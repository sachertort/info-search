import json
from tqdm import tqdm
import pymorphy2
import string
from nltk import word_tokenize
from nltk.corpus import stopwords
from gensim.models.fasttext import FastTextKeyedVectors
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix


def data(file):
    with open(file, 'r') as f:
        corpus = list(f)[:15000]

    questions = []
    answers = []
    for c in tqdm(corpus):
        one_quest_ans = json.loads(c)['answers']
        try:
            rates = [(i, int(a['author_rating']['value'])) for i, a in enumerate(one_quest_ans)]
        except ValueError:
            continue
        if rates == []:
            continue
        answers.append(one_quest_ans[sorted(rates, key=lambda k: k[1], reverse=True)[0][0]]['text'])
        questions.append(json.loads(c)['question'])

    return questions[:10000], answers[:10000]


def preprocess(texts):
    stops = stopwords.words('russian')
    stops.extend(['это', 'весь'])
    morph = pymorphy2.MorphAnalyzer()

    bad = []
    preprocessed_texts = []
    for t in tqdm(texts):
        t = t.translate(str.maketrans('', '', string.punctuation)).lower()
        t = word_tokenize(t)
        t = [morph.parse(word)[0].normal_form for word in t if morph.parse(word)[0].normal_form not in stops]
        preprocessed_texts.append(t)

    return preprocessed_texts


def metrics(q_vects, a_vects):
    matr = np.argsort(q_vects.dot(a_vects.T), axis=1)[:, -5:]

    count = 0
    for i, row in enumerate(matr):
        if i in row:
            count += 1
    metric = count / matr.shape[0]
    return metric


def classic_vects(vect, q, a):
    vect.fit(q)
    q_vects = normalize(vect.transform(q))
    a_vects = normalize(vect.transform(a))
    return q_vects.toarray(), a_vects.toarray()


def bm25(preproc_d):
    cv = CountVectorizer()
    index = cv.fit_transform(preproc_d)
    index = normalize(index)

    Tf_vec = TfidfVectorizer(use_idf=False, norm='l2')
    TF = Tf_vec.fit_transform(preproc_d)

    TfIdf_vec = TfidfVectorizer(use_idf=True, norm='l2')
    TfIdf_vec.fit(preproc_d)
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
    DT = normalize(csr_matrix((BM25, (rows, columns))))
    return DT.toarray(), cv


def bm25_answers(a, cv):
    return normalize(cv.transform(a).toarray())


def ft_vectorization(preprocessed_texts, model, corpus=True):
    vecs_ft = []
    for text in tqdm(preprocessed_texts):
        if text == []:
            vec = np.zeros(300)
        else:
            vec = model[text]
            vec = np.mean(vec, axis=0)
        vecs_ft.append(vec)

    vecs_ft = normalize(np.array(vecs_ft))
    if corpus:
        np.savetxt('ft.csv', vecs_ft, delimiter=",")
    return vecs_ft


def bert_vectorization(data, tokenizer, model, corpus=True):
    vecs_bert = []
    for a in tqdm(data):
        encoded_input = tokenizer(a, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = model(**{k: v.to('cuda') for k, v in encoded_input.items()})
        embeddings = model_output.last_hidden_state[:, 0, :]
        embeddings = torch.nn.functional.normalize(embeddings)
        vecs_bert.append(embeddings[0].cpu().numpy())

    vecs_bert = np.array(vecs_bert)
    if corpus:
        np.savetxt('bert.csv', vecs_bert, delimiter=',')

    return vecs_bert


def main():
    file = 'questions_about_love.jsonl'
    quests, ans = data(file)

    quests_prep = preprocess(quests)
    ans_prep = preprocess(ans)
    quests_l = [' '.join(text) for text in quests_prep]
    ans_l = [' '.join(text) for text in ans_prep]

    count_vect = CountVectorizer()
    q_vects_c, a_vects_c = classic_vects(count_vect, quests_l, ans_l)

    tfidf_vect = TfidfVectorizer()
    q_vects_t, a_vects_t = classic_vects(tfidf_vect, quests_l, ans_l)

    bm25_q, cv = bm25(quests_l)
    bm25_a = bm25_answers(ans_l, cv)

    path = 'araneum_none_fasttextcbow_300_5_2018.model'
    model_ft = FastTextKeyedVectors.load(path)
    if 'ft.csv' in os.listdir():
        vecs_ft_a = np.genfromtxt('ft.csv', delimiter=',')
    else:
        vecs_ft_a = ft_vectorization(ans_prep, model_ft)
    vecs_ft_q = ft_vectorization(quests_prep, model_ft, corpus=False)

    tokenizer = AutoTokenizer.from_pretrained('cointegrated/rubert-tiny')
    model_bert = AutoModel.from_pretrained('cointegrated/rubert-tiny').to('cuda')
    if 'bert.csv' in os.listdir():
        vecs_bert_a = np.genfromtxt('bert.csv', delimiter=',')
    else:
        vecs_bert_a = bert_vectorization(ans, tokenizer, model_bert)
    vecs_bert_q = bert_vectorization(quests, tokenizer, model_bert, corpus=False)

    print('Метрики:')
    print('CountVectorizer:', metrics(q_vects_c, a_vects_c))
    print('TfidfVectorizer:', metrics(q_vects_t, a_vects_t))
    print('BM25:', metrics(bm25_q, bm25_a))
    print('FastText:', metrics(vecs_ft_q, vecs_ft_a))
    print('ruBert-tiny:', metrics(vecs_bert_q, vecs_bert_a))


if __name__ == "__main__":
    main()