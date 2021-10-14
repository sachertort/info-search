import json
from tqdm import tqdm
import pymorphy2
import string
import nltk
from nltk import word_tokenize
from gensim.models.fasttext import FastTextKeyedVectors
from gensim.models import KeyedVectors
import numpy as np
from transformers import AutoTokenizer, AutoModel
from nltk.corpus import stopwords
import torch
import os
from sklearn.preprocessing import normalize


def data(file):
    with open(file, 'r') as f:
        corpus = list(f)[:15000]

    answers = []
    for c in tqdm(corpus):
        one_quest_ans = json.loads(c)['answers']
        try:
            rates = [(i, int(a['author_rating']['value'])) for i, a in enumerate(one_quest_ans)]
        except ValueError:
            continue
        if rates == []:
            continue
        answer = one_quest_ans[sorted(rates, key=lambda k: k[1], reverse=True)[0][0]]['text']
        answers.append(answer)

    return answers[:10000]


def preprocess(texts, stops, morph):
    bad = []
    preprocessed_texts = []
    for t in tqdm(texts):
        t = t.translate(str.maketrans('', '', string.punctuation)).lower()
        t = word_tokenize(t)
        t = [morph.parse(word)[0].normal_form for word in t if morph.parse(word)[0].normal_form not in stops]
        preprocessed_texts.append(t)

    return preprocessed_texts


def normalized(x):
    return x / np.linalg.norm(x)


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


def query(q, emb, model_ft, model_bert, stops, morph, tokenizer):
    if emb == 'ft':
        vec = ft_vectorization(preprocess([q], stops, morph),
                               model_ft, corpus=False)[0]
    else:
        vec = bert_vectorization([q], tokenizer,
                                 model_bert, corpus=False)[0]

    return vec


def best(vec_q, doc_term):
    vec_q = np.expand_dims(vec_q, axis=0)
    return np.squeeze(np.dot(doc_term, vec_q.T).argsort(axis=0)[::-1][:5])


def main():
    file = 'questions_about_love.jsonl'
    answers = data(file)
    nltk.download('stopwords')
    nltk.download('punkt')
    stops = stopwords.words('russian')
    stops.extend(['это', 'весь'])
    morph = pymorphy2.MorphAnalyzer()

    path = 'araneum_none_fasttextcbow_300_5_2018.model'
    model_ft = FastTextKeyedVectors.load(path)
    if 'ft.csv' in os.listdir():
        vecs_ft = np.genfromtxt('ft.csv', delimiter=',')
    else:
        preprocessed_texts = preprocess(answers, stops, morph)
        vecs_ft = ft_vectorization(preprocessed_texts, model_ft)

    tokenizer = AutoTokenizer.from_pretrained('cointegrated/rubert-tiny')
    model_bert = AutoModel.from_pretrained('cointegrated/rubert-tiny').to('cuda')
    if 'bert.csv' in os.listdir():
        vecs_bert = np.genfromtxt('bert.csv', delimiter=',')
    else:
        vecs_bert = bert_vectorization(answers, tokenizer, model_bert)

    print('Это поиск на основе FastText / BERT для Ответов Mail.ru.')
    emb = ''
    q = ''
    md = ['ft', 'bert']
    while emb not in md:
        emb = input('Введите предпочитаемую языковую модель (`bert` или `ft`): ')
    while q != 'stop':
        if q != '':
            vec = query(q, emb, model_ft, model_bert, stops, morph, tokenizer)
            if emb == 'ft':
                best_res_ind = best(vec, vecs_ft)
            else:
                best_res_ind = best(vec, vecs_bert)

            print('\nВот какие ответы (топ-5) нашлись по вашемему запросу:')
            for i in best_res_ind:
                print(answers[i])
        q = input('\nВведите запрос или введите `stop`, если больше не хотите пользоваться поиском: ')
    print('Спасибо за использование этого поиска!')

if __name__ == "__main__":
    main()