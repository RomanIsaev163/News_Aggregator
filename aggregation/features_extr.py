import numpy as np
from itertools import groupby

import torch
from tqdm import tqdm
from sklearn.metrics import pairwise_distances
from transformers import T5ForConditionalGeneration, T5Tokenizer


def get_top_articles(data, centroids_map, top_k=5, metric='cosine', text_col='content'):
    """
        Функция выделяет ближайших к центру top_k новостей для каждого кластера.
        Вход:
            data - данные о новостях с текстами и эмбеддингами,
            metric = cosine/euclidean

    """
    top_texts_list = []

    for label, cluster_center in centroids_map.items():
        cluster = data[data['label'] == label]
        embeddings = cluster['embedding'].to_list()
        texts = cluster[text_col].values.tolist()

        distances = [pairwise_distances(cluster_center.reshape(1, -1), e.reshape(1, -1), metric=metric)[0][0] for e in
                     embeddings]
        scores = list(zip(texts, distances))
        top_ = sorted(scores, key=lambda x: x[1])[:top_k]
        top_texts = list(zip(*top_))[0]
        top_texts_list.append(top_texts)
    return top_texts_list


class KeyWordsExtractor:

    def __init__(self, model_name="0x7194633/keyt5-large"):
        self.model_name = model_name

        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        if torch.cuda.is_available():
            self.model.cuda()

        self.device = self.model.device

    def generate(self, text, **kwargs):
        """
        Производим генерацию keywords
        """

        inputs = self.tokenizer(text, return_tensors='pt').to(self.device)

        with torch.no_grad():
            hypotheses = self.model.generate(**inputs, num_beams=5, **kwargs)

        s = self.tokenizer.decode(hypotheses[0], skip_special_tokens=True)
        s = s.replace('; ', ';').replace(' ;', ';').lower().split(';')[:-1]
        gamma = 1
        s = [el for el, _ in groupby(s)]
        weights = [gamma ** i for i in range(len(s))]

        return s, weights

    def get_keywords(self, set_of_articles, **kwargs):
        """
        Получаем отсортированные по частоте сгенерированные ключевые фразы из набора статей

        [(key_1, weight_1), (key_2, weight_2), ....]
        """

        keys_weights = []
        len_set = len(set_of_articles)

        for i in range(len_set):
            text = set_of_articles[i]
            keys_weights.append(self.generate(text, **kwargs))

        return sort_and_remove_repeat(keys_weights)

    def get_trends(self, set_of_articles, n=5, threshold=0.95, **kwargs):
        """
        Получаем n трендовых ключевых слов
        """

        keys = self.get_keywords(set_of_articles, **kwargs)

        keys, _ = self.cos_simularity(keys, threshold=threshold)

        return keys[:n]

    def get_embed(self, text):
        t = self.tokenizer(text.replace('\n', ''), padding=True, truncation=True, return_tensors='pt').to(self.device)
        with torch.no_grad():
            model_output = self.model.encoder(input_ids=t.input_ids, attention_mask=t.attention_mask, return_dict=True)
        embeddings = model_output.last_hidden_state[:, 0, :]
        embeddings = torch.nn.functional.normalize(embeddings)
        return embeddings[0].cpu()

    def cos_simularity(self, keys_weights, threshold=0.95):
        """
        Считаем косинусную схожесть, возвращаем отсортированный по частоте список
        объединённый ключей таблицу взаимной схожести
        """

        new_keys_weights = []

        cos_sim = np.ones((len(keys_weights), len(keys_weights)))

        cos = torch.nn.CosineSimilarity(dim=1)

        for i in range(len(keys_weights)):
            sim_keys = [keys_weights[i][0]]
            sim_weight = keys_weights[i][1]
            embed_i = torch.unsqueeze(self.get_embed(keys_weights[i][0]), 0)
            for j in range(i + 1, len(keys_weights)):
                embed_2 = torch.unsqueeze(self.get_embed(keys_weights[j][0]), 0)

                cos_sim[i][j] = cos(embed_i, embed_2).numpy()[0]
                cos_sim[j][i] = cos_sim[i][j]

                if cos_sim[i][j] > threshold:
                    sim_keys.append((keys_weights[j][0]))
                    sim_weight += keys_weights[j][1]

            new_keys_weights.append((sim_keys, [sim_weight]))
        return sorted(new_keys_weights, key=lambda tup: tup[1], reverse=True), cos_sim


def sort_and_remove_repeat(keys_weights):
    """
    Убираем повторения
    Сортируем по второму аргументу массив вида [([x1], [y1]), ([x2], [y2]), ...],
    где
    * y2 - это частота
    * x1 - это ключевое слово
    """
    dict_keys = {}
    set_stop_words = {'анализ и проектирование систем', 'промышленное программирование', 'читальный зал',
                      'разработка веб-сайтов', 'программирование микроконтроллеров',
                      'системное программирование', 'ненормальное программирование',
                      'мобильная разработка', 'разработка мобильных приложений', 'будущее здесь',
                      'научно-популярное', 'платежи в интернет', 'платежи с мобильного', 'разработка игр',
                      'монетизация игр', 'дизайн игр'}

    for i in range(len(keys_weights)):
        for j in range(len(keys_weights[i][0])):
            if keys_weights[i][0][j] in set_stop_words:
                continue
            elif not (keys_weights[i][0][j] in dict_keys):
                dict_keys[keys_weights[i][0][j]] = keys_weights[i][1][j]
            else:
                dict_keys[keys_weights[i][0][j]] += keys_weights[i][1][j]

    dict_zip = list(zip(dict_keys.keys(), dict_keys.values()))

    return sorted(dict_zip, key=lambda tup: tup[1], reverse=True)

#Trends
def get_trends(data, centroids_map, top_for_cluster=10, max_news_len=200, text_col='title_text'):
    words_extractor = KeyWordsExtractor()

    top_articles_for_clusters = get_top_articles(data, centroids_map, top_k=top_for_cluster, text_col=text_col)

    trends_list = []
    pbar = tqdm(total=len(top_articles_for_clusters))
    pbar.set_description("Getting trends...")
    for top_articles, clust_label in top_articles_for_clusters:
        top_articles = [' '.join(row.split(' ')[:max_news_len]) for row in top_articles]

        current_trends = words_extractor.get_trends(top_articles, threshold=0.95, top_p=1.0, max_length=256,
                                                    min_length=5)
        current_trends = [x[0][0] for x in current_trends]

        trends_list.append((current_trends, top_articles))
        pbar.update(1)

    pbar.close()

    return trends_list

#insight
def summarize(tokenizer, model, text, n_words=None, compression=None, max_length=1000, num_beams=3, do_sample=False,
              repetition_penalty=10.0):
    if n_words:
        text = '[{}] '.format(n_words) + text
    elif compression:
        text = '[{0:.1g}] '.format(compression) + text
    x = tokenizer(text, return_tensors='pt', padding=True)

    with torch.inference_mode():
        out = model.generate(
            **x,
            max_length=max_length, num_beams=num_beams,
            do_sample=do_sample, repetition_penalty=repetition_penalty
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)

def get_insights(data, centroids_map, top_for_cluster=10, max_news_len=200, text_col='title_text'):
    t5_model = T5ForConditionalGeneration.from_pretrained('cointegrated/rut5-base-absum')
    t5_tokenizer = T5Tokenizer.from_pretrained('cointegrated/rut5-base-absum')

    top_articles_for_clusters = get_top_articles(data, centroids_map, top_k=top_for_cluster, text_col=text_col)

    insights_list = []
    pbar = tqdm(total=len(top_articles_for_clusters))
    pbar.set_description("Getting insights...")
    for top_articles, clust_label in top_articles_for_clusters:
        top_articles = [' '.join(row.split(' ')[:max_news_len]) for row in top_articles]
        #каждая новость в top_articles урезана до 200 слов. Эти новости соединили в одну строку и отдали на суммаризацию
        insights_list.append((summarize(t5_tokenizer, t5_model, '\n'.join(top_articles)), top_articles, clust_label))
        pbar.update(1)

    pbar.close()

    return insights_list

#digest
def get_digest(data, centroids_map, top_clusters=5):
    print("Getting digest...")
    top_label = data['label'].value_counts().iloc[:top_clusters].index.to_list()
    local_centroids = {label: centroids_map[label] for label in top_label}
    current_digest = get_top_articles(data, local_centroids, top_k=1, text_col=['title', 'content', 'channel_name', 'date'])
    #current_digest = [x[0][0] for x in current_digest]

    return current_digest