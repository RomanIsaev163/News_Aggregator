import json
import pandas as pd
from embedder import Encoder
from clustering import Agglomerative_Clustering
from features_extr import get_digest, get_trends, get_insights
from read_write_funcs import read_data_by_period

def get_response(start_date: str = None, end_date: str = None) -> json:
    """
        Получение инсайтов, трендов и дайджеста.
        Вход:
            df - DataFrame с колонками 'content' и 'date'
            start_date - строка задающая начало временного периода,
        в формате yyyy-mm-dd
            end_date - строка задающая конец временного периода.
        Выход:
            кортеж из неотформатированных результатов (инсайты, тренды, дайджест)
    """
    # выделение новостей из конкретного временного периода
    text_pool = read_data_by_period(start_date=start_date, end_date=end_date)
    model_name = 'cointegrated/rubert-tiny2'
    data_column = 'content'
    embeddings_pool = Encoder(model_name=model_name).encode_data(text_pool, data_column=data_column)
    text_pool['embedding'] = embeddings_pool
    affinity = 'cosine'
    linkage = 'average'
    distance_threshold = 0.2
    min_elements = 2
    top_clusters = 15
    clustered_data, centroids_map = Agglomerative_Clustering(text_pool, affinity=affinity,
                                                             linkage=linkage,
                                                             distance_threshold=distance_threshold,
                                                             col='embedding').clustering(min_elements=min_elements,
                                                                                         top_clusters=top_clusters)

    digest = get_digest(clustered_data, centroids_map, top_clusters=3)  # дайджест
    trends = get_trends(clustered_data, centroids_map, top_for_cluster=5, max_news_len=80)  # тренды
    insights = get_insights(clustered_data, centroids_map, top_for_cluster=30, max_news_len=100)  # инсайты

    return digest, trends, insights