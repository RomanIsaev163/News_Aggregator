import pandas as pd
from embedder import Encoder
from clustering import Agglomerative_Clustering
from features_extr import get_digest, get_trends, get_insights
from read_write_funcs import read_data_by_period
#data = pd.read_csv('../data/parsed/parsed.csv')

start_date, end_date = '2023-05-10', '2023-05-12'
data = read_data_by_period(start_date=start_date, end_date=end_date)
print(data['date'])
model_name = 'cointegrated/rubert-tiny2'
data_column = 'content'
embeddings_pool = Encoder(model_name=model_name).encode_data(data, data_column=data_column)
data['embedding'] = embeddings_pool
affinity = 'cosine'
linkage = 'average'
distance_threshold = 0.2
min_elements=2
top_clusters=15
clustered_data, centroids_map = Agglomerative_Clustering(data, affinity=affinity,
                                                     linkage=linkage,
                                                     distance_threshold=distance_threshold,
                                                     col='embedding').clustering(min_elements=min_elements,
                                                                                            top_clusters=top_clusters)
print(clustered_data.head())
print(clustered_data.columns)

#digest = get_digest(clustered_data, centroids_map, top_clusters=3)  # дайджест
#print(digest)



#trends = get_trends(clustered_data, centroids_map, top_for_cluster=5, max_news_len=80)  # тренды
#print(trends)
#insights = get_insights(clustered_data, centroids_map, top_for_cluster=30, max_news_len=100)  # инсайты


