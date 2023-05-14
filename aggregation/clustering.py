from sklearn.cluster import AgglomerativeClustering
import pandas as pd


class Agglomerative_Clustering:
    def __init__(self, text_pool,
                 affinity='cosine', linkage='average',
                 distance_threshold=0.2, col='text_embed'):
        """
            Класс для кластеризации эмбеддингов.
            На вход принмает информацию о новостях, эмбеддинги и количество кластеров.
            Если оно не задано, то задаётся эмпирически.
        """
        self.col = col
        self.df = text_pool.copy()
        self.affinity = affinity  # метрика cosine/euclid
        self.linkage = linkage  # метод объединения классов single/average/complete
        self.distance_threshold = distance_threshold  # порог ниже которого происходит объединение кластеров
        self.model = AgglomerativeClustering(n_clusters=None, affinity=self.affinity,
                                             linkage=self.linkage,
                                             distance_threshold=self.distance_threshold)

    def calculate_centroids(self, emb_lab_df):
        # return array: shape(num_clust, embeddings_len)
        # sorted from 0->num_clust-1
        unique_labels = sorted(emb_lab_df.loc[:, 'label'].unique(), reverse=False)
        embeddings_size = emb_lab_df.loc[0, self.col].shape[0]
        centroids_map = {i: [] for i in unique_labels}
        for label in unique_labels:
            centroid = emb_lab_df[emb_lab_df['label'] == label][self.col].mean()
            centroids_map[label] = centroid
        return centroids_map

    def clustering(self, min_elements=1, top_clusters=None):
        """
            Функция возвращает выполняющая поиск центроид кластеров.
            Выход:
                data - датафрейм с информацией о новостях, в который добавлены
            метки кластеров и эмбеддинги,
                centoids_map - центроиды кластеров
        """
        print("Clustering data...")
        self.model = self.model.fit(self.df.loc[:, self.col].to_list())
        model_labels = self.model.labels_
        print(f"Найдено {self.model.n_clusters_} кластеров")

        self.df['label'] = model_labels


        counted_labels = self.df['label'].value_counts()
        filtered_labels = counted_labels[counted_labels >= min_elements].sort_values(ascending=False)
        if top_clusters:
            filtered_labels = filtered_labels.iloc[:top_clusters].index
        else:
            filtered_labels = filtered_labels.index
        self.df = self.df[self.df['label'].isin(filtered_labels)].reset_index(drop=True)
        print(
            f'Взято {len(filtered_labels)} кластеров из {top_clusters} запрошенных в которых минимум {min_elements} элементов')
        if not self.df.empty:
            centroids_map = self.calculate_centroids(self.df.loc[:, [self.col, 'label']])
        else:
            centroids_map = {}
        print("Clustering done!")

        return self.df, centroids_map