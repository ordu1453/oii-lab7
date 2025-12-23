import parser
import vectorization
import numpy as np
import cluster
import visualizer
import gg

import numpy as np
import multiprocessing
from mst_clustering.clustering_models import ZahnModel, GathGevaModel
from sklearn.datasets import make_blobs
from mst_clustering import Pipeline

    # Преобразование в матрицу
def dict_to_matrix(data_dict):
    """Преобразует словарь массивов в матрицу для кластеризации"""
    # Получаем список имен файлов для сохранения соответствия
    filenames = list(data_dict.keys())
    
    # Создаем матрицу признаков
    X = np.vstack([data_dict[name] for name in filenames])
    
    return X, filenames

def main():
    multiprocessing.freeze_support()

    folder_path = "тексты/"
    results = parser.process_folder_simple(folder_path, "результаты.txt")
    # print(results)

    # с лемматизацией
    vectors_lemmatized = vectorization.create_text_vectors(results, method='tfidf', use_lemmatization=True)
    # print(vectors_lemmatized)
    for filename, vector in vectors_lemmatized.items():
        print(f"\n{filename}:")
        print(f"  Размер вектора: {len(vector)}")
        print(f"  Ненулевые элементы: {np.sum(vector > 0)}")
        print(f"  Пример первых 10 значений: {vector[:10]}")

    # без лемматизации
    print("\n" + "="*50)
    print("Векторизация без лемматизации (словоформы, TF):")
    vectors_wordforms = vectorization.create_text_vectors(results, method='tf', use_lemmatization=False)
    for filename, vector in vectors_wordforms.items():
        print(f"\n{filename}:")
        print(f"  Размер вектора: {len(vector)}")
        print(f"  Ненулевые элементы: {np.sum(vector > 0)}")

    # Кластеризация методом K-means
    result_kmeans = cluster.cluster_files(vectors_lemmatized, method='kmeans', n_clusters=7)
    print(f"Метод: {result_kmeans['method']}")
    print(f"Количество кластеров: {result_kmeans['n_clusters']}")
    print(f"Silhouette Score: {result_kmeans['silhouette_score']:.3f}")
    print(f"Метки кластеров: {result_kmeans['labels']}")
    print()

    # Нечеткая кластеризация C-means
    print("Кластеризация методом Fuzzy C-means:")
    result_fcm = cluster.cluster_files(vectors_lemmatized, method='fcm', n_clusters=7, m=45)
    print(f"Метод: {result_fcm['method']}")
    print(f"Количество кластеров: {result_fcm['n_clusters']}")
    print(f"Silhouette Score: {result_fcm['silhouette_score']:.3f}")
    print(f"Коэффициент разделения: {result_fcm['partition_coefficient']:.3f}")
    print(f"Метки кластеров: {result_fcm['labels']}")
    print()

    # # Кластеризация методом Гат-Гевы
    # print("Кластеризация методом Gustafson-Kessel:")
    # result_gk = cluster.cluster_files(vectors_lemmatized, method='gk', n_clusters=7, m=0.05)
    # print(f"Метод: {result_gk['method']}")
    # print(f"Количество кластеров: {result_gk['n_clusters']}")
    # print(f"Silhouette Score: {result_gk['silhouette_score']:.3f}")
    # print(f"Метки кластеров: {result_gk['labels']}")
    # print()

    visualizer.visualize_clustering_result(result_kmeans, plot_type='2d_pca',
                              show_filenames=True,
                              filename_limit=None)
    visualizer.visualize_clustering_result(result_fcm, plot_type='2d_pca',
                              show_filenames=True,
                              filename_limit=None
                              )
    # visualizer.visualize_clustering_result(result_gk, plot_type='gk')
    # # visualizer.visualize_clustering_result(result_fcm, plot_type='silhouette'
    # #                           )
    # visualizer.visualize_clustering_result(result_gk, plot_type='2d_pca',
    #                           show_filenames=True,
    #                           filename_limit=None
    #                           )

    # result_gg = gg.cluster_files_gath_geva(
    #     vectors_dict=vectors_lemmatized,
    #     n_clusters=7,
    #     m=2.0,
    #     max_iter=100,
    #     epsilon=1e-6,
    #     random_state=42
    # )

    # visualizer.visualize_clustering_result(result_gg, plot_type='2d_pca',
    #                           show_filenames=True,
    #                           filename_limit=None
    #                           )
    
        # Использование
    X, filenames = dict_to_matrix(vectors_lemmatized)
    print("Матрица признаков:")
    print(X)
    print("\nСоответствие индексов файлам:")
    for i, name in enumerate(filenames):
        print(f"Индекс {i}: {name}")

    clustering = Pipeline(clustering_models=[
        ZahnModel(3, 1.5, 1e-4, max_num_of_clusters=7),GathGevaModel(0.0001, 2)
    ])
    # Теперь можно передать X в функцию кластеризации
    clustering.fit(data=X, workers_count=4)

    labels = clustering.labels
    partition = clustering.partition
    clusters_count = clustering.clusters_count

    print(labels)
    print(partition)
    print(clusters_count)
    

if __name__ == "__main__":
    main()
    
