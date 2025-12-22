import parser
import vectorization
import numpy as np
import cluster
import visualizer


def main():
    folder_path = "тексты/"
    results = parser.process_folder_simple(folder_path, "результаты.txt")
    print(results)
    # с лемматизацией
    vectors_lemmatized = vectorization.create_text_vectors(results, method='tfidf', use_lemmatization=True)
    print(vectors_lemmatized)
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
    result_fcm = cluster.cluster_files(vectors_lemmatized, method='fcm', n_clusters=7, m=0.0005)
    print(f"Метод: {result_fcm['method']}")
    print(f"Количество кластеров: {result_fcm['n_clusters']}")
    print(f"Silhouette Score: {result_fcm['silhouette_score']:.3f}")
    print(f"Коэффициент разделения: {result_fcm['partition_coefficient']:.3f}")
    print(f"Метки кластеров: {result_fcm['labels']}")
    print()

    # # Кластеризация методом Гат-Гевы
    # print("Кластеризация методом Gustafson-Kessel:")
    # result_gk = cluster.cluster_files(vectors_lemmatized, method='gk', n_clusters=7, m=2.0)
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
    # visualizer.visualize_clustering_result(result_fcm, plot_type='silhouette'
    #                           )
    # visualizer.visualize_clustering_result(result_gk, plot_type='2d_pca',
    #                           show_filenames=True,
    #                           filename_limit=None
    #                           )
    
    


if __name__ == "__main__":
    main()
