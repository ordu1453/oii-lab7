import numpy as np
from scipy.spatial.distance import euclidean

def calculate_cluster_distances(data_dict, clustering_result, metric='euclidean'):
    """
    Вычисляет среднее межкластерное и внутрикластерное расстояние.
    
    Parameters:
    -----------
    data_dict : dict
        Словарь {filename: numpy.array, ...} с исходными данными
    clustering_result : dict
        Словарь с результатами кластеризации, содержащий ключ 'labels'
    metric : str
        Метрика расстояния ('euclidean', 'manhattan', 'cosine')
        
    Returns:
    --------
    tuple : (avg_inter_cluster_dist, avg_intra_cluster_dist)
    """
    # Преобразуем данные в удобный формат
    filenames = list(data_dict.keys())
    data_vectors = list(data_dict.values())
    labels = clustering_result['labels']
    
    # Проверка согласованности данных
    if len(data_vectors) != len(labels):
        raise ValueError(f"Несоответствие размеров: данных {len(data_vectors)}, меток {len(labels)}")
    
    # Получаем уникальные метки кластеров
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    # Группируем данные по кластерам
    clusters = {}
    for label in unique_labels:
        # Индексы объектов, принадлежащих текущему кластеру
        indices = np.where(labels == label)[0]
        # Сохраняем векторы этих объектов
        clusters[label] = [data_vectors[i] for i in indices]
    
    print(f"Найдено {n_clusters} кластеров")
    for label in unique_labels:
        print(f"  Кластер {label}: {len(clusters[label])} объектов")
    
    # Функция для вычисления расстояния между двумя векторами
    def calculate_distance(vec1, vec2):
        if metric == 'euclidean':
            return euclidean(vec1, vec2)
        elif metric == 'manhattan':
            return np.sum(np.abs(vec1 - vec2))
        elif metric == 'cosine':
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            return 1 - dot_product / (norm1 * norm2) if norm1 * norm2 > 0 else 1
        else:
            raise ValueError(f"Неизвестная метрика: {metric}")
    
    # 1. Вычисляем среднее внутрикластерное расстояние
    print("\n1. Вычисляем среднее внутрикластерное расстояние...")
    intra_cluster_distances = []
    intra_pair_counts = 0
    
    for label, vectors in clusters.items():
        n_vectors = len(vectors)
        if n_vectors > 1:
            # Для каждого кластера вычисляем все попарные расстояния
            for i in range(n_vectors):
                for j in range(i + 1, n_vectors):  # чтобы избежать дублирования
                    dist = calculate_distance(vectors[i], vectors[j])
                    intra_cluster_distances.append(dist)
                    intra_pair_counts += 1
    
    avg_intra_cluster = np.mean(intra_cluster_distances) if intra_cluster_distances else 0
    
    # 2. Вычисляем среднее межкластерное расстояние
    print("2. Вычисляем среднее межкластерное расстояние...")
    inter_cluster_distances = []
    inter_pair_counts = 0
    
    # Получаем список всех кластеров для итерации
    cluster_labels = list(clusters.keys())
    
    for idx_i in range(len(cluster_labels)):
        for idx_j in range(idx_i + 1, len(cluster_labels)):  # чтобы избежать дублирования
            label_i = cluster_labels[idx_i]
            label_j = cluster_labels[idx_j]
            
            # Вычисляем все расстояния между объектами двух разных кластеров
            for vec_i in clusters[label_i]:
                for vec_j in clusters[label_j]:
                    dist = calculate_distance(vec_i, vec_j)
                    inter_cluster_distances.append(dist)
                    inter_pair_counts += 1
    
    avg_inter_cluster = np.mean(inter_cluster_distances) if inter_cluster_distances else 0
    
    # 3. Выводим результаты
    print("\n" + "="*60)
    print("РЕЗУЛЬТАТЫ АНАЛИЗА РАССТОЯНИЙ")
    print("="*60)
    
    print(f"\nСреднее внутрикластерное расстояние:")
    print(f"  Значение: {avg_intra_cluster:.4f}")
    print(f"  На основе: {intra_pair_counts} пар объектов")
    print(f"  Минимум: {np.min(intra_cluster_distances):.4f}" if intra_cluster_distances else "  Минимум: N/A")
    print(f"  Максимум: {np.max(intra_cluster_distances):.4f}" if intra_cluster_distances else "  Максимум: N/A")
    
    print(f"\nСреднее межкластерное расстояние:")
    print(f"  Значение: {avg_inter_cluster:.4f}")
    print(f"  На основе: {inter_pair_counts} пар объектов")
    print(f"  Минимум: {np.min(inter_cluster_distances):.4f}" if inter_cluster_distances else "  Минимум: N/A")
    print(f"  Максимум: {np.max(inter_cluster_distances):.4f}" if inter_cluster_distances else "  Максимум: N/A")
    
    print(f"\nСоотношение (меж/внутри): {avg_inter_cluster/avg_intra_cluster:.4f}" 
          if avg_intra_cluster > 0 else "\nСоотношение (меж/внутри): бесконечность (внутрикластерное расстояние = 0)")
    
    return avg_inter_cluster, avg_intra_cluster


# Пример использования функции:
def example_usage():
    """
    Пример использования функции calculate_cluster_distances
    """
    # Создаем пример данных
    np.random.seed(42)
    
    # Исходный словарь с данными
    data_dict = {}
    for i in range(100):
        data_dict[f"file_{i}.jpg"] = np.random.randn(10)  # 10-мерные векторы
    
    # Пример результата кластеризации (как у вас в задании)
    clustering_result = {
        'method': 'Fuzzy C-means',
        'labels': np.random.randint(0, 3, 100),  # 3 кластера
        'centers': [np.random.randn(10) for _ in range(3)],
        'membership_matrix': np.random.rand(100, 3),
        'silhouette_score': 0.5,
        'partition_coefficient': 0.8,
        'entropy_coefficient': 0.2,
        'n_clusters': 3,
        'n_iterations': 10
    }
    
    # Вычисляем расстояния
    print("Пример использования функции:")
    print("-" * 40)
    avg_inter, avg_intra = calculate_cluster_distances(data_dict, clustering_result, metric='euclidean')
    
    # Дополнительный анализ
    print("\n" + "="*60)
    print("ИНТЕРПРЕТАЦИЯ РЕЗУЛЬТАТОВ:")
    print("="*60)
    print("""
1. Внутрикластерное расстояние показывает, насколько компактны кластеры.
   - Чем МЕНЬЕ значение, тем более плотные и однородные кластеры.
   
2. Межкластерное расстояние показывает, насколько хорошо кластеры разделены.
   - Чем БОЛЬШЕ значение, тем лучше разделены кластеры.
   
3. Соотношение меж/внутри должно быть большим для хорошей кластеризации.
   - Идеальный случай: маленькое внутрикластерное и большое межкластерное расстояние.
    """)

# Если хотите протестировать, раскомментируйте следующую строку:
example_usage()
