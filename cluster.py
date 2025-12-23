import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
import warnings
from typing import Dict, Tuple, Any

class VectorClustering:
    def __init__(self):
        self.models = {}
        
    def k_means_clustering(self, vectors: np.ndarray, n_clusters: int = 3, 
                           random_state: int = 10, n_init: int = 10) -> Dict[str, Any]:

        if len(vectors) < n_clusters:
            n_clusters = max(2, len(vectors) // 2)
            
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
        labels = kmeans.fit_predict(vectors)
        
        # Вычисляем метрики качества
        if len(np.unique(labels)) > 1:
            try:
                silhouette = silhouette_score(vectors, labels)
            except:
                silhouette = -1
        else:
            silhouette = -1
            
        return {
            'method': 'K-means',
            'labels': labels,
            'centers': kmeans.cluster_centers_,
            'inertia': kmeans.inertia_,
            'silhouette_score': silhouette,
            'n_clusters': n_clusters
        }
    
    def fuzzy_c_means_clustering(self, vectors: np.ndarray, n_clusters: int = 3,
                                m: float = 2.0, max_iter: int = 100, 
                                error: float = 1e-5, random_state: int = 42) -> Dict[str, Any]:
        """
        Кластеризация методом Fuzzy C-means
        
        Args:
            vectors: Массив векторов для кластеризации
            n_clusters: Количество кластеров
            m: Параметр нечеткости (m > 1)
            max_iter: Максимальное количество итераций
            error: Критерий остановки
            
        Returns:
            Словарь с результатами кластеризации
        """
        if len(vectors) < n_clusters:
            n_clusters = max(2, len(vectors) // 2)
            
        np.random.seed(random_state)
        n_samples, n_features = vectors.shape
        
        # Инициализация матрицы принадлежности
        U = np.random.rand(n_samples, n_clusters)
        U = U / np.sum(U, axis=1, keepdims=True)
        
        for iteration in range(max_iter):
            U_old = U.copy()
            
            # Вычисление центров кластеров
            centers = np.zeros((n_clusters, n_features))
            for j in range(n_clusters):
                numerator = np.sum((U[:, j] ** m)[:, np.newaxis] * vectors, axis=0)
                denominator = np.sum(U[:, j] ** m)
                centers[j] = numerator / denominator
            
            # Вычисление расстояний
            distances = cdist(vectors, centers, metric='euclidean')
            distances = np.fmax(distances, np.finfo(np.float64).eps)
            
            # Обновление матрицы принадлежности
            temp = distances ** (-2/(m-1))
            U = temp / np.sum(temp, axis=1, keepdims=True)
            
            # Проверка критерия остановки
            if np.linalg.norm(U - U_old) < error:
                break
        
        # Жесткая кластеризация для меток
        labels = np.argmax(U, axis=1)
        
        # Вычисляем метрики качества
        if len(np.unique(labels)) > 1:
            try:
                silhouette = silhouette_score(vectors, labels)
            except:
                silhouette = -1
        else:
            silhouette = -1
            
        partition_coefficient = np.sum(U ** 2) / n_samples
        entropy_coefficient = -np.sum(U * np.log(U)) / n_samples
        
        return {
            'method': 'Fuzzy C-means',
            'labels': labels,
            'centers': centers,
            'membership_matrix': U,
            'silhouette_score': silhouette,
            'partition_coefficient': partition_coefficient,
            'entropy_coefficient': entropy_coefficient,
            'n_clusters': n_clusters,
            'n_iterations': iteration + 1
        }
    
    def gustafson_kessel_clustering(self, vectors: np.ndarray, n_clusters: int = 3,
                                   m: float = 2.0, max_iter: int = 100,
                                   error: float = 1e-5, random_state: int = 42) -> Dict[str, Any]:
        """
        Кластеризация методом Гюстафсона-Кесселя (Гат-Гевы)
        
        Args:
            vectors: Массив векторов для кластеризации
            n_clusters: Количество кластеров
            m: Параметр нечеткости (m > 1)
            max_iter: Максимальное количество итераций
            error: Критерий остановки
            
        Returns:
            Словарь с результатами кластеризации
        """
        if len(vectors) < n_clusters:
            n_clusters = max(2, len(vectors) // 2)
            
        np.random.seed(random_state)
        n_samples, n_features = vectors.shape
        
        # Инициализация матрицы принадлежности
        U = np.random.rand(n_samples, n_clusters)
        U = U / np.sum(U, axis=1, keepdims=True)
        
        # Инициализация матриц ковариации
        cov_matrices = [np.eye(n_features) for _ in range(n_clusters)]
        
        for iteration in range(max_iter):
            U_old = U.copy()
            
            # Вычисление центров кластеров
            centers = np.zeros((n_clusters, n_features))
            for j in range(n_clusters):
                numerator = np.sum((U[:, j] ** m)[:, np.newaxis] * vectors, axis=0)
                denominator = np.sum(U[:, j] ** m)
                centers[j] = numerator / denominator
            
            # Вычисление матриц ковариации и расстояний
            distances = np.zeros((n_samples, n_clusters))
            
            for j in range(n_clusters):
                # Вычисление матрицы ковариации
                diff = vectors - centers[j]
                weighted_diff = (U[:, j] ** m)[:, np.newaxis] * diff
                F_j = np.dot(weighted_diff.T, diff) / np.sum(U[:, j] ** m)
                
                # Добавление регуляризации для предотвращения вырожденности
                F_j = F_j + np.eye(n_features) * 1e-6
                
                # Вычисление определителя и нормировка
                rho = 1.0  # параметр объема
                det_F = np.linalg.det(F_j)
                if det_F <= 0:
                    det_F = 1e-6
                
                # Нормализованная матрица ковариации
                A_j = (rho * det_F) ** (1/n_features) * np.linalg.inv(F_j)
                
                # Вычисление расстояний Махалонобиса
                diff = vectors - centers[j]
                distances[:, j] = np.sum(np.dot(diff, A_j) * diff, axis=1)
                
                cov_matrices[j] = F_j
            
            distances = np.fmax(distances, np.finfo(np.float64).eps)
            
            # Обновление матрицы принадлежности
            temp = distances ** (-1/(m-1))
            U = temp / np.sum(temp, axis=1, keepdims=True)
            
            # Проверка критерия остановки
            if np.linalg.norm(U - U_old) < error:
                break
        
        # Жесткая кластеризация для меток
        labels = np.argmax(U, axis=1)
        
        # Вычисляем метрики качества
        if len(np.unique(labels)) > 1:
            try:
                silhouette = silhouette_score(vectors, labels)
            except:
                silhouette = -1
        else:
            silhouette = -1
            
        partition_coefficient = np.sum(U ** 2) / n_samples
        entropy_coefficient = -np.sum(U * np.log(U)) / n_samples
        
        return {
            'method': 'Gustafson-Kessel',
            'labels': labels,
            'centers': centers,
            'covariance_matrices': cov_matrices,
            'membership_matrix': U,
            'silhouette_score': silhouette,
            'partition_coefficient': partition_coefficient,
            'entropy_coefficient': entropy_coefficient,
            'n_clusters': n_clusters,
            'n_iterations': iteration + 1
        }
    
    def cluster_vectors(self, vectors_dict: Dict[str, np.ndarray], 
                       method: str = 'kmeans', n_clusters: int = 3, 
                       **kwargs) -> Dict[str, Any]:
        """
        Основная функция для кластеризации векторов
        
        Args:
            vectors_dict: Словарь {filename: numpy.array}
            method: Метод кластеризации ('kmeans', 'fcm', 'gk')
            n_clusters: Количество кластеров
            **kwargs: Дополнительные параметры для конкретных методов
            
        Returns:
            Словарь с результатами кластеризации и соответствием файлов кластерам
        """
        # Проверка входных данных
        if not vectors_dict:
            raise ValueError("Словарь векторов не должен быть пустым")
        
        # Извлечение векторов и имен файлов
        filenames = list(vectors_dict.keys())
        vectors = np.array(list(vectors_dict.values()))
        
        # Проверка размерности векторов
        if vectors.ndim != 2:
            raise ValueError("Векторы должны быть двумерным массивом")
        
        if len(vectors) < 2:
            raise ValueError("Для кластеризации необходимо как минимум 2 вектора")
        
        # Автоматический выбор количества кластеров, если не задано
        if n_clusters is None or n_clusters <= 0:
            n_clusters = min(5, max(2, len(vectors) // 3))
        
        # Выбор метода кластеризации
        method = method.lower()
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            if method == 'kmeans' or method == 'k-means':
                result = self.k_means_clustering(vectors, n_clusters, **kwargs)
            elif method == 'fcm' or method == 'fuzzy' or method == 'c-means':
                result = self.fuzzy_c_means_clustering(vectors, n_clusters, **kwargs)
            elif method == 'gk' or method == 'gustafson' or method == 'gustafson-kessel':
                result = self.gustafson_kessel_clustering(vectors, n_clusters, **kwargs)
            else:
                raise ValueError(f"Неизвестный метод: {method}. Доступные методы: 'kmeans', 'fcm', 'gk'")
        
        # Добавление соответствия файлов кластерам
        file_clusters = {filenames[i]: int(result['labels'][i]) 
                        for i in range(len(filenames))}
        
        result['file_clusters'] = file_clusters
        result['filenames'] = filenames
        result['vectors'] = vectors
        
        self.models[method] = result
        return result
    
    def find_optimal_clusters(self, vectors_dict: Dict[str, np.ndarray], 
                            method: str = 'kmeans', max_clusters: int = 10,
                            **kwargs) -> Dict[str, Any]:
        """
        Поиск оптимального количества кластеров
        
        Args:
            vectors_dict: Словарь {filename: numpy.array}
            method: Метод кластеризации
            max_clusters: Максимальное количество кластеров для проверки
            
        Returns:
            Результаты кластеризации с оптимальным количеством кластеров
        """
        vectors = np.array(list(vectors_dict.values()))
        max_clusters = min(max_clusters, len(vectors) - 1)
        
        best_score = -1
        best_result = None
        best_n = 2
        
        for n in range(2, max_clusters + 1):
            try:
                result = self.cluster_vectors(vectors_dict, method, n, **kwargs)
                score = result['silhouette_score']
                
                if score > best_score:
                    best_score = score
                    best_result = result
                    best_n = n
            except:
                continue
        
        if best_result is None:
            # Если не удалось найти оптимальное, используем 2 кластера
            best_result = self.cluster_vectors(vectors_dict, method, 2, **kwargs)
            best_n = 2
        
        best_result['optimal_n_clusters'] = best_n
        best_result['best_silhouette_score'] = best_score
        
        return best_result


# Пример использования функции
def cluster_files(vectors_dict: Dict[str, np.ndarray], 
                 method: str = 'kmeans', n_clusters: int = None,
                 find_optimal: bool = False, **kwargs) -> Dict[str, Any]:
    """
    Функция для кластеризации файлов по их векторным представлениям
    
    Args:
        vectors_dict: Словарь {filename: numpy.array}
        method: Метод кластеризации ('kmeans', 'fcm', 'gk')
        n_clusters: Количество кластеров (None для автоматического выбора)
        find_optimal: Найти оптимальное количество кластеров
        **kwargs: Дополнительные параметры
        
    Returns:
        Словарь с результатами кластеризации
    """
    clusterer = VectorClustering()
    
    if find_optimal and n_clusters is None:
        result = clusterer.find_optimal_clusters(vectors_dict, method, **kwargs)
    else:
        if n_clusters is None:
            n_clusters = min(5, max(2, len(vectors_dict) // 3))
        result = clusterer.cluster_vectors(vectors_dict, method, n_clusters, **kwargs)
    
    return result


# Пример использования
if __name__ == "__main__":
    # Создаем тестовые данные
    np.random.seed(42)
    n_files = 20
    vector_dim = 10
    
    # Создаем словарь с векторами
    vectors_dict = {
        f'file_{i}.txt': np.random.randn(vector_dim) for i in range(n_files)
    }
    
    # Добавляем структуру - создаем 3 группы
    for i in range(n_files):
        if i < 7:
            vectors_dict[f'file_{i}.txt'] += np.array([2, 2, 2] + [0] * (vector_dim - 3))
        elif i < 14:
            vectors_dict[f'file_{i}.txt'] += np.array([-2, -2, -2] + [0] * (vector_dim - 3))
        # Третья группа остается с исходными значениями
    
    # Пример 1: Кластеризация методом K-means
    print("Кластеризация методом K-means:")
    result_kmeans = cluster_files(vectors_dict, method='kmeans', n_clusters=3)
    print(f"Метод: {result_kmeans['method']}")
    print(f"Количество кластеров: {result_kmeans['n_clusters']}")
    print(f"Silhouette Score: {result_kmeans['silhouette_score']:.3f}")
    print(f"Метки кластеров: {result_kmeans['labels']}")
    print()
    
    # Пример 2: Нечеткая кластеризация C-means
    print("Кластеризация методом Fuzzy C-means:")
    result_fcm = cluster_files(vectors_dict, method='fcm', n_clusters=3, m=2.0)
    print(f"Метод: {result_fcm['method']}")
    print(f"Количество кластеров: {result_fcm['n_clusters']}")
    print(f"Silhouette Score: {result_fcm['silhouette_score']:.3f}")
    print(f"Коэффициент разделения: {result_fcm['partition_coefficient']:.3f}")
    print()
    
    # Пример 3: Кластеризация методом Гюстафсона-Кесселя
    print("Кластеризация методом Gustafson-Kessel:")
    result_gk = cluster_files(vectors_dict, method='gk', n_clusters=3, m=2.0)
    print(f"Метод: {result_gk['method']}")
    print(f"Количество кластеров: {result_gk['n_clusters']}")
    print(f"Silhouette Score: {result_gk['silhouette_score']:.3f}")
    print()
    
    # Пример 4: Поиск оптимального количества кластеров
    print("Поиск оптимального количества кластеров (K-means):")
    result_optimal = cluster_files(vectors_dict, method='kmeans', find_optimal=True, max_clusters=5)
    print(f"Оптимальное количество кластеров: {result_optimal['optimal_n_clusters']}")
    print(f"Лучший Silhouette Score: {result_optimal['best_silhouette_score']:.3f}")
    print()
    
    # Пример 5: Показ распределения файлов по кластерам
    print("Распределение файлов по кластерам (K-means):")
    for filename, cluster in result_kmeans['file_clusters'].items():
        print(f"  {filename}: кластер {cluster}")