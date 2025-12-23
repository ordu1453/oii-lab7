import numpy as np
from numpy.random import RandomState
from typing import Dict, List, Tuple, Optional, Any
import warnings


class GathGevaClustering:
    """
    Реализация алгоритма кластеризации Гат-Гева (Gath-Geva)
    
    Это нечеткий алгоритм кластеризации, который использует эллипсоидальные кластеры
    и адаптивные матрицы расстояний для лучшего выделения кластеров сложной формы.
    """
    
    def __init__(self, n_clusters: int = 3, m: float = 2.0,
                 max_iter: int = 100, epsilon: float = 1e-9,
                 random_state: Optional[int] = 42):
        """
        Инициализация алгоритма Гат-Гева
        
        Args:
            n_clusters: Количество кластеров
            m: Параметр нечеткости (должен быть > 1)
            max_iter: Максимальное количество итераций
            epsilon: Критерий остановки (изменение матрицы принадлежности)
            random_state: Seed для воспроизводимости
        """
        if m <= 1:
            raise ValueError("Параметр нечеткости m должен быть > 1")
            
        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.random_state = random_state
        self.random_generator = RandomState(random_state)
        
        # Результаты кластеризации
        self.membership_matrix = None
        self.centers = None
        self.covariance_matrices = None
        self.labels = None
        self.iterations = 0
        self.history = []
        
    def _initialize_membership(self, n_samples: int) -> np.ndarray:
        """
        Инициализация матрицы принадлежности случайными значениями
        
        Args:
            n_samples: Количество образцов
            
        Returns:
            Матрица принадлежности размером (n_samples, n_clusters)
        """
        # Генерация случайных чисел
        u = self.random_generator.randint(low=1, high=100, 
                                          size=(n_samples, self.n_clusters))
        
        # Нормализация по строкам (сумма принадлежностей = 1)
        u = u / u.sum(axis=1).reshape(-1, 1)
        
        # Корректировка для избежания нулевых сумм
        row_sums = u.sum(axis=1).reshape(-1, 1)
        u[:, 0] = u[:, 0] + np.where(row_sums.flatten() == 1, 
                                     row_sums.flatten() * 0, 
                                     1 - row_sums.flatten())
        return u
    
    def _initialize_centers_kmeans_pp(self, data: np.ndarray) -> np.ndarray:
        """
        Инициализация центров с помощью алгоритма k-means++
        
        Args:
            data: Матрица данных (n_samples, n_features)
            
        Returns:
            Центры кластеров (n_clusters, n_features)
        """
        n_samples, n_features = data.shape
        
        # Начинаем с случайного центра
        centers = np.zeros((self.n_clusters, n_features))
        first_idx = self.random_generator.randint(n_samples)
        centers[0] = data[first_idx]
        
        # Вычисление минимальных расстояний
        min_distances = np.full(n_samples, np.inf)
        
        for i in range(1, self.n_clusters):
            # Обновление минимальных расстояний до ближайшего центра
            for j in range(i):
                distances = np.sum((data - centers[j]) ** 2, axis=1)
                min_distances = np.minimum(min_distances, distances)
            
            # Вероятность выбора точки пропорциональна квадрату расстояния
            probabilities = min_distances / min_distances.sum()
            
            # Выбор следующего центра
            next_idx = self.random_generator.choice(n_samples, p=probabilities)
            centers[i] = data[next_idx]
        
        return centers
    
    def _calculate_distances(self, data: np.ndarray, centers: np.ndarray, 
                           cov_matrices: List[np.ndarray]) -> np.ndarray:
        """
        Вычисление адаптированных расстояний для алгоритма Гат-Гева
        
        Args:
            data: Матрица данных (n_samples, n_features)
            centers: Центры кластеров (n_clusters, n_features)
            cov_matrices: Список матриц ковариации для каждого кластера
            
        Returns:
            Матрица расстояний (n_samples, n_clusters)
        """
        n_samples, n_features = data.shape
        distances = np.zeros((n_samples, self.n_clusters))
        
        for j in range(self.n_clusters):
            # Приоритет кластера (вероятность)
            alpha_j = np.sum(self.membership_matrix[:, j] ** self.m) / np.sum(self.membership_matrix ** self.m)
            
            # Матрица ковариации для кластера j
            aj = cov_matrices[j]
            
            # Вычисление детерминанта (с защитой от вырожденных матриц)
            det_aj = np.linalg.det(aj)
            if det_aj <= 0:
                det_aj = 1e-10
                
            # Вычисление обратной матрицы (с псевдообратной для устойчивости)
            inv_aj = np.linalg.pinv(aj)
            
            # Вычисление расстояния для каждого образца
            for i in range(n_samples):
                z = data[i, :] - centers[j, :]
                mahalanobis_dist = z @ inv_aj @ z.T
                
                # Расстояние Гат-Гева
                d_ij = (np.sqrt(det_aj) / alpha_j) * np.exp(0.5 * mahalanobis_dist)
                distances[i, j] = d_ij
                
        return distances
    
    def _calculate_covariance_matrices(self, data: np.ndarray, centers: np.ndarray) -> List[np.ndarray]:
        """
        Вычисление матриц ковариации для каждого кластера
        
        Args:
            data: Матрица данных
            centers: Центры кластеров
            
        Returns:
            Список матриц ковариации
        """
        n_samples, n_features = data.shape
        cov_matrices = []
        
        for j in range(self.n_clusters):
            # Инициализация матрицы ковариации
            aj = np.zeros((n_features, n_features))
            
            # Веса принадлежности для кластера j
            weights = (self.membership_matrix[:, j] ** self.m).reshape(-1, 1)
            
            # Вычисление взвешенной ковариационной матрицы
            weighted_sum = 0
            for i in range(n_samples):
                z = data[i, :] - centers[j, :]
                z = z.reshape(-1, 1)
                aj += weights[i] * (z @ z.T)
                weighted_sum += weights[i]
            
            # Нормализация
            if weighted_sum > 0:
                aj = aj / weighted_sum
            else:
                aj = np.eye(n_features) * 1e-6
                
            # Добавление небольшого значения для устойчивости
            aj = aj + np.eye(n_features) * 1e-8
            cov_matrices.append(aj)
            
        return cov_matrices
    
    def fit(self, data: np.ndarray) -> 'GathGevaClustering':
        """
        Выполнение кластеризации Гат-Гева
        
        Args:
            data: Матрица данных для кластеризации (n_samples, n_features)
            
        Returns:
            self
        """
        n_samples, n_features = data.shape
        
        if n_samples < self.n_clusters:
            raise ValueError(f"Количество образцов ({n_samples}) должно быть >= количеству кластеров ({self.n_clusters})")
        
        # 1. Инициализация матрицы принадлежности
        self.membership_matrix = self._initialize_membership(n_samples)
        
        # 2. Инициализация центров с помощью k-means++
        self.centers = self._initialize_centers_kmeans_pp(data)
        
        # 3. Основной цикл алгоритма
        prev_u = self.membership_matrix + 2 * self.epsilon
        self.iterations = 0
        
        while (self.iterations < self.max_iter) and (np.linalg.norm(prev_u - self.membership_matrix) > self.epsilon):
            self.iterations += 1
            
            # Сохраняем предыдущие значения для отслеживания изменений
            prev_u = self.membership_matrix.copy()
            
            # 3.1. Вычисление центров кластеров
            weights = self.membership_matrix ** self.m
            weighted_sum = weights.sum(axis=0).reshape(-1, 1)
            
            # Защита от деления на ноль
            weighted_sum = np.where(weighted_sum == 0, 1e-10, weighted_sum)
            
            # Обновление центров
            self.centers = (weights.T @ data) / weighted_sum
            
            # 3.2. Вычисление матриц ковариации
            self.covariance_matrices = self._calculate_covariance_matrices(data, self.centers)
            
            # 3.3. Вычисление расстояний
            distances = self._calculate_distances(data, self.centers, self.covariance_matrices)
            
            # 3.4. Обновление матрицы принадлежности
            new_membership = np.zeros((n_samples, self.n_clusters))
            
            for i in range(n_samples):
                # Проверка на очень маленькие расстояния
                min_dist = np.min(distances[i])
                if min_dist < 1e-10:
                    # Если расстояние очень маленькое, образец точно принадлежит ближайшему кластеру
                    min_idx = np.argmin(distances[i])
                    new_membership[i, min_idx] = 1.0
                else:
                    # Нормальное вычисление принадлежности
                    for j in range(self.n_clusters):
                        sum_term = np.sum((distances[i, j] / distances[i]) ** (2 / (self.m - 1)))
                        new_membership[i, j] = 1.0 / sum_term
            
            self.membership_matrix = new_membership
            
            # Обеспечение, что сумма принадлежностей = 1
            row_sums = self.membership_matrix.sum(axis=1).reshape(-1, 1)
            row_sums = np.where(row_sums == 0, 1e-10, row_sums)
            self.membership_matrix = self.membership_matrix / row_sums
            
            # Ограничение значений принадлежности
            self.membership_matrix = np.clip(self.membership_matrix, 0, 1)
            
            # Сохранение истории для отладки
            self.history.append({
                'iteration': self.iterations,
                'centers': self.centers.copy(),
                'membership_norm': np.linalg.norm(prev_u - self.membership_matrix)
            })
        
        # 4. Определение четких меток кластеров
        self.labels = np.argmax(self.membership_matrix, axis=1)
        
        return self
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Предсказание принадлежности новых данных
        
        Args:
            data: Новые данные для классификации
            
        Returns:
            Матрица принадлежности для новых данных
        """
        if self.centers is None or self.covariance_matrices is None:
            raise ValueError("Модель должна быть обучена перед предсказанием")
        
        n_samples = data.shape[0]
        membership_new = np.zeros((n_samples, self.n_clusters))
        
        # Вычисление расстояний
        distances = self._calculate_distances(data, self.centers, self.covariance_matrices)
        
        # Вычисление принадлежности
        for i in range(n_samples):
            for j in range(self.n_clusters):
                sum_term = np.sum((distances[i, j] / distances[i]) ** (2 / (self.m - 1)))
                membership_new[i, j] = 1.0 / sum_term
        
        # Нормализация
        row_sums = membership_new.sum(axis=1).reshape(-1, 1)
        membership_new = membership_new / np.where(row_sums == 0, 1e-10, row_sums)
        
        return membership_new
    
    def get_cluster_info(self) -> Dict[str, Any]:
        """
        Получение информации о кластерах
        
        Returns:
            Словарь с информацией о кластерах
        """
        if self.labels is None:
            raise ValueError("Модель должна быть обучена")
        
        cluster_info = {}
        for cluster_id in range(self.n_clusters):
            cluster_indices = np.where(self.labels == cluster_id)[0]
            cluster_info[f'cluster_{cluster_id}'] = {
                'size': len(cluster_indices),
                'center': self.centers[cluster_id],
                'determinant': np.linalg.det(self.covariance_matrices[cluster_id])
            }
        
        return cluster_info


def cluster_files_gath_geva(vectors_dict: Dict[str, np.ndarray],
                           n_clusters: int = 3,
                           m: float = 2.0,
                           max_iter: int = 100,
                           epsilon: float = 1e-9,
                           random_state: int = 42) -> Dict[str, Any]:
    """
    Выполнение кластеризации файлов методом Гат-Гева
    
    Args:
        vectors_dict: Словарь {имя_файла: вектор}
        n_clusters: Количество кластеров
        m: Параметр нечеткости (> 1)
        max_iter: Максимальное количество итераций
        epsilon: Критерий остановки
        random_state: Seed для воспроизводимости
        
    Returns:
        Словарь с результатами кластеризации в формате, совместимом с визуализатором
    """
    # Подготовка данных
    filenames = list(vectors_dict.keys())
    vectors_list = list(vectors_dict.values())
    
    # Проверка, что все векторы имеют одинаковую размерность
    dims = [vec.shape[0] for vec in vectors_list]
    if len(set(dims)) > 1:
        raise ValueError("Все векторы должны иметь одинаковую размерность")
    
    # Преобразование в numpy массив
    data = np.array(vectors_list)
    
    # Выполнение кластеризации
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        gg = GathGevaClustering(n_clusters=n_clusters, m=m,
                               max_iter=max_iter, epsilon=epsilon,
                               random_state=random_state)
        gg.fit(data)
    
    # Формирование результата
    result = {
        'method': 'Gath-Geva',
        'vectors': data,
        'labels': gg.labels,
        'centers': gg.centers,
        'membership_matrix': gg.membership_matrix,
        'covariance_matrices': gg.covariance_matrices,
        'n_clusters': n_clusters,
        'parameters': {
            'm': m,
            'max_iter': max_iter,
            'epsilon': epsilon,
            'random_state': random_state,
            'iterations': gg.iterations
        },
        'filenames': filenames,
        'file_clusters': dict(zip(filenames, gg.labels)),
        'cluster_info': gg.get_cluster_info()
    }
    
    # Вычисление дополнительных метрик
    try:
        from sklearn.metrics import silhouette_score
        if n_clusters > 1 and len(set(gg.labels)) > 1:
            result['silhouette_score'] = silhouette_score(data, gg.labels)
    except ImportError:
        pass
    
    return result


def create_test_data(n_files: int = 100, n_features: int = 10, 
                    n_clusters: int = 3, random_state: int = 42) -> Dict[str, np.ndarray]:
    """
    Создание тестовых данных для проверки кластеризации
    
    Args:
        n_files: Количество файлов
        n_features: Размерность векторов
        n_clusters: Количество кластеров для генерации
        random_state: Seed для воспроизводимости
        
    Returns:
        Словарь с тестовыми данными
    """
    np.random.seed(random_state)
    
    # Создание центров кластеров
    cluster_centers = np.random.randn(n_clusters, n_features) * 3
    
    # Распределение файлов по кластерам
    vectors_dict = {}
    for i in range(n_files):
        # Выбор кластера
        cluster_id = i % n_clusters
        
        # Генерация вектора с шумом вокруг центра кластера
        vector = cluster_centers[cluster_id] + np.random.randn(n_features) * 0.5
        vectors_dict[f'file_{i:03d}.txt'] = vector
    
    return vectors_dict


def visualize_results_simple(result: Dict[str, Any]) -> None:
    """
    Простая визуализация результатов без внешних зависимостей
    
    Args:
        result: Результаты кластеризации
    """
    try:
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        
        print("\nСоздание простой визуализации...")
        
        # Уменьшение размерности
        vectors = result['vectors']
        labels = result['labels']
        
        if vectors.shape[1] > 2:
            pca = PCA(n_components=2)
            vectors_2d = pca.fit_transform(vectors)
        else:
            vectors_2d = vectors
        
        # Визуализация
        fig, ax = plt.subplots(figsize=(10, 8))
        
        scatter = ax.scatter(vectors_2d[:, 0], vectors_2d[:, 1], 
                           c=labels, cmap='tab10', alpha=0.7, s=100)
        
        # Центры кластеров
        if vectors.shape[1] > 2:
            centers_2d = pca.transform(result['centers'])
        else:
            centers_2d = result['centers']
        
        ax.scatter(centers_2d[:, 0], centers_2d[:, 1], 
                  c='red', marker='X', s=200, label='Центры')
        
        # Настройка графика
        ax.set_xlabel('Компонента 1')
        ax.set_ylabel('Компонента 2')
        ax.set_title(f'Кластеризация Гат-Гева (кластеров: {result["n_clusters"]})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.colorbar(scatter, label='Кластер')
        plt.tight_layout()
        plt.show()
        
    except ImportError as e:
        print(f"\nНе удалось создать визуализацию: {e}")
        print("Для визуализации установите matplotlib и scikit-learn:")
        print("pip install matplotlib scikit-learn")


if __name__ == "__main__":
    """
    Пример использования кластеризации Гат-Гева
    """
    # 1. Создание тестовых данных
    print("Создание тестовых данных...")
    test_data = create_test_data(n_files=50, n_features=5, n_clusters=3)
    
    # 2. Выполнение кластеризации
    print("\nВыполнение кластеризации Гат-Гева...")
    result = cluster_files_gath_geva(
        vectors_dict=test_data,
        n_clusters=3,
        m=2.0,
        max_iter=100,
        epsilon=1e-6,
        random_state=42
    )
    
    # 3. Вывод результатов
    print(f"\nМетод: {result['method']}")
    print(f"Количество кластеров: {result['n_clusters']}")
    print(f"Количество файлов: {len(result['filenames'])}")
    print(f"Выполнено итераций: {result['parameters']['iterations']}")
    
    if 'silhouette_score' in result:
        print(f"Silhouette Score: {result['silhouette_score']:.4f}")
    
    # 4. Информация о кластерах
    print("\nИнформация о кластерах:")
    for cluster_id in range(result['n_clusters']):
        cluster_size = np.sum(result['labels'] == cluster_id)
        print(f"  Кластер {cluster_id}: {cluster_size} файлов")
    
    # 5. Пример первых 10 файлов и их кластеров
    print("\nПример распределения файлов (первые 10):")
    filenames = result['filenames'][:10]
    labels = result['labels'][:10]
    for filename, label in zip(filenames, labels):
        print(f"  {filename}: кластер {label}")
    
    # 6. Простая визуализация
    visualize_results_simple(result)
    
    # 7. Сохранение результатов в файл
    print("\nСохранение результатов в файл 'gath_geva_results.npy'...")
    try:
        # Сохранение в бинарном формате
        np.save('gath_geva_results.npy', result, allow_pickle=True)
        print("Результаты сохранены успешно!")
        
        # Сохранение в текстовом формате
        with open('gath_geva_results.txt', 'w', encoding='utf-8') as f:
            f.write(f"Метод кластеризации: {result['method']}\n")
            f.write(f"Количество кластеров: {result['n_clusters']}\n")
            f.write(f"Количество файлов: {len(result['filenames'])}\n")
            f.write(f"Итераций: {result['parameters']['iterations']}\n\n")
            
            f.write("Распределение файлов по кластерам:\n")
            for cluster_id in range(result['n_clusters']):
                cluster_files = [fname for fname, label in result['file_clusters'].items() 
                               if label == cluster_id]
                f.write(f"\nКластер {cluster_id} ({len(cluster_files)} файлов):\n")
                for filename in cluster_files[:10]:  # Показываем только первые 10
                    f.write(f"  {filename}\n")
                if len(cluster_files) > 10:
                    f.write(f"  ... и еще {len(cluster_files) - 10} файлов\n")
        
        print("Текстовые результаты сохранены в 'gath_geva_results.txt'")
        
    except Exception as e:
        print(f"Ошибка при сохранении результатов: {e}")