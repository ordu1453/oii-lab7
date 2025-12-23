import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Ellipse
from matplotlib.colors import ListedColormap
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import warnings
from matplotlib import font_manager

class ClusteringVisualizer:
    def __init__(self, figsize=(15, 10)):
        self.figsize = figsize
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
    def _reduce_dimensions(self, vectors: np.ndarray, n_components: int = 2,
                          method: str = 'pca') -> np.ndarray:
        """
        Уменьшение размерности для визуализации
        
        Args:
            vectors: Исходные векторы
            n_components: Целевая размерность
            method: Метод уменьшения размерности ('pca', 'tsne')
            
        Returns:
            Векторы в уменьшенной размерности
        """
        if vectors.shape[1] <= n_components:
            return vectors[:, :n_components]
            
        if method == 'pca':
            reducer = PCA(n_components=n_components, random_state=42)
        elif method == 'tsne':
            reducer = TSNE(n_components=n_components, random_state=42,
                          perplexity=min(30, len(vectors) - 1))
        else:
            raise ValueError(f"Неизвестный метод: {method}")
            
        return reducer.fit_transform(vectors)
    
    def plot_clusters_2d(self, result: Dict[str, Any], 
                        reduction_method: str = 'pca',
                        show_centers: bool = True,
                        show_filenames: bool = True,
                        filename_limit: int = None,
                        filename_fontsize: int = 8,
                        filename_alpha: float = 0.7,
                        filename_offset: Tuple[float, float] = (5, 5),
                        highlight_files: List[str] = None,
                        highlight_color: str = 'red',
                        highlight_size: int = 150,
                        show_cluster_labels: bool = True,
                        alpha: float = 0.7,
                        figsize: Optional[tuple] = None) -> plt.Figure:
        """
        Визуализация кластеров в 2D пространстве с именами файлов
        
        Args:
            result: Результаты кластеризации
            reduction_method: Метод уменьшения размерности ('pca', 'tsne')
            show_centers: Показывать центры кластеров
            show_filenames: Показывать имена файлов
            filename_limit: Максимальное количество файлов для отображения (None = все)
            filename_fontsize: Размер шрифта для имен файлов
            filename_alpha: Прозрачность текста файлов
            filename_offset: Смещение текста от точки (dx, dy)
            highlight_files: Список файлов для выделения
            highlight_color: Цвет выделенных файлов
            highlight_size: Размер точек выделенных файлов
            show_cluster_labels: Показывать метки кластеров на центрах
            alpha: Прозрачность точек
            figsize: Размер фигуры
            
        Returns:
            Объект matplotlib Figure
        """
        if figsize is None:
            figsize = self.figsize
            
        vectors = result['vectors']
        labels = result['labels']
        n_clusters = result['n_clusters']
        method_name = result['method']
        
        # Получаем имена файлов
        if 'filenames' in result:
            filenames = result['filenames']
        elif 'file_clusters' in result:
            filenames = list(result['file_clusters'].keys())
        else:
            filenames = [f'Объект_{i}' for i in range(len(vectors))]
        
        # Уменьшение размерности
        vectors_2d = self._reduce_dimensions(vectors, 2, reduction_method)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Создание цветовой карты
        cmap = cm.get_cmap('tab10', n_clusters)
        
        # Определяем, какие файлы выделять
        highlight_indices = []
        if highlight_files:
            highlight_indices = [i for i, fname in enumerate(filenames) 
                               if fname in highlight_files]
        
        # Отрисовка обычных точек (не выделенных)
        if highlight_indices:
            # Создаем маску для невыделенных точек
            mask = np.ones(len(vectors), dtype=bool)
            mask[highlight_indices] = False
            
            if np.any(mask):
                scatter = ax.scatter(vectors_2d[mask, 0], vectors_2d[mask, 1],
                                   c=labels[mask], cmap=cmap, alpha=alpha, s=100,
                                   edgecolors='white', linewidth=1, zorder=1)
        
        # Отрисовка выделенных точек
        if highlight_indices:
            highlight_labels = [labels[i] for i in highlight_indices]
            ax.scatter(vectors_2d[highlight_indices, 0], 
                      vectors_2d[highlight_indices, 1],
                      c=highlight_color, alpha=1.0, s=highlight_size,
                      edgecolors='black', linewidth=2, 
                      marker='*', zorder=3,
                      label='Выделенные файлы')
        
        # Если нет выделенных точек, рисуем все точки обычным способом
        if not highlight_indices:
            scatter = ax.scatter(vectors_2d[:, 0], vectors_2d[:, 1],
                               c=labels, cmap=cmap, alpha=alpha, s=100,
                               edgecolors='white', linewidth=1, zorder=1)
        
        # Отрисовка имен файлов
        if show_filenames:
            # Определяем, сколько файлов показывать
            if filename_limit and filename_limit < len(filenames):
                indices_to_label = np.random.choice(len(filenames), 
                                                   filename_limit, 
                                                   replace=False)
            else:
                indices_to_label = range(len(filenames))
            
            # Добавляем выделенные файлы в список для отображения имен
            if highlight_indices:
                indices_to_label = list(indices_to_label) + highlight_indices
                indices_to_label = list(set(indices_to_label))  # Убираем дубликаты
            
            for i in indices_to_label:
                x, y = vectors_2d[i]
                filename = filenames[i]
                
                # Настройка цвета текста для выделенных файлов
                text_color = 'black'
                text_weight = 'normal'
                text_alpha = filename_alpha
                
                if i in highlight_indices:
                    text_color = highlight_color
                    text_weight = 'bold'
                    text_alpha = 1.0
                
                ax.annotate(filename, xy=(x, y), 
                          xytext=filename_offset,
                          textcoords='offset points',
                          fontsize=filename_fontsize,
                          fontweight=text_weight,
                          color=text_color,
                          alpha=text_alpha,
                          bbox=dict(boxstyle='round,pad=0.2',
                                  facecolor='white',
                                  alpha=0.7,
                                  edgecolor='none'),
                          zorder=2)
        
        # Отрисовка центров кластеров
        if show_centers and 'centers' in result:
            centers = result['centers']
            # Уменьшение размерности центров
            if reduction_method == 'pca':
                pca = PCA(n_components=2, random_state=42).fit(vectors)
                centers_2d = pca.transform(centers)
            else:
                # Для t-SNE центры нужно вычислять иначе
                centers_2d = np.array([np.mean(vectors_2d[labels == i], axis=0) 
                                      for i in range(n_clusters)])
            
            center_scatter = ax.scatter(centers_2d[:, 0], centers_2d[:, 1],
                                      c=range(n_clusters), cmap=cmap, s=300,
                                      marker='X', edgecolors='black', linewidth=2,
                                      label='Центры кластеров', zorder=4)
            
            # Добавление номеров центров
            if show_cluster_labels:
                for i, center in enumerate(centers_2d):
                    ax.annotate(f'Кластер {i}', xy=center, 
                              xytext=(10, 10),
                              textcoords='offset points', 
                              fontsize=11,
                              fontweight='bold',
                              color='black',
                              bbox=dict(boxstyle='round,pad=0.3',
                                      facecolor='yellow',
                                      alpha=0.7))
        
        # Настройка графика
        # reduction_name = 'PCA' if reduction_method == 'pca' else 't-SNE'
        # ax.set_xlabel(f'Компонента 1 ({reduction_name})', fontsize=12)
        # ax.set_ylabel(f'Компонента 2 ({reduction_name})', fontsize=12)
        
        title = f'Кластеризация методом {method_name}\n'
        title += f'Кластеров: {n_clusters}, '
        title += f'Объектов: {len(vectors)}'
        
        if 'silhouette_score' in result:
            title += f', Silhouette Score: {result["silhouette_score"]:.3f}'
            
        if show_filenames and filename_limit:
            title += f'\nПоказано имен файлов: {filename_limit} из {len(filenames)}'
        
        # ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Добавление легенды для кластеров
        if not highlight_indices or len(set(labels)) > 1:
            legend_elements = []
            for i in range(n_clusters):
                color = cmap(i)
                legend_elements.append(
                    plt.Line2D([0], [0], marker='o', color='w',
                             markerfacecolor=color,
                             markersize=10, 
                             label=f'Кластер {i} ({np.sum(labels == i)} объектов)')
                )
            
            # Добавляем элементы легенды для выделенных файлов, если они есть
            if highlight_indices:
                legend_elements.append(
                    plt.Line2D([0], [0], marker='*', color='w',
                             markerfacecolor=highlight_color,
                             markersize=12,
                             markeredgecolor='black',
                             markeredgewidth=1,
                             label=f'Выделенные файлы ({len(highlight_indices)})')
                )
            
            ax.legend(handles=legend_elements, loc='best', 
                     fontsize=10, framealpha=0.9)
        
        ax.grid(True, alpha=0.3)
        
        # Добавляем информацию о распределении по кластерам
        if n_clusters > 0 and n_clusters <= 10:
            cluster_info = "\nРаспределение по кластерам:\n"
            for i in range(n_clusters):
                count = np.sum(labels == i)
                percentage = count / len(labels) * 100
                cluster_info += f"Кластер {i}: {count} объектов ({percentage:.1f}%)\n"
            
            # Убираем последний перенос строки
            cluster_info = cluster_info.strip()
            
            # Добавляем текст на график
            # plt.figtext(0.02, 0.02, cluster_info, 
            #            fontsize=9, 
            #            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            #            verticalalignment='bottom')
        
        plt.tight_layout()
        return fig
    
    def plot_interactive_clusters(self, result: Dict[str, Any],
                                 reduction_method: str = 'pca',
                                 save_path: Optional[str] = None) -> None:
        """
        Создание интерактивной визуализации с отображением имен файлов при наведении
        
        Args:
            result: Результаты кластеризации
            reduction_method: Метод уменьшения размерности
            save_path: Путь для сохранения HTML файла
        """
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            
            vectors = result['vectors']
            labels = result['labels']
            
            # Получаем имена файлов
            if 'filenames' in result:
                filenames = result['filenames']
            elif 'file_clusters' in result:
                filenames = list(result['file_clusters'].keys())
            else:
                filenames = [f'Объект_{i}' for i in range(len(vectors))]
            
            # Уменьшение размерности
            vectors_2d = self._reduce_dimensions(vectors, 2, reduction_method)
            
            # Создаем DataFrame для Plotly
            import pandas as pd
            df = pd.DataFrame({
                'x': vectors_2d[:, 0],
                'y': vectors_2d[:, 1],
                'Кластер': labels,
                'Файл': filenames,
                'Размер_кластера': [np.sum(labels == label) for label in labels]
            })
            
            # Создаем интерактивный график
            fig = px.scatter(df, x='x', y='y', 
                           color='Кластер',
                           hover_name='Файл',
                           hover_data=['Кластер', 'Размер_кластера'],
                           title=f'Интерактивная визуализация кластеризации ({result["method"]})',
                           labels={'x': f'Компонента 1 ({reduction_method.upper()})',
                                  'y': f'Компонента 2 ({reduction_method.upper()})'},
                           template='plotly_white')
            
            # Настраиваем отображение
            fig.update_traces(marker=dict(size=10, line=dict(width=1, color='white')),
                            selector=dict(mode='markers'))
            
            if save_path:
                fig.write_html(save_path)
                print(f"Интерактивная визуализация сохранена: {save_path}")
            
            fig.show()
            
        except ImportError:
            print("Для интерактивной визуализации установите plotly:")
            print("pip install plotly")
            
            # Создаем статическую визуализацию вместо интерактивной
            self.plot_clusters_2d(result, reduction_method=reduction_method, 
                                show_filenames=True, filename_limit=20)
    
    def plot_cluster_with_file_table(self, result: Dict[str, Any],
                                    reduction_method: str = 'pca',
                                    max_files_per_cluster: int = 10,
                                    figsize: Optional[tuple] = None) -> plt.Figure:
        """
        Визуализация кластеров с таблицей файлов
        
        Args:
            result: Результаты кластеризации
            reduction_method: Метод уменьшения размерности
            max_files_per_cluster: Максимальное количество файлов для отображения в таблице
            figsize: Размер фигуры
            
        Returns:
            Объект matplotlib Figure
        """
        if figsize is None:
            figsize = (18, 12)
        
        # Создаем основной график
        fig = self.plot_clusters_2d(result, reduction_method=reduction_method,
                                  show_filenames=False, show_centers=True,
                                  figsize=(12, 8))
        
        # Создаем новую фигуру с двумя областями
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, 
                                      gridspec_kw={'width_ratios': [2, 1]})
        
        vectors = result['vectors']
        labels = result['labels']
        n_clusters = result['n_clusters']
        
        # Получаем имена файлов
        if 'filenames' in result:
            filenames = result['filenames']
        elif 'file_clusters' in result:
            filenames = list(result['file_clusters'].keys())
        else:
            filenames = [f'Объект_{i}' for i in range(len(vectors))]
        
        # Уменьшение размерности
        vectors_2d = self._reduce_dimensions(vectors, 2, reduction_method)
        
        # График кластеров на левой панели
        cmap = cm.get_cmap('tab10', n_clusters)
        scatter = ax1.scatter(vectors_2d[:, 0], vectors_2d[:, 1],
                            c=labels, cmap=cmap, alpha=0.7, s=100,
                            edgecolors='white', linewidth=1)
        
        # Настройка левой панели
        reduction_name = 'PCA' if reduction_method == 'pca' else 't-SNE'
        ax1.set_xlabel(f'Компонента 1 ({reduction_name})', fontsize=12)
        ax1.set_ylabel(f'Компонента 2 ({reduction_name})', fontsize=12)
        ax1.set_title(f'Визуализация кластеров\n{result["method"]}', 
                     fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Таблица файлов на правой панели
        ax2.axis('off')
        ax2.set_title('Файлы по кластерам', fontsize=14, fontweight='bold', 
                     pad=20, loc='left')
        
        # Создаем текст с файлами по кластерам
        table_text = ""
        
        for cluster_id in range(n_clusters):
            cluster_indices = np.where(labels == cluster_id)[0]
            cluster_files = [filenames[i] for i in cluster_indices]
            
            table_text += f"\n{'='*50}\n"
            table_text += f"Кластер {cluster_id}: {len(cluster_files)} файлов\n"
            table_text += f"{'='*50}\n"
            
            # Ограничиваем количество отображаемых файлов
            files_to_show = cluster_files[:max_files_per_cluster]
            for i, filename in enumerate(files_to_show, 1):
                table_text += f"{i:2d}. {filename}\n"
            
            if len(cluster_files) > max_files_per_cluster:
                table_text += f"... и еще {len(cluster_files) - max_files_per_cluster} файлов\n"
        
        # Отображаем текст
        ax2.text(0, 1, table_text, fontfamily='monospace', fontsize=9,
                verticalalignment='top', transform=ax2.transAxes,
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        return fig

    
    def plot_membership_matrix(self, result: Dict[str, Any],
                              figsize: Optional[tuple] = None) -> plt.Figure:
        """
        Визуализация матрицы принадлежности для нечетких методов
        
        Args:
            result: Результаты кластеризации
            figsize: Размер фигуры
            
        Returns:
            Объект matplotlib Figure
        """
        if 'membership_matrix' not in result:
            raise ValueError("Матрица принадлежности не найдена в результатах")
            
        if figsize is None:
            figsize = self.figsize
            
        U = result['membership_matrix']
        labels = result['labels']
        n_clusters = result['n_clusters']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Тепловая карта матрицы принадлежности
        im = ax1.imshow(U.T, aspect='auto', cmap='viridis',
                       interpolation='nearest')
        ax1.set_xlabel('Объекты', fontsize=12)
        ax1.set_ylabel('Кластеры', fontsize=12)
        ax1.set_title('Матрица принадлежности', fontsize=14, fontweight='bold')
        
        # Добавление цветовой шкалы
        plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
        
        # Упорядочивание по кластерам
        sorted_indices = np.argsort(labels)
        U_sorted = U[sorted_indices]
        
        # График принадлежности
        x = np.arange(len(U))
        bottom = np.zeros(len(U))
        
        for i in range(n_clusters):
            ax2.bar(x, U_sorted[:, i], bottom=bottom, label=f'Кластер {i}')
            bottom += U_sorted[:, i]
        
        ax2.set_xlabel('Объекты (отсортированы по кластерам)', fontsize=12)
        ax2.set_ylabel('Степень принадлежности', fontsize=12)
        ax2.set_title('Распределение принадлежности по кластерам',
                     fontsize=14, fontweight='bold')
        ax2.legend(loc='upper right')
        ax2.set_ylim([0, 1.1])
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_gk_clusters_with_ellipses(self, result: Dict[str, Any],
                                      reduction_method: str = 'pca',
                                      n_std: float = 2.0,
                                      figsize: Optional[tuple] = None) -> plt.Figure:
        """
        Визуализация кластеров метода Гюстафсона-Кесселя с эллипсами
        
        Args:
            result: Результаты кластеризации
            reduction_method: Метод уменьшения размерности
            n_std: Количество стандартных отклонений для эллипсов
            figsize: Размер фигуры
            
        Returns:
            Объект matplotlib Figure
        """
        if 'covariance_matrices' not in result:
            raise ValueError("Матрицы ковариации не найдены в результатах")
            
        if figsize is None:
            figsize = (12, 10)
            
        vectors = result['vectors']
        labels = result['labels']
        centers = result['centers']
        cov_matrices = result['covariance_matrices']
        n_clusters = result['n_clusters']
        
        # Уменьшение размерности
        vectors_2d = self._reduce_dimensions(vectors, 2, reduction_method)
        
        # Уменьшение размерности центров и матриц ковариации
        if reduction_method == 'pca':
            pca = PCA(n_components=2, random_state=42).fit(vectors)
            centers_2d = pca.transform(centers)
            
            # Проекция матриц ковариации
            cov_matrices_2d = []
            for cov in cov_matrices:
                # Приближенная проекция через PCA компоненты
                cov_2d = pca.components_ @ cov @ pca.components_.T
                cov_matrices_2d.append(cov_2d)
        else:
            # Для t-SNE используем эмпирические оценки
            centers_2d = np.array([np.mean(vectors_2d[labels == i], axis=0) 
                                  for i in range(n_clusters)])
            cov_matrices_2d = []
            for i in range(n_clusters):
                cluster_points = vectors_2d[labels == i]
                if len(cluster_points) > 1:
                    cov_2d = np.cov(cluster_points.T)
                else:
                    cov_2d = np.eye(2) * 0.1
                cov_matrices_2d.append(cov_2d)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Создание цветовой карты
        cmap = cm.get_cmap('tab10', n_clusters)
        
        # Отрисовка точек
        scatter = ax.scatter(vectors_2d[:, 0], vectors_2d[:, 1],
                           c=labels, cmap=cmap, alpha=0.6, s=80,
                           edgecolors='white', linewidth=1)
        
        # Отрисовка эллипсов для каждого кластера
        for i in range(n_clusters):
            color = cmap(i)
            
            # Центр эллипса
            center = centers_2d[i]
            
            # Ковариационная матрица
            cov = cov_matrices_2d[i]
            
            # Вычисление собственных значений и векторов для эллипса
            if np.linalg.det(cov) > 0:
                eigvals, eigvecs = np.linalg.eigh(cov)
                
                # Угол поворота эллипса
                angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
                
                # Полуоси эллипса
                width, height = 2 * n_std * np.sqrt(eigvals)
                
                # Отрисовка эллипса
                ellipse = Ellipse(xy=center, width=width, height=height,
                                angle=angle, edgecolor=color,
                                facecolor=color, alpha=0.2, linewidth=2)
                ax.add_patch(ellipse)
            
            # Отрисовка центра
            ax.scatter(center[0], center[1], c=[color], s=200,
                      marker='X', edgecolors='black', linewidth=2)
            
            # Подпись центра
            ax.annotate(f'C{i}', xy=center, xytext=(5, 5),
                      textcoords='offset points', fontsize=12,
                      fontweight='bold', color='black')
        
        # Настройка графика
        ax.set_xlabel(f'Компонента 1 ({reduction_method.upper()})', fontsize=12)
        ax.set_ylabel(f'Компонента 2 ({reduction_method.upper()})', fontsize=12)
        ax.set_title(f'Кластеризация методом {result["method"]}\n'
                    f'Эллипсы показывают {n_std}σ область кластеров',
                    fontsize=14, fontweight='bold')
        
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return fig
    
    def plot_silhouette_analysis(self, result: Dict[str, Any],
                                figsize: Optional[tuple] = None) -> plt.Figure:
        """
        Анализ силуэтов для кластеров
        
        Args:
            result: Результаты кластеризации
            figsize: Размер фигуры
            
        Returns:
            Объект matplotlib Figure
        """
        from sklearn.metrics import silhouette_samples
        
        if figsize is None:
            figsize = (12, 8)
            
        vectors = result['vectors']
        labels = result['labels']
        n_clusters = result['n_clusters']
        
        # Вычисление силуэтных коэффициентов для каждого образца
        silhouette_vals = silhouette_samples(vectors, labels)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # График силуэтов
        y_lower = 10
        for i in range(n_clusters):
            # Сбор и сортировка силуэтных коэффициентов для кластера i
            ith_cluster_silhouette_vals = silhouette_vals[labels == i]
            ith_cluster_silhouette_vals.sort()
            
            size_cluster_i = ith_cluster_silhouette_vals.shape[0]
            y_upper = y_lower + size_cluster_i
            
            color = cm.get_cmap('tab10')(i / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                            0, ith_cluster_silhouette_vals,
                            facecolor=color, edgecolor=color, alpha=0.7)
            
            # Подпись кластеров
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            
            # Вычисление нового y_lower для следующего графика
            y_lower = y_upper + 10  # 10 для пробела между кластерами
        
        ax1.set_xlabel("Значение силуэтного коэффициента", fontsize=12)
        ax1.set_ylabel("Кластер", fontsize=12)
        
        # Вертикальная линия для среднего силуэтного коэффициента
        silhouette_avg = np.mean(silhouette_vals)
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--",
                   label=f'Среднее: {silhouette_avg:.3f}')
        
        ax1.set_yticks([])  # Очистка меток по y
        ax1.set_xticks([-0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
        ax1.legend(loc='best')
        ax1.set_title(f"Анализ силуэтов\n{result['method']}",
                     fontsize=14, fontweight='bold')
        
        # Распределение силуэтных коэффициентов
        ax2.hist(silhouette_vals, bins=20, edgecolor='black', alpha=0.7)
        ax2.axvline(x=silhouette_avg, color="red", linestyle="--",
                   label=f'Среднее: {silhouette_avg:.3f}')
        ax2.set_xlabel("Значение силуэтного коэффициента", fontsize=12)
        ax2.set_ylabel("Частота", fontsize=12)
        ax2.set_title("Распределение силуэтных коэффициентов",
                     fontsize=14, fontweight='bold')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_cluster_metrics_comparison(self, results_list: List[Dict[str, Any]],
                                       metrics: List[str] = None,
                                       figsize: Optional[tuple] = None) -> plt.Figure:
        """
        Сравнение метрик различных методов кластеризации
        
        Args:
            results_list: Список результатов кластеризации
            metrics: Список метрик для сравнения
            figsize: Размер фигуры
            
        Returns:
            Объект matplotlib Figure
        """
        if metrics is None:
            metrics = ['silhouette_score', 'inertia', 'partition_coefficient']
            
        if figsize is None:
            figsize = (15, 5)
            
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
        
        if n_metrics == 1:
            axes = [axes]
            
        method_names = []
        metric_values = {metric: [] for metric in metrics}
        
        for result in results_list:
            method_names.append(result['method'])
            for metric in metrics:
                if metric in result:
                    metric_values[metric].append(result[metric])
                else:
                    metric_values[metric].append(0)
        
        # Создание графиков для каждой метрики
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            bars = ax.bar(method_names, metric_values[metric],
                         color=cm.get_cmap('tab10')(np.arange(len(method_names))))
            
            # Добавление значений на столбцы
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom')
            
            ax.set_xlabel('Метод кластеризации', fontsize=12)
            ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
            ax.set_title(f'Сравнение {metric}', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Наклон подписей методов
            ax.set_xticklabels(method_names, rotation=45, ha='right')
        
        plt.tight_layout()
        return fig
    
    def plot_cluster_size_distribution(self, result: Dict[str, Any],
                                      figsize: Optional[tuple] = None) -> plt.Figure:
        """
        Визуализация распределения размеров кластеров
        
        Args:
            result: Результаты кластеризации
            figsize: Размер фигуры
            
        Returns:
            Объект matplotlib Figure
        """
        if figsize is None:
            figsize = (10, 6)
            
        labels = result['labels']
        n_clusters = result['n_clusters']
        
        # Подсчет размеров кластеров
        cluster_sizes = np.bincount(labels, minlength=n_clusters)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Столбчатая диаграмма
        colors = cm.get_cmap('tab10')(np.arange(n_clusters) / n_clusters)
        bars = ax1.bar(range(n_clusters), cluster_sizes, color=colors,
                      edgecolor='black')
        
        ax1.set_xlabel('Кластер', fontsize=12)
        ax1.set_ylabel('Количество объектов', fontsize=12)
        ax1.set_title('Распределение размеров кластеров', fontsize=14, fontweight='bold')
        ax1.set_xticks(range(n_clusters))
        
        # Добавление значений на столбцы
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom')
        
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Круговая диаграмма
        if n_clusters > 1:
            wedges, texts, autotexts = ax2.pie(cluster_sizes,
                                              colors=colors,
                                              autopct='%1.1f%%',
                                              startangle=90)
            
            # Улучшение подписей
            for text in texts:
                text.set_fontsize(10)
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            
            ax2.set_title('Процентное распределение', fontsize=14, fontweight='bold')
            ax2.legend(wedges, [f'Кластер {i}' for i in range(n_clusters)],
                      title="Кластеры", loc="center left",
                      bbox_to_anchor=(1, 0, 0.5, 1))
        
        plt.tight_layout()
        return fig
    
    def create_comprehensive_report(self, result: Dict[str, Any],
                                   save_path: Optional[str] = None) -> None:
        """
        Создание комплексного отчета с визуализациями
        
        Args:
            result: Результаты кластеризации
            save_path: Путь для сохранения отчета (если None, показываем графики)
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Создаем все графики
            figs = []
            
            try:
                fig1 = self.plot_clusters_2d(result, reduction_method='pca')
                figs.append(('2D визуализация (PCA)', fig1))
            except Exception as e:
                print(f"Ошибка при создании PCA визуализации: {e}")
            
            try:
                fig2 = self.plot_clusters_2d(result, reduction_method='tsne')
                figs.append(('2D визуализация (t-SNE)', fig2))
            except Exception as e:
                print(f"Ошибка при создании t-SNE визуализации: {e}")
            
            if 'membership_matrix' in result:
                try:
                    fig3 = self.plot_membership_matrix(result)
                    figs.append(('Матрица принадлежности', fig3))
                except Exception as e:
                    print(f"Ошибка при создании матрицы принадлежности: {e}")
            
            if 'covariance_matrices' in result:
                try:
                    fig4 = self.plot_gk_clusters_with_ellipses(result)
                    figs.append(('Кластеры с эллипсами (GK)', fig4))
                except Exception as e:
                    print(f"Ошибка при создании GK визуализации: {e}")
            
            try:
                fig5 = self.plot_silhouette_analysis(result)
                figs.append(('Анализ силуэтов', fig5))
            except Exception as e:
                print(f"Ошибка при создании анализа силуэтов: {e}")
            
            try:
                fig6 = self.plot_cluster_size_distribution(result)
                figs.append(('Распределение размеров кластеров', fig6))
            except Exception as e:
                print(f"Ошибка при создании распределения размеров: {e}")
            
            # Показываем или сохраняем графики
            if save_path:
                import os
                os.makedirs(save_path, exist_ok=True)
                
                for name, fig in figs:
                    filename = f"{name.replace(' ', '_').replace('(', '').replace(')', '')}.png"
                    filepath = os.path.join(save_path, filename)
                    fig.savefig(filepath, dpi=300, bbox_inches='tight')
                    print(f"Сохранено: {filepath}")
                    plt.close(fig)
            else:
                # Показываем все графики по очереди
                for name, fig in figs:
                    fig.suptitle(name, fontsize=16, fontweight='bold', y=1.02)
                    plt.show()
                    plt.close(fig)

# Обновленная функция визуализации
def visualize_clustering_result(result: Dict[str, Any],
                               plot_type: str = 'all',
                               save_path: Optional[str] = None,
                               show_filenames: bool = True,
                               filename_limit: int = None,
                               highlight_files: List[str] = None,
                               **kwargs) -> None:
    """
    Универсальная функция для визуализации результатов кластеризации
    
    Args:
        result: Результаты кластеризации
        plot_type: Тип визуализации ('2d', 'membership', 'silhouette', 
                   'distribution', 'gk_ellipses', 'all', 'interactive', 'table')
        save_path: Путь для сохранения графиков
        show_filenames: Показывать имена файлов на 2D графике
        filename_limit: Максимальное количество имен файлов для отображения
        highlight_files: Список файлов для выделения на графике
        **kwargs: Дополнительные параметры для визуализатора
    """
    visualizer = ClusteringVisualizer(**kwargs)
    
    if plot_type == 'all':
        # Создаем график с именами файлов отдельно
        if show_filenames:
            fig = visualizer.plot_clusters_2d(result, 
                                            reduction_method='pca',
                                            show_filenames=True,
                                            filename_limit=filename_limit,
                                            highlight_files=highlight_files)
            if save_path:
                fig.savefig(f"{save_path}_with_filenames.png", 
                          dpi=300, bbox_inches='tight')
                print(f"График с именами файлов сохранен: {save_path}_with_filenames.png")
            else:
                plt.show()
                plt.close(fig)
        
        # Создаем остальные графики без имен файлов
        visualizer.create_comprehensive_report(result, save_path)
        
    else:
        fig = None
        plot_type = plot_type.lower()
        
        if plot_type in ['2d', '2d_pca']:
            fig = visualizer.plot_clusters_2d(result, 
                                            reduction_method='pca',
                                            show_filenames=show_filenames,
                                            filename_limit=filename_limit,
                                            highlight_files=highlight_files)
        elif plot_type == '2d_tsne':
            fig = visualizer.plot_clusters_2d(result, 
                                            reduction_method='tsne',
                                            show_filenames=show_filenames,
                                            filename_limit=filename_limit,
                                            highlight_files=highlight_files)
        elif plot_type in ['membership', 'fuzzy']:
            fig = visualizer.plot_membership_matrix(result)
        elif plot_type in ['silhouette', 'silhouette_analysis']:
            fig = visualizer.plot_silhouette_analysis(result)
        elif plot_type in ['distribution', 'cluster_sizes']:
            fig = visualizer.plot_cluster_size_distribution(result)
        elif plot_type in ['gk', 'ellipses', 'gustafson_kessel']:
            fig = visualizer.plot_gk_clusters_with_ellipses(result)
        elif plot_type == 'interactive':
            visualizer.plot_interactive_clusters(result, save_path=save_path)
            return
        elif plot_type == 'table':
            fig = visualizer.plot_cluster_with_file_table(result)
        else:
            raise ValueError(f"Неизвестный тип визуализации: {plot_type}")
        
        if fig:
            if save_path:
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"График сохранен: {save_path}")
                plt.close(fig)
            else:
                plt.show()

# def visualize_clustering_result(result: Dict[str, Any],
#                                plot_type: str = 'all',
#                                save_path: Optional[str] = None,
#                                show_filenames: bool = True,
#                                filename_limit: int = None,
#                                highlight_files: List[str] = None,
#                                **kwargs) -> None:
#     """
#     Универсальная функция для визуализации результатов кластеризации
    
#     Args:
#         result: Результаты кластеризации
#         plot_type: Тип визуализации ('2d', 'membership', 'silhouette', 
#                    'distribution', 'gk_ellipses', 'all')
#         save_path: Путь для сохранения графиков
#         **kwargs: Дополнительные параметры для визуализатора
#     """
#     visualizer = ClusteringVisualizer(**kwargs)
    
#     if plot_type == 'all':
#         visualizer.create_comprehensive_report(result, save_path)
#     else:
#         fig = None
#         plot_type = plot_type.lower()
        
#         if plot_type in ['2d', '2d_pca']:
#             fig = visualizer.plot_clusters_2d(result, reduction_method='pca')
#         elif plot_type == '2d_tsne':
#             fig = visualizer.plot_clusters_2d(result, reduction_method='tsne')
#         elif plot_type in ['membership', 'fuzzy']:
#             fig = visualizer.plot_membership_matrix(result)
#         elif plot_type in ['silhouette', 'silhouette_analysis']:
#             fig = visualizer.plot_silhouette_analysis(result)
#         elif plot_type in ['distribution', 'cluster_sizes']:
#             fig = visualizer.plot_cluster_size_distribution(result)
#         elif plot_type in ['gk', 'ellipses', 'gustafson_kessel']:
#             fig = visualizer.plot_gk_clusters_with_ellipses(result)
#         elif plot_type == 'table':
#             fig = visualizer.plot_cluster_with_file_table(result)
#         elif plot_type == 'interactive':
#             visualizer.plot_interactive_clusters(result, save_path=save_path)
#             return
#         else:
#             raise ValueError(f"Неизвестный тип визуализации: {plot_type}")
        
#         if fig:
#             if save_path:
#                 fig.savefig(save_path, dpi=300, bbox_inches='tight')
#                 print(f"График сохранен: {save_path}")
#                 plt.close(fig)
#             else:
#                 plt.show()


def compare_clustering_methods(results_list: List[Dict[str, Any]],
                              save_path: Optional[str] = None,
                              **kwargs) -> None:
    """
    Сравнение нескольких методов кластеризации
    
    Args:
        results_list: Список результатов кластеризации разных методов
        save_path: Путь для сохранения графика сравнения
        **kwargs: Дополнительные параметры
    """
    visualizer = ClusteringVisualizer(**kwargs)
    
    fig = visualizer.plot_cluster_metrics_comparison(results_list)
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"График сравнения сохранен: {save_path}")
        plt.close(fig)
    else:
        plt.show()


# Пример использования визуализатора
if __name__ == "__main__":
    # Импортируем функции кластеризации из предыдущего кода
    # (предполагаем, что они доступны в том же модуле)
    from cluster import cluster_files
    
    # Создаем тестовые данные
    np.random.seed(42)
    n_files = 50
    vector_dim = 10
    
    vectors_dict = {
        f'file_{i}.txt': np.random.randn(vector_dim) for i in range(n_files)
    }
    
    # Добавляем структуру
    for i in range(n_files):
        if i < 15:
            vectors_dict[f'file_{i}.txt'] += np.array([3, 3] + [0] * (vector_dim - 2))
        elif i < 35:
            vectors_dict[f'file_{i}.txt'] += np.array([-2, -2] + [0] * (vector_dim - 2))
        else:
            vectors_dict[f'file_{i}.txt'] += np.array([0, 0, 2, -2] + [0] * (vector_dim - 4))
    
    # Выполняем кластеризацию разными методами
    print("Выполняем кластеризацию...")
    
    # K-means
    result_kmeans = cluster_files(vectors_dict, method='kmeans', n_clusters=3)
    
    # Fuzzy C-means
    result_fcm = cluster_files(vectors_dict, method='fcm', n_clusters=3, m=2.0)
    
    # Gustafson-Kessel
    result_gk = cluster_files(vectors_dict, method='gk', n_clusters=3, m=2.0)
    
    # Пример 1: Визуализация результатов K-means
    print("\n1. Визуализация K-means кластеризации:")
    visualize_clustering_result(result_kmeans, plot_type='2d_pca')
    
    # Пример 2: Матрица принадлежности для Fuzzy C-means
    print("\n2. Матрица принадлежности Fuzzy C-means:")
    visualize_clustering_result(result_fcm, plot_type='membership')
    
    # Пример 3: Анализ силуэтов
    print("\n3. Анализ силуэтов для K-means:")
    visualize_clustering_result(result_kmeans, plot_type='silhouette')
    
    # Пример 4: GK кластеризация с эллипсами
    print("\n4. Gustafson-Kessel кластеризация с эллипсами:")
    visualize_clustering_result(result_gk, plot_type='gk')
    
    # Пример 5: Распределение размеров кластеров
    print("\n5. Распределение размеров кластеров (K-means):")
    visualize_clustering_result(result_kmeans, plot_type='distribution')
    
    # Пример 6: Сравнение всех методов
    print("\n6. Сравнение метрик всех методов кластеризации:")
    compare_clustering_methods([result_kmeans, result_fcm, result_gk])
    
    # Пример 7: Полный отчет
    print("\n7. Создание полного отчета для K-means:")
    visualizer = ClusteringVisualizer()
    visualizer.create_comprehensive_report(result_kmeans)
    
    # Пример 8: Сохранение графиков в файл
    print("\n8. Сохранение графиков в файлы...")
    visualize_clustering_result(result_kmeans, plot_type='2d_pca', 
                              save_path='kmeans_2d_pca.png')
    visualize_clustering_result(result_fcm, plot_type='membership',
                              save_path='fcm_membership.png')