%% Генерация тестовых данных
clear all; close all; clc;

% Создаем три кластера
rng('default'); % Для воспроизводимости
n = 150; % Количество точек в каждом кластере

% Кластер 1
data1 = [randn(n,1)*0.5+2, randn(n,1)*0.5+2];
% Кластер 2
data2 = [randn(n,1)*0.5+5, randn(n,1)*0.5+5];
% Кластер 3
data3 = [randn(n,1)*0.5+8, randn(n,1)*0.5+2];

% Объединяем все данные
data = [data1; data2; data3];

%% Визуализация исходных данных
figure(1)
scatter(data(:,1), data(:,2), 10, 'filled')
title('Исходные данные (без кластеризации)')
xlabel('Признак 1'); ylabel('Признак 2')
grid on

%% Кластеризация FCM
num_clusters = 3; % Количество кластеров
options = fcmOptions(NumClusters=3, DistanceMetric = "fmle");

% Выполняем FCM кластеризацию
[centers, U, obj_func] = fcm(data, options);

%% Присваиваем точки к кластерам
% Находим максимальное значение принадлежности для каждой точки
[maxU, cluster_idx] = max(U);

%% Визуализация результатов
figure(2)

% Создаем цветовую карту для кластеров
colors = {'r', 'g', 'b', 'm', 'c', 'y', 'k'};

% Отображаем точки с цветом соответствующего кластера
for i = 1:num_clusters
    idx = find(cluster_idx == i);
    scatter(data(idx,1), data(idx,2), 20, colors{mod(i-1, length(colors))+1}, 'filled')
    hold on
end

% Отображаем центры кластеров
plot(centers(:,1), centers(:,2), 'kx', 'MarkerSize', 15, 'LineWidth', 3)
plot(centers(:,1), centers(:,2), 'ko', 'MarkerSize', 12, 'LineWidth', 2)

title('Результаты кластеризации FCM')
xlabel('Признак 1'); ylabel('Признак 2')
legend('Кластер 1', 'Кластер 2', 'Кластер 3', 'Центры кластеров')
grid on
hold off

%% Визуализация функции принадлежности
figure(3)
for i = 1:num_clusters
    subplot(num_clusters, 1, i)
    bar(U(i,:))
    title(['Функция принадлежности кластера ', num2str(i)])
    ylabel('Степень принадлежности')
    xlabel('Точка данных')
    ylim([0 1])
end

%% Дополнительная информация
fprintf('\n=== Информация о кластеризации ===\n');
fprintf('Количество кластеров: %d\n', num_clusters);
fprintf('Количество точек данных: %d\n', size(data, 1));
fprintf('Координаты центров кластеров:\n');
for i = 1:num_clusters
    fprintf('  Кластер %d: [%.3f, %.3f]\n', i, centers(i,1), centers(i,2));
end

%% Определение оптимального числа кластеров (опционально)
% Метод локтя (Elbow method) для выбора числа кластеров
max_test_clusters = 6;
obj_func_values = zeros(1, max_test_clusters);

figure(4)
for k = 1:max_test_clusters
    [~, ~, obj_func] = fcm(data, k);
    obj_func_values(k) = obj_func(end);
    plot(obj_func)
    hold on
end

figure(5)
plot(1:max_test_clusters, obj_func_values, '-o', 'LineWidth', 2)
xlabel('Число кластеров')
ylabel('Значение целевой функции')
title('Метод локтя для определения оптимального числа кластеров')
grid on

%% Вычисление средних расстояний (без pdist)
fprintf('\n=== Оценка качества кластеризации ===\n');

% Подготовка переменных для хранения расстояний
intra_distances = cell(1, num_clusters);
inter_distances = cell(num_clusters, num_clusters);
mean_intra = zeros(1, num_clusters);
mean_inter = zeros(num_clusters, num_clusters);

% Функция для вычисления евклидова расстояния между двумя точками
euclidean_dist = @(x, y) sqrt(sum((x - y).^2));

% Вычисляем внутриклассовые расстояния
for i = 1:num_clusters
    % Индексы точек, принадлежащих кластеру i
    idx_i = find(cluster_idx == i);
    
    if length(idx_i) < 2
        fprintf('Кластер %d содержит менее 2 точек\n', i);
        intra_distances{i} = [];
        mean_intra(i) = 0;
        continue;
    end
    
    % Точки кластера i
    points_i = data(idx_i, :);
    n_i = length(idx_i);
    
    % Вычисляем все попарные расстояния внутри кластера (без pdist)
    dist_list_i = [];
    for m = 1:n_i
        for n = m+1:n_i
            dist = euclidean_dist(points_i(m, :), points_i(n, :));
            dist_list_i = [dist_list_i; dist];
        end
    end
    
    intra_distances{i} = dist_list_i;
    
    % Среднее внутриклассовое расстояние для кластера i
    mean_intra(i) = mean(dist_list_i);
    
    fprintf('Кластер %d: среднее внутриклассовое расстояние = %.4f (на основе %d точек)\n', ...
        i, mean_intra(i), n_i);
end

% Вычисляем межклассовые расстояния
for i = 1:num_clusters
    idx_i = find(cluster_idx == i);
    points_i = data(idx_i, :);
    n_i = length(idx_i);
    
    for j = i+1:num_clusters
        idx_j = find(cluster_idx == j);
        points_j = data(idx_j, :);
        n_j = length(idx_j);
        
        % Вычисляем все попарные расстояния между кластерами i и j
        dist_list_ij = [];
        for m = 1:n_i
            for n = 1:n_j
                dist = euclidean_dist(points_i(m, :), points_j(n, :));
                dist_list_ij = [dist_list_ij; dist];
            end
        end
        
        inter_distances{i, j} = dist_list_ij;
        
        % Среднее межклассовое расстояние между кластерами i и j
        mean_inter(i, j) = mean(dist_list_ij);
        mean_inter(j, i) = mean_inter(i, j); % Симметричная матрица
        
        fprintf('Расстояние между кластерами %d и %d = %.4f\n', i, j, mean_inter(i, j));
    end
end

%% Вычисление общих метрик
% Общее среднее внутриклассовое расстояние (по всем кластерам)
overall_mean_intra = mean(mean_intra(mean_intra > 0));

% Общее среднее межклассовое расстояние
valid_inter = [];
for i = 1:num_clusters
    for j = i+1:num_clusters
        if ~isempty(inter_distances{i, j})
            valid_inter = [valid_inter; inter_distances{i, j}];
        end
    end
end

if ~isempty(valid_inter)
    overall_mean_inter = mean(valid_inter);
else
    overall_mean_inter = 0;
end

fprintf('\n--- Сводные метрики ---\n');
fprintf('Общее среднее внутриклассовое расстояние: %.4f\n', overall_mean_intra);
fprintf('Общее среднее межклассовое расстояние: %.4f\n', overall_mean_inter);

% Отношение межклассового к внутриклассовому расстоянию
if overall_mean_intra > 0
    separation_ratio = overall_mean_inter / overall_mean_intra;
    fprintf('Отношение меж/внутриклассового расстояния: %.4f\n', separation_ratio);
    
    if separation_ratio > 1
        fprintf('Интерпретация: кластеры хорошо разделены (отношение > 1)\n');
    else
        fprintf('Интерпретация: кластеры плохо разделены (отношение <= 1)\n');
    end
end

%% Оптимизированный способ вычисления расстояний (более эффективный)
% Альтернативный метод с использованием матричных операций
fprintf('\n--- Альтернативное вычисление (матричный метод) ---\n');

for i = 1:num_clusters
    idx_i = find(cluster_idx == i);
    points_i = data(idx_i, :);
    n_i = length(idx_i);
    
    if n_i < 2
        continue;
    end
    
    % Матричный метод для внутриклассовых расстояний
    % Создаем матрицу квадратов расстояний
    sum_sq_i = sum(points_i.^2, 2);
    dist_sq_matrix = sum_sq_i + sum_sq_i' - 2 * (points_i * points_i');
    
    % Извлекаем верхнюю треугольную матрицу без диагонали
    triu_indices = triu(true(size(dist_sq_matrix)), 1);
    distances_i = sqrt(dist_sq_matrix(triu_indices));
    
    mean_intra_alt = mean(distances_i);
    fprintf('Кластер %d (матричный метод): %.4f\n', i, mean_intra_alt);
end

%% Визуализация распределения расстояний
figure(6)
subplot(1, 2, 1)

% Собираем все внутриклассовые расстояния
all_intra = [];
for i = 1:num_clusters
    if ~isempty(intra_distances{i})
        all_intra = [all_intra; intra_distances{i}];
    end
end

% Гистограмма внутриклассовых расстояний
if ~isempty(all_intra)
    histogram(all_intra, 50, 'FaceColor', 'blue', 'EdgeColor', 'none')
    hold on
    line([overall_mean_intra overall_mean_intra], ylim, 'Color', 'red', ...
        'LineWidth', 2, 'LineStyle', '--')
    xlabel('Расстояние')
    ylabel('Частота')
    title('Распределение внутриклассовых расстояний')
    legend('Распределение', 'Среднее значение')
    grid on
else
    text(0.5, 0.5, 'Нет данных', 'HorizontalAlignment', 'center')
end

subplot(1, 2, 2)
% Гистограмма межклассовых расстояний
if ~isempty(valid_inter)
    histogram(valid_inter, 50, 'FaceColor', 'green', 'EdgeColor', 'none')
    hold on
    line([overall_mean_inter overall_mean_inter], ylim, 'Color', 'red', ...
        'LineWidth', 2, 'LineStyle', '--')
    xlabel('Расстояние')
    ylabel('Частота')
    title('Распределение межклассовых расстояний')
    legend('Распределение', 'Среднее значение')
    grid on
else
    text(0.5, 0.5, 'Нет данных', 'HorizontalAlignment', 'center')
end

sgtitle('Анализ расстояний в кластеризации')

%% Матрица средних межклассовых расстояний
figure(7)
% Создаем полную матрицу для визуализации
vis_matrix = zeros(num_clusters, num_clusters);
for i = 1:num_clusters
    for j = 1:num_clusters
        if i == j
            vis_matrix(i, j) = mean_intra(i);
        elseif i < j && mean_inter(i, j) > 0
            vis_matrix(i, j) = mean_inter(i, j);
            vis_matrix(j, i) = mean_inter(i, j);
        elseif j < i && mean_inter(j, i) > 0
            vis_matrix(i, j) = mean_inter(j, i);
        end
    end
end

imagesc(vis_matrix)
colorbar
title('Матрица средних расстояний (диагональ - внутриклассовые)')
xlabel('Кластер')
ylabel('Кластер')
axis square

% Добавляем значения на матрицу
textStrings = num2str(vis_matrix(:), '%.2f');
textStrings = strtrim(cellstr(textStrings));
[x, y] = meshgrid(1:num_clusters);
hStrings = text(x(:), y(:), textStrings(:), ...
    'HorizontalAlignment', 'center', 'FontSize', 10);
% Устанавливаем цвет текста в зависимости от фона
midValue = mean(vis_matrix(:));
for k = 1:length(hStrings)
    if vis_matrix(k) > midValue
        set(hStrings(k), 'Color', 'white');
    else
        set(hStrings(k), 'Color', 'black');
    end
end

%% Упрощенный индекс Данна
if ~isempty(all_intra) && ~isempty(valid_inter)
    max_intra = max(all_intra);
    min_inter = min(valid_inter);
    
    if max_intra > 0
        dunn_index = min_inter / max_intra;
        fprintf('\nИндекс Данна: %.4f\n', dunn_index);
        fprintf('(чем выше значение, тем лучше разделены кластеры)\n');
    end
end

%% Простая оценка компактности и разделимости
fprintf('\n--- Оценка качества кластеризации ---\n');
fprintf('1. Компактность кластеров (внутриклассовые расстояния):\n');
for i = 1:num_clusters
    if mean_intra(i) > 0
        if mean_intra(i) < 1
            fprintf('   Кластер %d: очень компактный (%.3f)\n', i, mean_intra(i));
        elseif mean_intra(i) < 2
            fprintf('   Кластер %d: умеренно компактный (%.3f)\n', i, mean_intra(i));
        else
            fprintf('   Кластер %d: слабо компактный (%.3f)\n', i, mean_intra(i));
        end
    end
end

fprintf('\n2. Разделимость кластеров (межклассовые расстояния):\n');
for i = 1:num_clusters
    for j = i+1:num_clusters
        if mean_inter(i, j) > 0
            if mean_inter(i, j) / max(mean_intra(i), mean_intra(j)) > 3
                fprintf('   Кластеры %d-%d: отлично разделены (отношение: %.2f)\n', ...
                    i, j, mean_inter(i, j) / max(mean_intra(i), mean_intra(j)));
            elseif mean_inter(i, j) / max(mean_intra(i), mean_intra(j)) > 2
                fprintf('   Кластеры %d-%d: хорошо разделены (отношение: %.2f)\n', ...
                    i, j, mean_inter(i, j) / max(mean_intra(i), mean_intra(j)));
            else
                fprintf('   Кластеры %d-%d: слабо разделены (отношение: %.2f)\n', ...
                    i, j, mean_inter(i, j) / max(mean_intra(i), mean_intra(j)));
            end
        end
    end
end