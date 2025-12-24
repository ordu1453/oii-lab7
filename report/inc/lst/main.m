clear; close all; clc;

filename = 'vectors_wordforms.json'; % файл по умолчанию

json_text = fileread(filename);
data_struct = jsondecode(json_text);

doc_names = fieldnames(data_struct);
num_docs = length(doc_names);

fprintf('Загружено %d документов\n', num_docs);

vectors = [];
for i = 1:num_docs
    doc_name = doc_names{i};
    doc_vector = data_struct.(doc_name);
    
    if i == 1
        vector_length = length(doc_vector);
        vectors = zeros(num_docs, vector_length);
    end
    
    vectors(i, :) = doc_vector;
end
fprintf('Размерность векторов: %d\n', vector_length);

k = input('Введите число кластеров: ');
if isempty(k) || k < 2 || k > num_docs
    k = min(3, floor(num_docs/2));
    fprintf('Используем %d кластеров по умолчанию\n', k);
end

% Определение количества компонент (например, сохраняем 95% дисперсии)
desiredVariance = 0.10;
[coeff, score, latent, ~, explained] = pca(vectors);

% Находим количество компонент, объясняющих desiredVariance дисперсии
cumulativeVariance = cumsum(explained);
kComponents = find(cumulativeVariance >= desiredVariance * 100, 1);

fprintf('Количество главных компонент для сохранения %.1f%% дисперсии: %d\n', ...
    desiredVariance * 100, kComponents);

% Альтернативно: фиксированное количество компонент
% kComponents = min(50, size(vectors, 2)); % например, максимум 50 компонент

% Применяем PCA
vectors_pca = score(:, 1:kComponents);

fprintf('Размерность данных до PCA: %d\n', size(vectors, 2));
fprintf('Размерность данных после PCA: %d\n', kComponents);

% 2. Выполняем FCM кластеризацию на данных после PCA
fprintf('\nВыполняем FCM кластеризацию на данных после PCA...\n');

options = fcmOptions(...
    NumClusters=k, ...
    DistanceMetric="fmle", ...
    Exponent=2, ...
    Verbose=true);

[centers_pca, U, ~] = fcm(vectors_pca, options);

% Присваиваем документы к кластерам
[~, cluster_idx] = max(U);

% % 3. Выполняем FCM кластеризацию
% fprintf('\nВыполняем FCM кластеризацию...\n');
% options = fcmOptions(NumClusters=k, DistanceMetric = "euclidean", Exponent = 10, Verbose = true);
% [centers, U, ~] = fcm(vectors, options);



fprintf('\n=== РЕЗУЛЬТАТЫ КЛАСТЕРИЗАЦИИ ===\n');
fprintf('Всего кластеров: %d\n', k);
fprintf('Всего документов: %d\n\n', num_docs);

for i = 1:k
    cluster_size = sum(cluster_idx == i);
    fprintf('Кластер %d: %d документов (%.1f%%)\n', ...
        i, cluster_size, cluster_size/num_docs*100);
end

fprintf('\nДокументы по кластерам:\n');
for i = 1:k
    fprintf('\n--- Кластер %d ---\n', i);
    idx = find(cluster_idx == i);
    
    num_to_show = min(20, length(idx));
    for j = 1:num_to_show
        fprintf('%s\n', doc_names{idx(j)});
    end
    
    if length(idx) > num_to_show
        fprintf('... и еще %d документов\n', length(idx) - num_to_show);
    end
end

if vector_length == 2
    figure(1);
    colors = lines(k);
    for i = 1:k
        idx = find(cluster_idx == i);
        scatter(vectors(idx, ...
            1), vectors(idx,2), 50, colors(i,:), 'filled');
        hold on;
    end
    plot(centers(:,1), centers(:,2), 'kx', 'MarkerSize', 15, 'LineWidth', 2);
    xlabel('Признак 1'); ylabel('Признак 2');
    title(sprintf('Кластеризация документов (k=%d)', k));
    grid on;
    hold off;
    
elseif vector_length == 3
    figure(1);
    colors = lines(k);
    for i = 1:k
        idx = find(cluster_idx == i);
        scatter3(vectors(idx,1), vectors(idx,2), vectors(idx,3), ...
                50, colors(i,:), 'filled');
        hold on;
    end
    plot3(centers(:,1), centers(:,2), centers(:,3), ...
          'kx', 'MarkerSize', 15, 'LineWidth', 2);
    xlabel('Признак 1'); ylabel('Признак 2'); zlabel('Признак 3');
    title(sprintf('Кластеризация документов (k=%d)', k));
    grid on;
    hold off;
    
else
    fprintf('\nДля визуализации применяем PCA (данные %d-мерные)\n', vector_length);
    [~, score] = pca(vectors);
    
    figure(1);
    colors = lines(k);
    for i = 1:k
        idx = find(cluster_idx == i);
        scatter(score(idx,1), score(idx,2), 50, colors(i,:), 'filled');
        hold on;
    end
    xlabel('Первая главная компонента'); 
    ylabel('Вторая главная компонента');
    % title(sprintf('Кластеризация (PCA проекция, k=%d)', k));
    grid on;
    hold off;
end

fprintf('\nСохраняем результаты...\n');

results_table = table();
results_table.Document = doc_names;
results_table.Cluster = cluster_idx';

for i = 1:k
    results_table.(sprintf('Prob_Cluster_%d', i)) = U(i,:)';
end

% writetable(results_table, 'clustering_results.csv');
% fprintf('Результаты сохранены в clustering_results.csv\n');

fid = fopen('clustering_report.txt', 'w');
fprintf(fid, 'Отчет по кластеризации документов\n');
fprintf(fid, '==================================\n');
fprintf(fid, 'Дата: %s\n', datestr(now));
fprintf(fid, 'Файл данных: %s\n', filename);
fprintf(fid, 'Число документов: %d\n', num_docs);
fprintf(fid, 'Число кластеров: %d\n\n', k);

for i = 1:k
    cluster_size = sum(cluster_idx == i);
    fprintf(fid, 'Кластер %d (%d документов, %.1f%%):\n', ...
        i, cluster_size, cluster_size/num_docs*100);
    
    idx = find(cluster_idx == i);
    for j = 1:min(10, length(idx))
        fprintf(fid, '  - %s\n', doc_names{idx(j)});
    end
    if length(idx) > 10
        fprintf(fid, '  ... и еще %d документов\n', length(idx)-10);
    end
    fprintf(fid, '\n');
end
fclose(fid);
fprintf('Отчет сохранен в clustering_report.txt\n');

fprintf('\nГотово!\n');