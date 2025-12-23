function [U, centers, obj_func_history] = gathgeva_cluster(data, cluster_n, options)
% GATHGEVA_CLUSTER - Нечеткая кластеризация методом Gath-Geva
%
% Синтаксис:
%   [U, centers, obj_func_history] = gathgeva_cluster(data, cluster_n, options)
%
% Входные параметры:
%   data     - матрица данных (N x D), N - количество точек, D - размерность
%   cluster_n - количество кластеров
%   options  - структура с параметрами (опционально)
%     options.exp       - параметр нечеткости m (default = 2)
%     options.max_iter  - максимальное число итераций (default = 100)
%     options.min_improve - минимальное улучшение (default = 1e-5)
%     options.verbose   - вывод информации (default = 1)
%     options.init_method - метод инициализации: 'fcm', 'rand', 'kmeans' (default = 'fcm')
%
% Выходные параметры:
%   U        - матрица принадлежностей (N x cluster_n)
%   centers  - центры кластеров (cluster_n x D)
%   obj_func_history - история значений целевой функции

    % Проверка входных параметров
    if nargin < 3
        options = struct();
    end
    
    % Установка параметров по умолчанию
    default_options = struct(...
        'exp', 2, ...           % Параметр нечеткости
        'max_iter', 100, ...    % Максимум итераций
        'min_improve', 1e-5, ... % Минимальное улучшение
        'verbose', 1, ...       % Вывод информации
        'init_method', 'fcm' ... % Метод инициализации
    );
    
    % Объединение с пользовательскими параметрами
    option_names = fieldnames(default_options);
    for i = 1:length(option_names)
        if ~isfield(options, option_names{i})
            options.(option_names{i}) = default_options.(option_names{i});
        end
    end
    
    % Проверка данных
    if size(data, 1) < cluster_n
        error('Количество точек данных должно быть больше количества кластеров');
    end
    
    % Параметры
    m = options.exp;  % Параметр нечеткости
    max_iter = options.max_iter;
    min_improve = options.min_improve;
    verbose = options.verbose;
    
    N = size(data, 1);  % Количество точек
    D = size(data, 2);  % Размерность
    
    % Инициализация
    if verbose
        fprintf('Инициализация кластеризации Gath-Geva...\n');
    end
    
    switch lower(options.init_method)
        case 'fcm'
            % Используем FCM для инициализации
            [init_U, init_centers] = fcm_init(data, cluster_n, m);
        case 'kmeans'
            % Используем k-means для инициализации
            [idx, init_centers] = kmeans(data, cluster_n);
            init_U = zeros(N, cluster_n);
            for i = 1:cluster_n
                init_U(idx == i, i) = 1;
            end
        case 'rand'
            % Случайная инициализация
            init_U = rand(N, cluster_n);
            init_U = init_U ./ sum(init_U, 2);  % Нормализация
            init_centers = (init_U' * data) ./ sum(init_U)';
        otherwise
            error('Неизвестный метод инициализации');
    end
    
    U = init_U;
    centers = init_centers;
    
    % Инициализация истории целевой функции
    obj_func_history = zeros(max_iter, 1);
    
    % Основной цикл алгоритма
    if verbose
        fprintf('Начало кластеризации Gath-Geva...\n');
        fprintf('Итерация\tЦелевая функция\tИзменение\n');
    end
    
    for iter = 1:max_iter
        % Шаг 1: Вычисление ковариационных матриц и априорных вероятностей
        [F, prior_prob] = compute_covariances(data, U, centers, m);
        
        % Шаг 2: Вычисление расстояний Gath-Geva
        D_gg = compute_gg_distance(data, centers, F, prior_prob, D);
        
        % Шаг 3: Обновление матрицы принадлежностей
        U_new = update_membership(D_gg, m);
        
        % Шаг 4: Обновление центров кластеров
        centers_new = update_centers(data, U_new, m);
        
        % Вычисление целевой функции
        obj_func = compute_objective(U_new, D_gg, m);
        obj_func_history(iter) = obj_func;
        
        % Проверка сходимости
        if iter > 1
            improve = abs(obj_func_history(iter-1) - obj_func);
            if verbose
                fprintf('%d\t\t%.6f\t%.6f\n', iter, obj_func, improve);
            end
            if improve < min_improve
                if verbose
                    fprintf('Сходимость достигнута на итерации %d\n', iter);
                end
                obj_func_history = obj_func_history(1:iter);
                break;
            end
        else
            if verbose
                fprintf('%d\t\t%.6f\t-\n', iter, obj_func);
            end
        end
        
        % Обновление значений для следующей итерации
        U = U_new;
        centers = centers_new;
    end
    
    if iter == max_iter && verbose
        fprintf('Достигнуто максимальное число итераций\n');
    end
end

function [U, centers] = fcm_init(data, cluster_n, m)
% FCM_INIT - Инициализация с помощью FCM
    options_fcm = [m, 100, 1e-5, 0];
    [centers, U] = fcm(data, cluster_n, options_fcm);
    U = U';
end

function [F, prior_prob] = compute_covariances(data, U, centers, m)
% COMPUTE_COVARIANCES - Вычисление ковариационных матриц и априорных вероятностей
    
    N = size(data, 1);
    cluster_n = size(centers, 1);
    D = size(data, 2);
    
    F = zeros(D, D, cluster_n);
    prior_prob = zeros(cluster_n, 1);
    
    for j = 1:cluster_n
        % Априорная вероятность
        prior_prob(j) = sum(U(:, j).^m) / N;
        
        % Вычисление ковариационной матрицы
        diff = data - centers(j, :);
        weighted_diff = (U(:, j).^m)' .* diff;
        
        % Ковариационная матрица
        F_j = (diff' * weighted_diff) / sum(U(:, j).^m);
        
        % Регуляризация для обеспечения обратимости
        F_j = F_j + eye(D) * 1e-6;
        
        F(:, :, j) = F_j;
    end
end

function D_gg = compute_gg_distance(data, centers, F, prior_prob, D)
% COMPUTE_GG_DISTANCE - Вычисление расстояний Gath-Geva
    
    N = size(data, 1);
    cluster_n = size(centers, 1);
    D_gg = zeros(N, cluster_n);
    
    for j = 1:cluster_n
        diff = data - centers(j, :);
        
        % Вычисление расстояния Махаланобиса
        inv_F = inv(F(:, :, j));
        det_F = det(F(:, :, j));
        
        % Расстояние Gath-Geva
        for i = 1:N
            mahalanobis_dist = diff(i, :) * inv_F * diff(i, :)';
            D_gg(i, j) = (sqrt(det_F) / prior_prob(j)) * ...
                         exp(0.5 * mahalanobis_dist);
        end
    end
    
    % Защита от нулевых значений
    D_gg(D_gg == 0) = realmin;
end

function U_new = update_membership(D_gg, m)
% UPDATE_MEMBERSHIP - Обновление матрицы принадлежностей
    
    N = size(D_gg, 1);
    cluster_n = size(D_gg, 2);
    
    U_new = zeros(N, cluster_n);
    
    % Вычисление новых принадлежностей
    exp_factor = 2/(m-1);
    
    for i = 1:N
        for j = 1:cluster_n
            sum_terms = 0;
            for k = 1:cluster_n
                sum_terms = sum_terms + (D_gg(i, j) / D_gg(i, k))^exp_factor;
            end
            U_new(i, j) = 1 / sum_terms;
        end
    end
    
    % Нормализация (на всякий случай)
    U_new = U_new ./ sum(U_new, 2);
end

function centers_new = update_centers(data, U, m)
% UPDATE_CENTERS - Обновление центров кластеров
    
    cluster_n = size(U, 2);
    D = size(data, 2);
    centers_new = zeros(cluster_n, D);
    
    for j = 1:cluster_n
        numerator = sum((U(:, j).^m) .* data, 1);
        denominator = sum(U(:, j).^m);
        centers_new(j, :) = numerator / denominator;
    end
end

function obj_value = compute_objective(U, D_gg, m)
% COMPUTE_OBJECTIVE - Вычисление значения целевой функции
    
    N = size(U, 1);
    cluster_n = size(U, 2);
    
    obj_value = 0;
    for i = 1:N
        for j = 1:cluster_n
            obj_value = obj_value + (U(i, j)^m) * (D_gg(i, j)^2);
        end
    end
end