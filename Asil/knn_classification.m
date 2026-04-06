clear; clc; close all;

% Configuración para figuras de calidad académica
set(0, 'DefaultAxesFontSize', 12);
set(0, 'DefaultAxesFontName', 'Times New Roman');
set(0, 'DefaultTextFontSize', 12);
set(0, 'DefaultTextFontName', 'Times New Roman');
set(0, 'DefaultLineLineWidth', 1.5);
set(0, 'DefaultAxesGridLineStyle', ':');
set(0, 'DefaultAxesXGrid', 'on');
set(0, 'DefaultAxesYGrid', 'on');
set(0, 'DefaultAxesGridColor', [0.5 0.5 0.5]);
set(0, 'DefaultAxesGridAlpha', 0.3);
set(0, 'DefaultAxesXColor', [0 0 0]);
set(0, 'DefaultAxesYColor', [0 0 0]);
set(0, 'DefaultTextColor', [0 0 0]);
set(0, 'DefaultAxesTitleFontWeight', 'bold');

rng(42);

resultsDir = 'results';
figuresDir = fullfile(resultsDir, 'figures');
mkdir(resultsDir);
mkdir(figuresDir);

trainRatio = 0.8;
nSamples = 10000;
nTrain = round(nSamples * trainRatio);
nTest = nSamples - nTrain;

fprintf('=== Cargando Datos ===\n');
load('digits.mat');

X = double(digits.image);
Y = digits.label';

fprintf('Conjunto de datos: %d muestras, %d características, %d clases\n', ...
    size(X, 2), size(X, 1), length(unique(Y)));

X = (X - min(X, [], 'all')) / (max(X, [], 'all') - min(X, [], 'all'));

perm = randperm(nSamples);
trainIdx = perm(1:nTrain);
testIdx = perm(nTrain+1:end);

X_train = X(:, trainIdx)';
Y_train = Y(trainIdx);
X_test = X(:, testIdx)';
Y_test = Y(testIdx);

fprintf('Conjunto de entrenamiento: %d muestras\n', nTrain);
fprintf('Conjunto de prueba: %d muestras\n', nTest);

fprintf('\n=== Configuración de Parámetros ===\n');

K_neighbors = [1, 3, 5, 7, 9, 11, 15, 20, 25, 30];

PCA_dims = [10, 15, 20, 30, 50, 75, 100, 150, 200, 300];

distanceMetrics = {'euclidean', 'cosine', 'cityblock', 'chebychev', 'correlation'};

weightingMethods = {'equal', 'squaredinverse', 'inverse'};

fprintf('Vecinos K a probar: %s\n', mat2str(K_neighbors));
fprintf('Dimensiones PCA a probar: %s\n', mat2str(PCA_dims));
fprintf('Métricas de distancia: %s\n', strjoin(distanceMetrics, ', '));
fprintf('Métodos de ponderación: %s\n', strjoin(weightingMethods, ', '));

results = struct();
results.K_neighbors = K_neighbors;
results.PCA_dims = PCA_dims;
results.distanceMetrics = distanceMetrics;
results.weightingMethods = weightingMethods;
results.accuracy = cell(length(K_neighbors), length(PCA_dims), length(distanceMetrics));
results.precision = cell(length(K_neighbors), length(PCA_dims), length(distanceMetrics));
results.recall = cell(length(K_neighbors), length(PCA_dims), length(distanceMetrics));
results.f1 = cell(length(K_neighbors), length(PCA_dims), length(distanceMetrics));
results.time = cell(length(K_neighbors), length(PCA_dims), length(distanceMetrics));
results.pcaVariance = zeros(length(PCA_dims), 1);
results.bestConfig = [];

fprintf('\n=== Computando PCA ===\n');
[coeff, score, latent, tsquared, explained] = pca(X_train);

for i = 1:length(PCA_dims)
    pd = PCA_dims(i);
    results.pcaVariance(i) = sum(explained(1:pd));
end

fprintf('Varianza explicada por PCA:\n');
for i = 1:length(PCA_dims)
    fprintf('  %d componentes: %.2f%%\n', PCA_dims(i), results.pcaVariance(i));
end

fprintf('\n=== Ejecutando Experimentos k-NN ===\n');

overallBestAccuracy = 0;
overallBestConfig = struct();

results.accuracy = cell(length(K_neighbors), length(PCA_dims), length(distanceMetrics), length(weightingMethods));
results.precision = cell(length(K_neighbors), length(PCA_dims), length(distanceMetrics), length(weightingMethods));
results.recall = cell(length(K_neighbors), length(PCA_dims), length(distanceMetrics), length(weightingMethods));
results.f1 = cell(length(K_neighbors), length(PCA_dims), length(distanceMetrics), length(weightingMethods));
results.time = cell(length(K_neighbors), length(PCA_dims), length(distanceMetrics), length(weightingMethods));

for wIdx = 1:length(weightingMethods)
    weightingMethod = weightingMethods{wIdx};
    fprintf('\n=== Ponderación: %s ===\n', weightingMethod);

    for pIdx = 1:length(PCA_dims)
        pd = PCA_dims(pIdx);
        fprintf('\n--- PCA dimensions = %d (%.1f%% variance) ---\n', pd, results.pcaVariance(pIdx));

        X_train_pca = score(:, 1:pd);
        X_test_pca = (X_test - mean(X_train)) * coeff(:, 1:pd);

        for dIdx = 1:length(distanceMetrics)
            dist = distanceMetrics{dIdx};
            fprintf('  Distancia: %s\n', dist);

            for kIdx = 1:length(K_neighbors)
                K = K_neighbors(kIdx);
                fprintf('    K=%d... ', K);

                tic;
                try
                    knnModel = fitcknn(X_train_pca, Y_train, ...
                        'NumNeighbors', K, ...
                        'Distance', dist, ...
                        'DistanceWeight', weightingMethod);

                    preds = predict(knnModel, X_test_pca);
                    accuracy = mean(preds == Y_test) * 100;

                    classes = unique(Y_test);
                    precision = zeros(length(classes), 1);
                    recall = zeros(length(classes), 1);
                    f1 = zeros(length(classes), 1);

                    for c = 1:length(classes)
                        classIdx = classes(c);
                        tp = sum(preds(Y_test == classIdx) == classIdx);
                        fp = sum(preds(Y_test ~= classIdx) == classIdx);
                        fn = sum(preds(Y_test == classIdx) ~= classIdx);

                        if tp + fp > 0
                            precision(c) = tp / (tp + fp);
                        end
                        if tp + fn > 0
                            recall(c) = tp / (tp + fn);
                        end
                        if precision(c) + recall(c) > 0
                            f1(c) = 2 * precision(c) * recall(c) / (precision(c) + recall(c));
                        end
                    end

                    avgPrecision = mean(precision);
                    avgRecall = mean(recall);
                    avgF1 = mean(f1);

                    elapsed = toc;

                    fprintf('Prec: %.2f%%, F1: %.3f, Tiempo: %.2fs\n', accuracy, avgF1, elapsed);

                    results.accuracy{kIdx, pIdx, dIdx, wIdx} = accuracy;
                    results.precision{kIdx, pIdx, dIdx, wIdx} = avgPrecision;
                    results.recall{kIdx, pIdx, dIdx, wIdx} = avgRecall;
                    results.f1{kIdx, pIdx, dIdx, wIdx} = avgF1;
                    results.time{kIdx, pIdx, dIdx, wIdx} = elapsed;

                    if accuracy > overallBestAccuracy
                        overallBestAccuracy = accuracy;
                        overallBestConfig.K = K;
                        overallBestConfig.PCA_dim = pd;
                        overallBestConfig.distance = dist;
                        overallBestConfig.weighting = weightingMethod;
                        overallBestConfig.accuracy = accuracy;
                        overallBestConfig.model = knnModel;
                        overallBestConfig.preds = preds;
                    end

                catch ME
                    fprintf('FALLÓ: %s\n', ME.message);
                    elapsed = toc;
                    results.time{kIdx, pIdx, dIdx, wIdx} = elapsed;
                end
            end
        end
    end
end

fprintf('\n=== Mejor Configuración ===\n');
if isfield(overallBestConfig, 'weighting')
    fprintf('K=%d, PCA=%d, Distancia=%s, Ponderación=%s, Precisión=%.2f%%\n', ...
        overallBestConfig.K, overallBestConfig.PCA_dim, ...
        overallBestConfig.distance, overallBestConfig.weighting, overallBestConfig.accuracy);
else
    fprintf('K=%d, PCA=%d, Distancia=%s, Precisión=%.2f%%\n', ...
        overallBestConfig.K, overallBestConfig.PCA_dim, ...
        overallBestConfig.distance, overallBestConfig.accuracy);
end

results.bestConfig = overallBestConfig;

fprintf('\n=== Generando Gráficos ===\n');

bestPCAIdx = find(PCA_dims == overallBestConfig.PCA_dim);
if isempty(bestPCAIdx)
    bestPCAIdx = 1;
end
bestWeightingIdx = find(strcmp(weightingMethods, overallBestConfig.weighting));
if isempty(bestWeightingIdx)
    bestWeightingIdx = 1;
end

fig = figure('Color', 'white', 'Position', [100, 100, 900, 600]);
ax = axes('Parent', fig);
hold(ax, 'on');
colors = lines(length(distanceMetrics));
for dIdx = 1:length(distanceMetrics)
    accVals = zeros(length(K_neighbors), 1);
    for kIdx = 1:length(K_neighbors)
        if ~isempty(results.accuracy{kIdx, bestPCAIdx, dIdx, bestWeightingIdx})
            accVals(kIdx) = results.accuracy{kIdx, bestPCAIdx, dIdx, bestWeightingIdx};
        else
            accVals(kIdx) = NaN;
        end
    end
    plot(ax, K_neighbors, accVals, '-o', 'Color', colors(dIdx,:), ...
        'LineWidth', 1.5, 'MarkerSize', 7, 'MarkerFaceColor', colors(dIdx,:), ...
        'DisplayName', distanceMetrics{dIdx});
end
xlabel(ax, 'K (Número de Vecinos)', 'FontSize', 13, 'FontWeight', 'bold');
ylabel(ax, 'Precisión de Clasificación (%)', 'FontSize', 13, 'FontWeight', 'bold');
title(ax, sprintf('k-NN: Precisión vs K (PCA=%d, Ponderación=%s)', overallBestConfig.PCA_dim, overallBestConfig.weighting), ...
    'FontSize', 14, 'FontWeight', 'bold');
legend(ax, 'Location', 'best', 'FontSize', 11);
grid(ax, 'on');
set(ax, 'Color', 'white');
set(ax, 'Box', 'on');
set(ax, 'LineWidth', 1);
set(ax, 'XColor', [0 0 0]);
set(ax, 'YColor', [0 0 0]);
set(ax, 'GridColor', [0.5 0.5 0.5]);
set(ax, 'MinorGridColor', [0.7 0.7 0.7]);
xlim(ax, [min(K_neighbors)-1, max(K_neighbors)+1]);
saveas(fig, fullfile(figuresDir, 'knn_precision_vs_k.png'));
fprintf('  Guardado: precision_vs_k.png\n');

bestKIdx = find(K_neighbors == overallBestConfig.K);
if isempty(bestKIdx)
    bestKIdx = 1;
end

fig = figure('Color', 'white', 'Position', [100, 100, 900, 600]);
ax = axes('Parent', fig);
hold(ax, 'on');
for dIdx = 1:length(distanceMetrics)
    accVals = zeros(length(PCA_dims), 1);
    for pIdx = 1:length(PCA_dims)
        if ~isempty(results.accuracy{bestKIdx, pIdx, dIdx, bestWeightingIdx})
            accVals(pIdx) = results.accuracy{bestKIdx, pIdx, dIdx, bestWeightingIdx};
        else
            accVals(pIdx) = NaN;
        end
    end
    plot(ax, PCA_dims, accVals, '-s', 'Color', colors(dIdx,:), ...
        'LineWidth', 1.5, 'MarkerSize', 7, 'MarkerFaceColor', colors(dIdx,:), ...
        'DisplayName', distanceMetrics{dIdx});
end
xlabel(ax, 'Dimensiones PCA', 'FontSize', 13, 'FontWeight', 'bold');
ylabel(ax, 'Precisión de Clasificación (%)', 'FontSize', 13, 'FontWeight', 'bold');
title(ax, sprintf('k-NN: Precisión vs Dimensiones PCA (K=%d, Ponderación=%s)', overallBestConfig.K, overallBestConfig.weighting), ...
    'FontSize', 14, 'FontWeight', 'bold');
legend(ax, 'Location', 'best', 'FontSize', 11);
grid(ax, 'on');
set(ax, 'Color', 'white');
set(ax, 'Box', 'on');
set(ax, 'LineWidth', 1);
set(ax, 'XColor', [0 0 0]);
set(ax, 'YColor', [0 0 0]);
set(ax, 'GridColor', [0.5 0.5 0.5]);
set(ax, 'MinorGridColor', [0.7 0.7 0.7]);
saveas(fig, fullfile(figuresDir, 'knn_precision_vs_pca.png'));
fprintf('  Guardado: precision_vs_pca.png\n');

fig = figure('Color', 'white', 'Position', [100, 100, 800, 600]);
ax = axes('Parent', fig);
hold(ax, 'on');
plot(ax, PCA_dims, results.pcaVariance, '-o', 'Color', [0.2 0.4 0.8], ...
    'LineWidth', 1.5, 'MarkerSize', 8, 'MarkerFaceColor', [0.2 0.4 0.8]);
xlabel(ax, 'Dimensiones PCA', 'FontSize', 13, 'FontWeight', 'bold');
ylabel(ax, 'Varianza Explicada (%)', 'FontSize', 13, 'FontWeight', 'bold');
title(ax, 'PCA: Varianza Explicada vs Número de Componentes', 'FontSize', 14, 'FontWeight', 'bold');
grid(ax, 'on');
yline(ax, 95, '--r', '95%', 'LineWidth', 1.5);
yline(ax, 99, '--g', '99%', 'LineWidth', 1.5);
set(ax, 'Color', 'white');
set(ax, 'Box', 'on');
set(ax, 'LineWidth', 1);
set(ax, 'XColor', [0 0 0]);
set(ax, 'YColor', [0 0 0]);
set(ax, 'GridColor', [0.5 0.5 0.5]);
set(ax, 'MinorGridColor', [0.7 0.7 0.7]);
saveas(fig, fullfile(figuresDir, 'pca_varianza_explicada.png'));
fprintf('  Guardado: varianza_explicada_pca.png\n');

bestDistIdx = find(strcmp(distanceMetrics, overallBestConfig.distance));
if isempty(bestDistIdx)
    bestDistIdx = 1;
end

fig = figure('Color', 'white', 'Position', [100, 100, 1000, 700]);
ax = axes('Parent', fig);
accMatrix = zeros(length(K_neighbors), length(PCA_dims));
for kIdx = 1:length(K_neighbors)
    for pIdx = 1:length(PCA_dims)
        if ~isempty(results.accuracy{kIdx, pIdx, bestDistIdx, bestWeightingIdx})
            accMatrix(kIdx, pIdx) = results.accuracy{kIdx, pIdx, bestDistIdx, bestWeightingIdx};
        end
    end
end
imagesc(ax, accMatrix);
c = colorbar(ax);
c.Label.String = 'Precisión (%)';
c.Label.FontSize = 12;
c.Label.FontWeight = 'bold';
xlabel(ax, 'Dimensiones PCA', 'FontSize', 13, 'FontWeight', 'bold');
ylabel(ax, 'K (Número de Vecinos)', 'FontSize', 13, 'FontWeight', 'bold');
title(ax, sprintf('Mapa de Calor Precisión k-NN (Distancia: %s, Ponderación: %s)', distanceMetrics{bestDistIdx}, overallBestConfig.weighting), ...
    'FontSize', 14, 'FontWeight', 'bold');
xticks(ax, 1:length(PCA_dims));
xticklabels(ax, string(PCA_dims));
xtickangle(ax, 45);
yticks(ax, 1:length(K_neighbors));
yticklabels(ax, string(K_neighbors));
set(ax, 'Color', 'white');
set(ax, 'Box', 'on');
set(ax, 'LineWidth', 1);
set(ax, 'XColor', [0 0 0]);
set(ax, 'YColor', [0 0 0]);
set(ax, 'GridColor', [0.5 0.5 0.5]);
set(ax, 'MinorGridColor', [0.7 0.7 0.7]);
for kIdx = 1:length(K_neighbors)
    for pIdx = 1:length(PCA_dims)
        if ~isnan(accMatrix(kIdx, pIdx))
            text(ax, pIdx, kIdx, sprintf('%.1f', accMatrix(kIdx, pIdx)), ...
                'HorizontalAlignment', 'center', 'Color', 'k', 'FontSize', 8);
        end
    end
end
saveas(fig, fullfile(figuresDir, 'knn_precision_mapa_calor.png'));
fprintf('  Guardado: precision_mapa_calor.png\n');

if ~isempty(overallBestConfig.preds)
    fig = figure('Color', 'white', 'Position', [100, 100, 900, 750]);
    cm = confusionchart(Y_test, overallBestConfig.preds, ...
        'Title', '', ...
        'ColumnSummary', 'column-normalized', ...
        'RowSummary', 'row-normalized');
    cm.FontSize = 12;
    cm.FontName = 'Times New Roman';
    % Use a proper grayscale colormap for academic papers
    colormap(flipud(gray(256)));
    % Add title using annotation text box
    annotation('textbox', [0.3 0.92 0.4 0.08], 'String', ...
        sprintf('Matriz de Confusión (Mejor: K=%d, PCA=%d, %s)', ...
        overallBestConfig.K, overallBestConfig.PCA_dim, overallBestConfig.distance), ...
        'FontSize', 14, 'FontWeight', 'bold', 'Color', [0 0 0], ...
        'EdgeColor', 'none', 'HorizontalAlignment', 'center');
    saveas(fig, fullfile(figuresDir, 'knn_matriz_confusion.png'));
    fprintf('  Guardado: matriz_confusion.png\n');
end

fig = figure('Color', 'white', 'Position', [100, 100, 800, 600]);
ax = axes('Parent', fig);
hold(ax, 'on');
for dIdx = 1:length(distanceMetrics)
    timeVals = zeros(length(K_neighbors), 1);
    for kIdx = 1:length(K_neighbors)
        if ~isempty(results.time{kIdx, bestPCAIdx, dIdx, bestWeightingIdx})
            timeVals(kIdx) = results.time{kIdx, bestPCAIdx, dIdx, bestWeightingIdx};
        else
            timeVals(kIdx) = NaN;
        end
    end
    plot(ax, K_neighbors, timeVals, '-^', 'Color', colors(dIdx,:), ...
        'LineWidth', 1.5, 'MarkerSize', 7, 'MarkerFaceColor', colors(dIdx,:), ...
        'DisplayName', distanceMetrics{dIdx});
end
xlabel(ax, 'K (Número de Vecinos)', 'FontSize', 13, 'FontWeight', 'bold');
ylabel(ax, 'Tiempo de Cómputo (segundos)', 'FontSize', 13, 'FontWeight', 'bold');
title(ax, sprintf('k-NN: Tiempo de Entrenamiento vs K (PCA=%d, Ponderación=%s)', overallBestConfig.PCA_dim, overallBestConfig.weighting), ...
    'FontSize', 14, 'FontWeight', 'bold');
legend(ax, 'Location', 'best', 'FontSize', 11);
grid(ax, 'on');
set(ax, 'Color', 'white');
set(ax, 'Box', 'on');
set(ax, 'LineWidth', 1);
set(ax, 'XColor', [0 0 0]);
set(ax, 'YColor', [0 0 0]);
set(ax, 'GridColor', [0.5 0.5 0.5]);
set(ax, 'MinorGridColor', [0.7 0.7 0.7]);
saveas(fig, fullfile(figuresDir, 'knn_tiempo_vs_k.png'));
fprintf('  Guardado: tiempo_vs_k.png\n');

bestK = overallBestConfig.K;
bestPCA = overallBestConfig.PCA_dim;
bestDist = overallBestConfig.distance;

fig = figure('Color', 'white', 'Position', [100, 100, 800, 600]);
ax = axes('Parent', fig);
hold(ax, 'on');
weightingAcc = zeros(length(weightingMethods), 1);
for wIdx = 1:length(weightingMethods)
    if ~isempty(results.accuracy{bestKIdx, bestPCAIdx, bestDistIdx, wIdx})
        weightingAcc(wIdx) = results.accuracy{bestKIdx, bestPCAIdx, bestDistIdx, wIdx};
    else
        weightingAcc(wIdx) = NaN;
    end
end
bar(ax, 1:length(weightingMethods), weightingAcc, 'FaceColor', [0.3 0.5 0.8]);
set(ax, 'XTick', 1:length(weightingMethods), 'XTickLabel', weightingMethods);
xlabel(ax, 'Método de Ponderación', 'FontSize', 13, 'FontWeight', 'bold');
ylabel(ax, 'Precisión de Clasificación (%)', 'FontSize', 13, 'FontWeight', 'bold');
title(ax, sprintf('Comparación de Ponderación de Distancia (K=%d, PCA=%d, Distancia=%s)', bestK, bestPCA, bestDist), ...
    'FontSize', 14, 'FontWeight', 'bold');
grid(ax, 'on');
set(ax, 'Color', 'white');
set(ax, 'Box', 'on');
set(ax, 'LineWidth', 1);
set(ax, 'XColor', [0 0 0]);
set(ax, 'YColor', [0 0 0]);
set(ax, 'GridColor', [0.5 0.5 0.5]);
set(ax, 'MinorGridColor', [0.7 0.7 0.7]);
for i = 1:length(weightingMethods)
    if ~isnan(weightingAcc(i))
        text(ax, i, weightingAcc(i), sprintf('%.2f', weightingAcc(i)), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 10);
    end
end
saveas(fig, fullfile(figuresDir, 'knn_comparacion_ponderacion.png'));
fprintf('  Guardado: comparacion_ponderacion.png\n');

fprintf('\n=== Guardando Resultados ===\n');
save(fullfile(resultsDir, 'knn_metricas.mat'), 'results');
fprintf('Métricas guardadas en: %s\n', fullfile(resultsDir, 'knn_metricas.mat'));

fprintf('\n========================================\n');
fprintf('¡Clasificación k-NN Completada!\n');
fprintf('========================================\n');
fprintf('Mejor configuración:\n');
if isfield(overallBestConfig, 'weighting')
    fprintf('  K = %d, PCA = %d, Distancia = %s, Ponderación = %s\n', ...
        overallBestConfig.K, overallBestConfig.PCA_dim, ...
        overallBestConfig.distance, overallBestConfig.weighting);
else
    fprintf('  K = %d, PCA = %d, Distancia = %s\n', ...
        overallBestConfig.K, overallBestConfig.PCA_dim, overallBestConfig.distance);
end
fprintf('  Precisión = %.2f%%\n', overallBestConfig.accuracy);
fprintf('\nGráficos guardados en: %s\n', figuresDir);
fprintf('========================================\n');
