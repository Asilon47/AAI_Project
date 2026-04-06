
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

X_train = X(:, trainIdx);
Y_train = Y(trainIdx);
X_test = X(:, testIdx);
Y_test = Y(testIdx);

fprintf('Conjunto de entrenamiento: %d muestras\n', nTrain);
fprintf('Conjunto de prueba: %d muestras\n', nTest);

fprintf('\n=== Configuración de Parámetros ===\n');

K_values = [5, 10, 15, 20, 30, 50, 75, 100];

distanceMetrics = { 'cosine', 'correlation'};

initMethods = {'plus', 'sample', 'cluster'};

replicates = 5;

fprintf('Valores de K a probar: %s\n', mat2str(K_values));
fprintf('Métricas de distancia: %s\n', strjoin(distanceMetrics, ', '));
fprintf('Métodos de inicialización: %s\n', strjoin(initMethods, ', '));

results = struct();
results.K_values = K_values;
results.distanceMetrics = distanceMetrics;
results.initMethods = initMethods;
results.accuracy = cell(length(K_values), length(distanceMetrics));
results.nmi = cell(length(K_values), length(distanceMetrics));
results.ari = cell(length(K_values), length(distanceMetrics));
results.inertia = cell(length(K_values), length(distanceMetrics));
results.time = cell(length(K_values), length(distanceMetrics));
results.centroids = cell(length(K_values), length(distanceMetrics));
results.bestConfig = [];

fprintf('\n=== Ejecutando Experimentos K-means ===\n');

overallBestAccuracy = 0;
overallBestConfig = struct();

for kIdx = 1:length(K_values)
    K = K_values(kIdx);
    fprintf('\n--- K = %d ---\n', K);

    for dIdx = 1:length(distanceMetrics)
        dist = distanceMetrics{dIdx};
        fprintf('  Distancia: %s\n', dist);

        bestInitAccuracy = 0;
        bestInit = initMethods{1};
        bestModel = [];

        for iIdx = 1:length(initMethods)
            init = initMethods{iIdx};

            if K > 50
                nReps = 10;
            else
                nReps = replicates;
            end

            tic;
            try
                [idx, C, sumd, D] = kmeans(X_train', K, ...
                    'Distance', dist, ...
                    'Replicates', nReps, ...
                    'Start', init, ...
                    'MaxIter', 500, ...
                    'Display', 'off');
                elapsed = toc;

                D_train = pdist2(X_train', C, dist);
                D_test = pdist2(X_test', C, dist);

                classifier = fitcecoc(D_train, Y_train);
                preds = predict(classifier, D_test);
                accuracy = mean(preds == Y_test) * 100;

                nmi = evalClusteringNMI(idx, Y_train);
                ari = evalClusteringARI(idx, Y_train);
                inertia_val = sum(sumd);

                if accuracy > bestInitAccuracy
                    bestInitAccuracy = accuracy;
                    bestInit = init;
                    bestModel.D = D_test;
                    bestModel.C = C;
                    bestModel.idx = idx;
                    bestModel.accuracy = accuracy;
                    bestModel.nmi = nmi;
                    bestModel.ari = ari;
                    bestModel.inertia = inertia_val;
                    bestModel.time = elapsed;
                end

                fprintf('    Inicialización: %-10s Prec: %.2f%% NMI: %.3f ARI: %.3f Tiempo: %.2fs\n', ...
                    init, accuracy, nmi, ari, elapsed);

            catch ME
                fprintf('    Inicialización: %-10s FALLÓ: %s\n', init, ME.message);
                continue;
            end
        end

        if ~isempty(bestModel)
            results.accuracy{kIdx, dIdx} = bestModel.accuracy;
            results.nmi{kIdx, dIdx} = bestModel.nmi;
            results.ari{kIdx, dIdx} = bestModel.ari;
            results.inertia{kIdx, dIdx} = bestModel.inertia;
            results.time{kIdx, dIdx} = bestModel.time;
            results.centroids{kIdx, dIdx} = bestModel.C;

            if bestModel.accuracy > overallBestAccuracy
                overallBestAccuracy = bestModel.accuracy;
                overallBestConfig.K = K;
                overallBestConfig.distance = dist;
                overallBestConfig.accuracy = bestModel.accuracy;
                overallBestConfig.D_test = bestModel.D;
                overallBestConfig.C = bestModel.C;
            end
        else
            fprintf('  Advertencia: Sin inicialización exitosa para K=%d, dist=%s\n', K, dist);
        end
    end
end

fprintf('\n=== Mejor Configuración ===\n');
fprintf('K = %d, Distancia = %s, Precisión = %.2f%%\n', ...
    overallBestConfig.K, overallBestConfig.distance, overallBestConfig.accuracy);

results.bestConfig = overallBestConfig;

fprintf('\n=== Generando Gráficos ===\n');

fig = figure('Color', 'white', 'Position', [100, 100, 800, 600]);
ax = axes('Parent', fig);
hold(ax, 'on');
colors = lines(length(distanceMetrics));
for dIdx = 1:length(distanceMetrics)
    accVals = zeros(length(K_values), 1);
    for kIdx = 1:length(K_values)
        if ~isempty(results.accuracy{kIdx, dIdx})
            accVals(kIdx) = results.accuracy{kIdx, dIdx};
        else
            accVals(kIdx) = NaN;
        end
    end
    plot(ax, K_values, accVals, '-o', 'Color', colors(dIdx,:), ...
        'LineWidth', 1.5, 'MarkerSize', 8, 'MarkerFaceColor', colors(dIdx,:));
end
xlabel(ax, 'Número de Clústeres (K)', 'FontSize', 13, 'FontWeight', 'bold');
ylabel(ax, 'Precisión de Clasificación (%)', 'FontSize', 13, 'FontWeight', 'bold');
title(ax, 'K-means: Precisión vs Número de Clústeres', 'FontSize', 14, 'FontWeight', 'bold');
legend(ax, distanceMetrics, 'Location', 'best', 'FontSize', 11);
grid(ax, 'on');
set(ax, 'Color', 'white');
set(ax, 'Box', 'on');
set(ax, 'LineWidth', 1);
set(ax, 'XColor', [0 0 0]);
set(ax, 'YColor', [0 0 0]);
set(ax, 'GridColor', [0.5 0.5 0.5]);
set(ax, 'MinorGridColor', [0.7 0.7 0.7]);
xlim(ax, [min(K_values)-5, max(K_values)+5]);
ylim(ax, [min(accVals)-5, max(accVals)+5]);
saveas(fig, fullfile(figuresDir, 'kmeans_precision_vs_k.png'));
fprintf('  Guardado: precision_vs_k.png\n');

fig = figure('Color', 'white', 'Position', [100, 100, 800, 600]);
ax = axes('Parent', fig);
hold(ax, 'on');
inertiaVals = zeros(length(K_values), 1);
for kIdx = 1:length(K_values)
    if ~isempty(results.inertia{kIdx, 1})
        inertiaVals(kIdx) = results.inertia{kIdx, 1};
    else
        inertiaVals(kIdx) = NaN;
    end
end
plot(ax, K_values, inertiaVals, '-o', 'Color', [0.85 0.33 0.1], ...
    'LineWidth', 1.5, 'MarkerSize', 8, 'MarkerFaceColor', [0.85 0.33 0.1]);
xlabel(ax, 'Número de Clústeres (K)', 'FontSize', 13, 'FontWeight', 'bold');
ylabel(ax, 'Inercia (Suma de Cuadrados Intra-clúster)', 'FontSize', 13, 'FontWeight', 'bold');
title(ax, 'K-means: Inercia vs K', 'FontSize', 14, 'FontWeight', 'bold');
grid(ax, 'on');
set(ax, 'Color', 'white');
set(ax, 'Box', 'on');
set(ax, 'LineWidth', 1);
set(ax, 'XColor', [0 0 0]);
set(ax, 'YColor', [0 0 0]);
set(ax, 'GridColor', [0.5 0.5 0.5]);
set(ax, 'MinorGridColor', [0.7 0.7 0.7]);
saveas(fig, fullfile(figuresDir, 'kmeans_inercia_vs_k.png'));
fprintf('  Guardado: inercia_vs_k.png\n');

fig = figure('Color', 'white', 'Position', [100, 100, 800, 600]);
ax = axes('Parent', fig);
hold(ax, 'on');
for dIdx = 1:length(distanceMetrics)
    nmiVals = zeros(length(K_values), 1);
    for kIdx = 1:length(K_values)
        if ~isempty(results.nmi{kIdx, dIdx})
            nmiVals(kIdx) = results.nmi{kIdx, dIdx};
        else
            nmiVals(kIdx) = NaN;
        end
    end
    plot(ax, K_values, nmiVals, '-s', 'Color', colors(dIdx,:), ...
        'LineWidth', 1.5, 'MarkerSize', 8, 'MarkerFaceColor', colors(dIdx,:));
end
xlabel(ax, 'Número de Clústeres (K)', 'FontSize', 13, 'FontWeight', 'bold');
ylabel(ax, 'Información Mutua Normalizada (NMI)', 'FontSize', 13, 'FontWeight', 'bold');
title(ax, 'K-means: Calidad de Agrupamiento (NMI vs K)', 'FontSize', 14, 'FontWeight', 'bold');
legend(ax, distanceMetrics, 'Location', 'best', 'FontSize', 11);
grid(ax, 'on');
set(ax, 'Color', 'white');
set(ax, 'Box', 'on');
set(ax, 'LineWidth', 1);
set(ax, 'XColor', [0 0 0]);
set(ax, 'YColor', [0 0 0]);
set(ax, 'GridColor', [0.5 0.5 0.5]);
set(ax, 'MinorGridColor', [0.7 0.7 0.7]);
saveas(fig, fullfile(figuresDir, 'kmeans_nmi_vs_k.png'));
fprintf('  Guardado: nmi_vs_k.png\n');

fig = figure('Color', 'white', 'Position', [100, 100, 900, 700]);
ax = axes('Parent', fig);
accMatrix = zeros(length(K_values), length(distanceMetrics));
for kIdx = 1:length(K_values)
    for dIdx = 1:length(distanceMetrics)
        if ~isempty(results.accuracy{kIdx, dIdx})
            accMatrix(kIdx, dIdx) = results.accuracy{kIdx, dIdx};
        end
    end
end
imagesc(ax, accMatrix);
c = colorbar(ax);
c.Label.String = 'Precisión (%)';
c.Label.FontSize = 12;
c.Label.FontWeight = 'bold';
xlabel(ax, 'Métrica de Distancia', 'FontSize', 13, 'FontWeight', 'bold');
ylabel(ax, 'Número de Clústeres (K)', 'FontSize', 13, 'FontWeight', 'bold');
title(ax, 'Mapa de Calor de Precisión de Clasificación', 'FontSize', 14, 'FontWeight', 'bold');
xticks(ax, 1:length(distanceMetrics));
xticklabels(ax, distanceMetrics);
xtickangle(ax, 45);
yticks(ax, 1:length(K_values));
yticklabels(ax, string(K_values));
set(ax, 'Color', 'white');
set(ax, 'Box', 'on');
set(ax, 'LineWidth', 1);
set(ax, 'XColor', [0 0 0]);
set(ax, 'YColor', [0 0 0]);
set(ax, 'GridColor', [0.5 0.5 0.5]);
set(ax, 'MinorGridColor', [0.7 0.7 0.7]);
for kIdx = 1:length(K_values)
    for dIdx = 1:length(distanceMetrics)
        text(ax, dIdx, kIdx, sprintf('%.1f', accMatrix(kIdx, dIdx)), ...
            'HorizontalAlignment', 'center', 'Color', 'k', 'FontSize', 10);
    end
end
saveas(fig, fullfile(figuresDir, 'kmeans_precision_mapa_calor.png'));
fprintf('  Guardado: precision_mapa_calor.png\n');

fig = figure('Color', 'white', 'Position', [100, 100, 800, 600]);
ax = axes('Parent', fig);
hold(ax, 'on');
for dIdx = 1:length(distanceMetrics)
    timeVals = zeros(length(K_values), 1);
    for kIdx = 1:length(K_values)
        if ~isempty(results.time{kIdx, dIdx})
            timeVals(kIdx) = results.time{kIdx, dIdx};
        else
            timeVals(kIdx) = NaN;
        end
    end
    plot(ax, K_values, timeVals, '-^', 'Color', colors(dIdx,:), ...
        'LineWidth', 1.5, 'MarkerSize', 8, 'MarkerFaceColor', colors(dIdx,:));
end
xlabel(ax, 'Número de Clústeres (K)', 'FontSize', 13, 'FontWeight', 'bold');
ylabel(ax, 'Tiempo de Cómputo (segundos)', 'FontSize', 13, 'FontWeight', 'bold');
title(ax, 'K-means: Tiempo de Cómputo vs K', 'FontSize', 14, 'FontWeight', 'bold');
legend(ax, distanceMetrics, 'Location', 'best', 'FontSize', 11);
grid(ax, 'on');
set(ax, 'Color', 'white');
set(ax, 'Box', 'on');
set(ax, 'LineWidth', 1);
set(ax, 'XColor', [0 0 0]);
set(ax, 'YColor', [0 0 0]);
set(ax, 'GridColor', [0.5 0.5 0.5]);
set(ax, 'MinorGridColor', [0.7 0.7 0.7]);
saveas(fig, fullfile(figuresDir, 'kmeans_tiempo_vs_k.png'));
fprintf('  Guardado: tiempo_vs_k.png\n');

fprintf('\n=== Guardando Resultados ===\n');
save(fullfile(resultsDir, 'kmeans_metricas.mat'), 'results');
fprintf('Métricas guardadas en: %s\n', fullfile(resultsDir, 'kmeans_metricas.mat'));

fprintf('\n========================================\n');
fprintf('¡Agrupamiento K-means Completado!\n');
fprintf('========================================\n');
fprintf('Mejor configuración:\n');
fprintf('  K = %d, Distancia = %s\n', overallBestConfig.K, overallBestConfig.distance);
fprintf('  Precisión = %.2f%%\n', overallBestConfig.accuracy);
fprintf('\nGráficos guardados en: %s\n', figuresDir);
fprintf('========================================\n');


function nmi = evalClusteringNMI(clusterIdx, trueLabels)
N = length(clusterIdx);

K = max(clusterIdx);
C = length(unique(trueLabels));
contingency = zeros(K, C);

for i = 1:N
    contingency(clusterIdx(i), trueLabels(i)+1) = ...
        contingency(clusterIdx(i), trueLabels(i)+1) + 1;
end

p_ij = contingency / N;
p_i = sum(p_ij, 2);
p_j = sum(p_ij, 1);

MI = 0;
for i = 1:K
    for j = 1:C
        if p_ij(i,j) > 0
            MI = MI + p_ij(i,j) * log(p_ij(i,j) / (p_i(i) * p_j(j) + eps));
        end
    end
end

H_i = -sum(p_i .* log(p_i + eps));
H_j = -sum(p_j .* log(p_j + eps));

nmi = MI / sqrt(H_i * H_j + eps);
end

function ari = evalClusteringARI(clusterIdx, trueLabels)
N = length(clusterIdx);

K = max(clusterIdx);
C = length(unique(trueLabels));
contingency = zeros(K, C);

for i = 1:N
    contingency(clusterIdx(i), trueLabels(i)+1) = ...
        contingency(clusterIdx(i), trueLabels(i)+1) + 1;
end

sumCombNij = sum(sum(nchoosek2(contingency)));
sumCombAi = sum(nchoosek2(sum(contingency, 2)));
sumCombBj = sum(nchoosek2(sum(contingency, 1)));
combN = nchoosek2(N);

expectedIndex = sumCombAi * sumCombBj / combN;
maxIndex = (sumCombAi + sumCombBj) / 2;

if maxIndex == expectedIndex
    ari = 1;
else
    ari = (sumCombNij - expectedIndex) / (maxIndex - expectedIndex);
end
end

function c = nchoosek2(n)
c = n .* (n - 1) / 2;
end
