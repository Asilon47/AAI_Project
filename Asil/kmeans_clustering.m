
clear; clc; close all;

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

figure('Position', [100, 100, 800, 600]);
hold on;
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
    plot(K_values, accVals, '-o', 'Color', colors(dIdx,:), ...
        'LineWidth', 2, 'MarkerSize', 8);
end
xlabel('Número de Clústeres (K)');
ylabel('Precisión de Clasificación (%)');
title('K-means: Precisión vs Número de Clústeres');
legend(distanceMetrics, 'Location', 'best');
grid on;
xlim([min(K_values)-5, max(K_values)+5]);
ylim([min(accVals)-5, max(accVals)+5]);
saveas(gcf, fullfile(figuresDir, 'kmeans_precision_vs_k.png'));
fprintf('  Guardado: precision_vs_k.png\n');

figure('Position', [100, 100, 800, 600]);
hold on;
inertiaVals = zeros(length(K_values), 1);
for kIdx = 1:length(K_values)
    if ~isempty(results.inertia{kIdx, 1})
        inertiaVals(kIdx) = results.inertia{kIdx, 1};
    else
        inertiaVals(kIdx) = NaN;
    end
end
plot(K_values, inertiaVals, '-o', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Número de Clústeres (K)');
ylabel('Inercia (Suma de Cuadrados Intra-clúster)');
title('K-means: Inercia vs K');
grid on;
saveas(gcf, fullfile(figuresDir, 'kmeans_inercia_vs_k.png'));
fprintf('  Guardado: inercia_vs_k.png\n');

figure('Position', [100, 100, 800, 600]);
hold on;
for dIdx = 1:length(distanceMetrics)
    nmiVals = zeros(length(K_values), 1);
    for kIdx = 1:length(K_values)
        if ~isempty(results.nmi{kIdx, dIdx})
            nmiVals(kIdx) = results.nmi{kIdx, dIdx};
        else
            nmiVals(kIdx) = NaN;
        end
    end
    plot(K_values, nmiVals, '-s', 'Color', colors(dIdx,:), ...
        'LineWidth', 2, 'MarkerSize', 8);
end
xlabel('Número de Clústeres (K)');
ylabel('Información Mutua Normalizada (NMI)');
title('K-means: Calidad de Agrupamiento (NMI vs K)');
legend(distanceMetrics, 'Location', 'best');
grid on;
saveas(gcf, fullfile(figuresDir, 'kmeans_nmi_vs_k.png'));
fprintf('  Guardado: nmi_vs_k.png\n');

figure('Position', [100, 100, 900, 700]);
accMatrix = zeros(length(K_values), length(distanceMetrics));
for kIdx = 1:length(K_values)
    for dIdx = 1:length(distanceMetrics)
        if ~isempty(results.accuracy{kIdx, dIdx})
            accMatrix(kIdx, dIdx) = results.accuracy{kIdx, dIdx};
        end
    end
end
imagesc(accMatrix);
colorbar;
xlabel('Métrica de Distancia');
ylabel('Número de Clústeres (K)');
title('Mapa de Calor de Precisión de Clasificación');
xticks(1:length(distanceMetrics));
xticklabels(distanceMetrics);
xtickangle(45);
yticks(1:length(K_values));
yticklabels(string(K_values));
for kIdx = 1:length(K_values)
    for dIdx = 1:length(distanceMetrics)
        text(dIdx, kIdx, sprintf('%.1f', accMatrix(kIdx, dIdx)), ...
            'HorizontalAlignment', 'center', 'Color', ...
            'k');
    end
end
saveas(gcf, fullfile(figuresDir, 'kmeans_precision_mapa_calor.png'));
fprintf('  Guardado: precision_mapa_calor.png\n');

figure('Position', [100, 100, 800, 600]);
hold on;
for dIdx = 1:length(distanceMetrics)
    timeVals = zeros(length(K_values), 1);
    for kIdx = 1:length(K_values)
        if ~isempty(results.time{kIdx, dIdx})
            timeVals(kIdx) = results.time{kIdx, dIdx};
        else
            timeVals(kIdx) = NaN;
        end
    end
    plot(K_values, timeVals, '-^', 'Color', colors(dIdx,:), ...
        'LineWidth', 2, 'MarkerSize', 8);
end
xlabel('Número de Clústeres (K)');
ylabel('Tiempo de Cómputo (segundos)');
title('K-means: Tiempo de Cómputo vs K');
legend(distanceMetrics, 'Location', 'best');
grid on;
saveas(gcf, fullfile(figuresDir, 'kmeans_tiempo_vs_k.png'));
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
