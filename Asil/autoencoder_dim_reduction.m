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

maxEpochs = 200;
miniBatchSize = 64;
learnRate = 0.001;

fprintf('=== Cargando Datos===\n', nSamples);
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

fprintf('\n=== Configuración de Parámetros  ===\n');

bottleneckSizes = [10, 50, 100, 128, 256];

epochValues = [50, 100, 200];

fprintf('Tamaños de bottleneck a probar: %s\n', mat2str(bottleneckSizes));
fprintf('Valores de epoch a probar: %s\n', mat2str(epochValues));
fprintf('Tasa de aprendizaje: %.4f\n', learnRate);
fprintf('Tamaño de mini-lote: %d\n', miniBatchSize);

results = struct();
results.bottleneckSizes = bottleneckSizes;
results.epochValues = epochValues;
results.accuracy = cell(length(bottleneckSizes), length(epochValues));
results.reconMSE = cell(length(bottleneckSizes), length(epochValues));
results.trainTime = cell(length(bottleneckSizes), length(epochValues));
results.bestConfig = [];

fprintf('\n=== Ejecutando Experimentos de Autoencoder ===\n');

overallBestAccuracy = 0;
overallBestConfig = struct();

for bIdx = 1:length(bottleneckSizes)
    bottleneck = bottleneckSizes(bIdx);
    fprintf('\n=== Tamaño de bottleneck = %d ===\n', bottleneck);

    for eIdx = 1:length(epochValues)
        epochs = epochValues(eIdx);
        fprintf('  Epochs: %d... ', epochs);

        tic;
        try
            autoenc = trainAutoencoder(X_train, bottleneck, ...
                'MaxEpochs', epochs, ...
                'ShowProgressWindow', false);

            trainTime = toc;

            encoded_train = encode(autoenc, X_train);
            encoded_test = encode(autoenc, X_test);

            reconstructed_test = decode(autoenc, encoded_test);
            reconMSE = mean((X_test - reconstructed_test).^2, 'all');

            classifier = fitcecoc(encoded_train', Y_train);
            preds = predict(classifier, encoded_test');
            accuracy = mean(preds == Y_test) * 100;

            fprintf('Prec: %.2f%%, MSE Recon: %.4f\n', accuracy, reconMSE);

            results.accuracy{bIdx, eIdx} = accuracy;
            results.reconMSE{bIdx, eIdx} = reconMSE;
            results.trainTime{bIdx, eIdx} = trainTime;

            if accuracy > overallBestAccuracy
                overallBestAccuracy = accuracy;
                overallBestConfig.bottleneck = bottleneck;
                overallBestConfig.epochs = epochs;
                overallBestConfig.accuracy = accuracy;
                overallBestConfig.classifier = classifier;
                overallBestConfig.autoencoder = autoenc;
                overallBestConfig.encoded_test = encoded_test;
            end

        catch ME
            fprintf('FAILED: %s\n', ME.message);
            results.accuracy{bIdx, eIdx} = NaN;
            results.reconMSE{bIdx, eIdx} = NaN;
            results.trainTime{bIdx, eIdx} = NaN;
        end
    end
end

fprintf('\n=== Mejor Configuración ===\n');
fprintf('Bottleneck = %d, Epochs = %d, Precisión = %.2f%%\n', ...
    overallBestConfig.bottleneck, overallBestConfig.epochs, overallBestConfig.accuracy);

results.bestConfig = overallBestConfig;

fprintf('\n=== Generando Gráficos ===\n');

figure('Position', [100, 100, 800, 600]);
hold on;
colors = lines(length(epochValues));
for eIdx = 1:length(epochValues)
    accVals = zeros(length(bottleneckSizes), 1);
    for bIdx = 1:length(bottleneckSizes)
        if ~isempty(results.accuracy{bIdx, eIdx}) && ~isnan(results.accuracy{bIdx, eIdx})
            accVals(bIdx) = results.accuracy{bIdx, eIdx};
        else
            accVals(bIdx) = NaN;
        end
    end
    plot(bottleneckSizes, accVals, '-o', 'Color', colors(eIdx,:), ...
        'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', sprintf('%d epochs', epochValues(eIdx)));
end
xlabel('Tamaño del Bottleneck');
ylabel('Precisión de Clasificación (%)');
title('Autoencoder: Precisión vs Tamaño del Bottleneck');
legend('Location', 'best');
grid on;
xlim([min(bottleneckSizes)-5, max(bottleneckSizes)+5]);
saveas(gcf, fullfile(figuresDir, 'autoencoder_precision_vs_bottleneck.png'));
fprintf('  Guardado: precision_vs_bottleneck.png\n');

figure('Position', [100, 100, 800, 600]);
hold on;
for eIdx = 1:length(epochValues)
    mseVals = zeros(length(bottleneckSizes), 1);
    for bIdx = 1:length(bottleneckSizes)
        if ~isempty(results.reconMSE{bIdx, eIdx}) && ~isnan(results.reconMSE{bIdx, eIdx})
            mseVals(bIdx) = results.reconMSE{bIdx, eIdx};
        else
            mseVals(bIdx) = NaN;
        end
    end
    plot(bottleneckSizes, mseVals, '-s', 'Color', colors(eIdx,:), ...
        'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', sprintf('%d epochs', epochValues(eIdx)));
end
xlabel('Tamaño del Bottleneck');
ylabel('MSE de Reconstrucción');
title('Autoencoder: Error de Reconstrucción vs Tamaño del Bottleneck');
legend('Location', 'best');
grid on;
saveas(gcf, fullfile(figuresDir, 'autoencoder_mse_reconstruccion.png'));
fprintf('  Guardado: mse_reconstruccion.png\n');

figure('Position', [100, 100, 900, 700]);
accMatrix = zeros(length(bottleneckSizes), length(epochValues));
for bIdx = 1:length(bottleneckSizes)
    for eIdx = 1:length(epochValues)
        if ~isempty(results.accuracy{bIdx, eIdx}) && ~isnan(results.accuracy{bIdx, eIdx})
            accMatrix(bIdx, eIdx) = results.accuracy{bIdx, eIdx};
        end
    end
end
imagesc(accMatrix);
colorbar;
xlabel('Epochs');
ylabel('Tamaño del Bottleneck');
title('Mapa de Calor de Precisión de Clasificación');
xticks(1:length(epochValues));
xticklabels(string(epochValues));
yticks(1:length(bottleneckSizes));
yticklabels(string(bottleneckSizes));
for bIdx = 1:length(bottleneckSizes)
    for eIdx = 1:length(epochValues)
        if ~isnan(accMatrix(bIdx, eIdx))
            text(eIdx, bIdx, sprintf('%.1f', accMatrix(bIdx, eIdx)), ...
                'HorizontalAlignment', 'center', 'Color', 'k');
        end
    end
end
saveas(gcf, fullfile(figuresDir, 'autoencoder_precision_mapa_calor.png'));
fprintf('  Guardado: precision_mapa_calor.png\n');

figure('Position', [100, 100, 800, 600]);
hold on;
for eIdx = 1:length(epochValues)
    timeVals = zeros(length(bottleneckSizes), 1);
    for bIdx = 1:length(bottleneckSizes)
        if ~isempty(results.trainTime{bIdx, eIdx}) && ~isnan(results.trainTime{bIdx, eIdx})
            timeVals(bIdx) = results.trainTime{bIdx, eIdx};
        else
            timeVals(bIdx) = NaN;
        end
    end
    plot(bottleneckSizes, timeVals, '-^', 'Color', colors(eIdx,:), ...
        'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', sprintf('%d epochs', epochValues(eIdx)));
end
xlabel('Tamaño del Bottleneck');
ylabel('Tiempo de Entrenamiento (segundos)');
title('Autoencoder: Tiempo de Entrenamiento vs Tamaño del Bottleneck');
legend('Location', 'best');
grid on;
saveas(gcf, fullfile(figuresDir, 'autoencoder_tiempo_entrenamiento.png'));
fprintf('  Guardado: tiempo_entrenamiento.png\n');

if ~isempty(overallBestConfig.bottleneck)
    bestBottleneck = overallBestConfig.bottleneck;
    bestEpochs = overallBestConfig.epochs;
    bIdx = find(bottleneckSizes == bestBottleneck);

    eIdx = find(epochValues == bestEpochs);

    if ~isempty(bIdx) && ~isempty(eIdx)
        autoenc = overallBestConfig.autoencoder;

        nSamples = 10;
        X_sample = X_test(:, 1:nSamples);
        encoded = encode(autoenc, X_sample);
        reconstructed = decode(autoenc, encoded);

        figure('Position', [100, 100, 1200, 400]);
        for i = 1:nSamples
            subplot(2, nSamples, i);
            imshow(reshape(X_sample(:, i), 28, 28), []);
            title(sprintf('Orig: %d', Y_test(i)));
            axis off;

            subplot(2, nSamples, i + nSamples);
            imshow(reshape(reconstructed(:, i), 28, 28), []);
            axis off;
        end
        sgtitle(sprintf('Reconstrucciones del Autoencoder (Bottleneck=%d, Epochs=%d)', ...
            bestBottleneck, bestEpochs));
        saveas(gcf, fullfile(figuresDir, 'autoencoder_reconstrucciones.png'));
        fprintf('  Guardado: reconstrucciones.png\n');
    end
end

if ~isempty(overallBestConfig.encoded_test)
    encoded = overallBestConfig.encoded_test;

    if size(encoded, 1) > 2
        [coeff, score] = pca(encoded');
        pca2D = score(:, 1:2);

        figure('Position', [100, 100, 800, 600]);
        gscatter(pca2D(:,1), pca2D(:,2), Y_test, [], 'o');
        xlabel('PC1');
        ylabel('PC2');
        title('Características del Bottleneck - Proyección PCA 2D');
        legend('Location', 'bestoutside');
        saveas(gcf, fullfile(figuresDir, 'autoencoder_bottleneck_pca2d.png'));
        fprintf('  Guardado: bottleneck_pca2d.png\n');
    end
end

fprintf('\n=== Guardando Resultados ===\n');
save(fullfile(resultsDir, 'autoencoder_metricas.mat'), 'results');
fprintf('Métricas guardadas en: %s\n', fullfile(resultsDir, 'autoencoder_metricas.mat'));

fprintf('\n========================================\n');
fprintf('¡Autoencoder Finalizado!\n');
fprintf('========================================\n');
fprintf('Mejor configuración:\n');
fprintf('  Bottleneck = %d\n', overallBestConfig.bottleneck);
fprintf('  Epochs = %d\n', overallBestConfig.epochs);
fprintf('  Precisión = %.2f%%\n', overallBestConfig.accuracy);
fprintf('\nGráficos guardados en: %s\n', figuresDir);
fprintf('========================================\n');
