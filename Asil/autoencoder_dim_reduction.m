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

fig = figure('Color', 'white', 'Position', [100, 100, 800, 600]);
ax = axes('Parent', fig);
hold(ax, 'on');
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
    plot(ax, bottleneckSizes, accVals, '-o', 'Color', colors(eIdx,:), ...
        'LineWidth', 1.5, 'MarkerSize', 8, 'MarkerFaceColor', colors(eIdx,:), ...
        'DisplayName', sprintf('%d epochs', epochValues(eIdx)));
end
xlabel(ax, 'Tamaño del Bottleneck', 'FontSize', 13, 'FontWeight', 'bold');
ylabel(ax, 'Precisión de Clasificación (%)', 'FontSize', 13, 'FontWeight', 'bold');
title(ax, 'Autoencoder: Precisión vs Tamaño del Bottleneck', 'FontSize', 14, 'FontWeight', 'bold');
legend(ax, 'Location', 'best', 'FontSize', 11);
grid(ax, 'on');
set(ax, 'Color', 'white');
set(ax, 'Box', 'on');
set(ax, 'LineWidth', 1);
set(ax, 'XColor', [0 0 0]);
set(ax, 'YColor', [0 0 0]);
set(ax, 'GridColor', [0.5 0.5 0.5]);
set(ax, 'MinorGridColor', [0.7 0.7 0.7]);
xlim(ax, [min(bottleneckSizes)-5, max(bottleneckSizes)+5]);
saveas(fig, fullfile(figuresDir, 'autoencoder_precision_vs_bottleneck.png'));
fprintf('  Guardado: precision_vs_bottleneck.png\n');

fig = figure('Color', 'white', 'Position', [100, 100, 800, 600]);
ax = axes('Parent', fig);
hold(ax, 'on');
for eIdx = 1:length(epochValues)
    mseVals = zeros(length(bottleneckSizes), 1);
    for bIdx = 1:length(bottleneckSizes)
        if ~isempty(results.reconMSE{bIdx, eIdx}) && ~isnan(results.reconMSE{bIdx, eIdx})
            mseVals(bIdx) = results.reconMSE{bIdx, eIdx};
        else
            mseVals(bIdx) = NaN;
        end
    end
    plot(ax, bottleneckSizes, mseVals, '-s', 'Color', colors(eIdx,:), ...
        'LineWidth', 1.5, 'MarkerSize', 8, 'MarkerFaceColor', colors(eIdx,:), ...
        'DisplayName', sprintf('%d epochs', epochValues(eIdx)));
end
xlabel(ax, 'Tamaño del Bottleneck', 'FontSize', 13, 'FontWeight', 'bold');
ylabel(ax, 'MSE de Reconstrucción', 'FontSize', 13, 'FontWeight', 'bold');
title(ax, 'Autoencoder: Error de Reconstrucción vs Tamaño del Bottleneck', 'FontSize', 14, 'FontWeight', 'bold');
legend(ax, 'Location', 'best', 'FontSize', 11);
grid(ax, 'on');
set(ax, 'Color', 'white');
set(ax, 'Box', 'on');
set(ax, 'LineWidth', 1);
set(ax, 'XColor', [0 0 0]);
set(ax, 'YColor', [0 0 0]);
set(ax, 'GridColor', [0.5 0.5 0.5]);
set(ax, 'MinorGridColor', [0.7 0.7 0.7]);
saveas(fig, fullfile(figuresDir, 'autoencoder_mse_reconstruccion.png'));
fprintf('  Guardado: mse_reconstruccion.png\n');

fig = figure('Color', 'white', 'Position', [100, 100, 900, 700]);
ax = axes('Parent', fig);
accMatrix = zeros(length(bottleneckSizes), length(epochValues));
for bIdx = 1:length(bottleneckSizes)
    for eIdx = 1:length(epochValues)
        if ~isempty(results.accuracy{bIdx, eIdx}) && ~isnan(results.accuracy{bIdx, eIdx})
            accMatrix(bIdx, eIdx) = results.accuracy{bIdx, eIdx};
        end
    end
end
imagesc(ax, accMatrix);
c = colorbar(ax);
c.Label.String = 'Precisión (%)';
c.Label.FontSize = 12;
c.Label.FontWeight = 'bold';
xlabel(ax, 'Epochs', 'FontSize', 13, 'FontWeight', 'bold');
ylabel(ax, 'Tamaño del Bottleneck', 'FontSize', 13, 'FontWeight', 'bold');
title(ax, 'Mapa de Calor de Precisión de Clasificación', 'FontSize', 14, 'FontWeight', 'bold');
xticks(ax, 1:length(epochValues));
xticklabels(ax, string(epochValues));
yticks(ax, 1:length(bottleneckSizes));
yticklabels(ax, string(bottleneckSizes));
set(ax, 'Color', 'white');
set(ax, 'Box', 'on');
set(ax, 'LineWidth', 1);
set(ax, 'XColor', [0 0 0]);
set(ax, 'YColor', [0 0 0]);
set(ax, 'GridColor', [0.5 0.5 0.5]);
set(ax, 'MinorGridColor', [0.7 0.7 0.7]);
for bIdx = 1:length(bottleneckSizes)
    for eIdx = 1:length(epochValues)
        if ~isnan(accMatrix(bIdx, eIdx))
            text(ax, eIdx, bIdx, sprintf('%.1f', accMatrix(bIdx, eIdx)), ...
                'HorizontalAlignment', 'center', 'Color', 'k', 'FontSize', 10);
        end
    end
end
saveas(fig, fullfile(figuresDir, 'autoencoder_precision_mapa_calor.png'));
fprintf('  Guardado: precision_mapa_calor.png\n');

fig = figure('Color', 'white', 'Position', [100, 100, 800, 600]);
ax = axes('Parent', fig);
hold(ax, 'on');
for eIdx = 1:length(epochValues)
    timeVals = zeros(length(bottleneckSizes), 1);
    for bIdx = 1:length(bottleneckSizes)
        if ~isempty(results.trainTime{bIdx, eIdx}) && ~isnan(results.trainTime{bIdx, eIdx})
            timeVals(bIdx) = results.trainTime{bIdx, eIdx};
        else
            timeVals(bIdx) = NaN;
        end
    end
    plot(ax, bottleneckSizes, timeVals, '-^', 'Color', colors(eIdx,:), ...
        'LineWidth', 1.5, 'MarkerSize', 8, 'MarkerFaceColor', colors(eIdx,:), ...
        'DisplayName', sprintf('%d epochs', epochValues(eIdx)));
end
xlabel(ax, 'Tamaño del Bottleneck', 'FontSize', 13, 'FontWeight', 'bold');
ylabel(ax, 'Tiempo de Entrenamiento (segundos)', 'FontSize', 13, 'FontWeight', 'bold');
title(ax, 'Autoencoder: Tiempo de Entrenamiento vs Tamaño del Bottleneck', 'FontSize', 14, 'FontWeight', 'bold');
legend(ax, 'Location', 'best', 'FontSize', 11);
grid(ax, 'on');
set(ax, 'Color', 'white');
set(ax, 'Box', 'on');
set(ax, 'LineWidth', 1);
set(ax, 'XColor', [0 0 0]);
set(ax, 'YColor', [0 0 0]);
set(ax, 'GridColor', [0.5 0.5 0.5]);
set(ax, 'MinorGridColor', [0.7 0.7 0.7]);
saveas(fig, fullfile(figuresDir, 'autoencoder_tiempo_entrenamiento.png'));
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

        fig = figure('Color', 'white', 'Position', [100, 100, 1200, 400]);
        for i = 1:nSamples
            ax1 = subplot(2, nSamples, i);
            imshow(reshape(X_sample(:, i), 28, 28), []);
            title(ax1, sprintf('Orig: %d', Y_test(i)), 'FontSize', 10);
            axis off;

            ax2 = subplot(2, nSamples, i + nSamples);
            imshow(reshape(reconstructed(:, i), 28, 28), []);
            axis(ax2, 'off');
        end
        sgtitle(sprintf('Reconstrucciones del Autoencoder (Bottleneck=%d, Epochs=%d)', ...
            bestBottleneck, bestEpochs), 'FontSize', 12, 'FontWeight', 'bold');
        saveas(fig, fullfile(figuresDir, 'autoencoder_reconstrucciones.png'));
        fprintf('  Guardado: reconstrucciones.png\n');
    end
end

if ~isempty(overallBestConfig.encoded_test)
    encoded = overallBestConfig.encoded_test;

    if size(encoded, 1) > 2
        [coeff, score] = pca(encoded');
        pca2D = score(:, 1:2);

        fig = figure('Color', 'white', 'Position', [100, 100, 800, 600]);
        ax = axes('Parent', fig);
        gscatter(ax, pca2D(:,1), pca2D(:,2), Y_test, [], 'o');
        xlabel(ax, 'PC1', 'FontSize', 13, 'FontWeight', 'bold');
        ylabel(ax, 'PC2', 'FontSize', 13, 'FontWeight', 'bold');
        title(ax, 'Características del Bottleneck - Proyección PCA 2D', 'FontSize', 14, 'FontWeight', 'bold');
        legend(ax, 'Location', 'bestoutside', 'FontSize', 10);
        grid(ax, 'on');
        set(ax, 'Color', 'white');
        set(ax, 'Box', 'on');
        set(ax, 'LineWidth', 1);
        saveas(fig, fullfile(figuresDir, 'autoencoder_bottleneck_pca2d.png'));
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
