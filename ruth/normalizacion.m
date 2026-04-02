clear; clc; close all;

% Cargar datos de entrenamiento
load('digits.mat');   % contiene digits.image y digits.label

X = double(digits.image);   % 784 x 10000
labels = digits.label;
[D, N] = size(X);

% Media por píxel
mean_d = mean(X, 2);

% Desviación estándar por píxel
std_d = std(X, 0, 2);

% Evitar división por cero
std_d(std_d == 0) = 1;

% Normalización Z-score
Xn = (X - mean_d) ./ std_d;

% Verificación global
fprintf('Global mean after normalization: %.6f\n', mean(Xn(:)));
fprintf('Global std after normalization: %.6f\n', std(Xn(:)));

% Figura 1: mapa de media y desviación estándar
figure('Name', 'Statistical Maps', 'Position', [100, 100, 800, 400]);

subplot(1, 2, 1);
imshow(reshape(mean_d, 28, 28)', []);
title('Mean Image');
colorbar;

subplot(1, 2, 2);
imshow(reshape(std_d, 28, 28)', []);
title('Standard Deviation Map');
colorbar;

% Figura 2: histogramas antes y después de normalizar
figure('Name', 'Distribution Histograms', 'Position', [150, 150, 800, 400]);

subplot(1, 2, 1);
histogram(X(:), 50);
title('Original Distribution');

subplot(1, 2, 2);
histogram(Xn(:), 50);
title('Normalized Distribution (Z-score)');

% Figura 3: comparación visual entre imágenes originales y normalizadas
target_numbers = 0:9;
figure('Name', 'Full Visual Comparison', 'Position', [50, 250, 1400, 400]);

for i = 1:length(target_numbers)
    num = target_numbers(i);
    idx = find(labels == num, 1);

    subplot(2, 10, i);
    imshow(reshape(X(:, idx), 28, 28)', []);
    title(['Digit ' num2str(num)]);
    set(gca, 'XTick', [], 'YTick', []);
    if i == 1
        ylabel('Original', 'Visible', 'on', 'FontWeight', 'bold');
    end

    subplot(2, 10, i + 10);
    imshow(reshape(Xn(:, idx), 28, 28)', []);
    set(gca, 'XTick', [], 'YTick', []);
    if i == 1
        ylabel('Normalized', 'Visible', 'on', 'FontWeight', 'bold');
    end
end

% Guardar para reutilizar en clasificación
save('digits_normalized.mat', 'Xn', 'mean_d', 'std_d', '-v7.3');