clear; clc; close all;
load('digits.mat');
X = double(digits.image);
labels = digits.label;
[D, N] = size(X);

mean_d = mean(X, 2);    
std_d = std(X, 0, 2);   
std_d(std_d == 0) = 1;
Xn = (X - mean_d) ./ std_d;

fprintf('Global mean after normalization: %.4f\n', mean(Xn(:)));
fprintf('Global standard deviation after normalization: %.4f\n', std(Xn(:)));

figure('Name', 'Statistical Maps', 'Position', [100, 100, 800, 400]);
subplot(1, 2, 1);
imshow(reshape(mean_d, 28, 28)', []);
title('Mean Image');
colorbar;

subplot(1, 2, 2);
imshow(reshape(std_d, 28, 28)', []);
title('Standard Deviation Map');
colorbar;

figure('Name', 'Distribution Histograms', 'Position', [150, 150, 800, 400]);
subplot(1, 2, 1);
histogram(X(:), 50);
title('Original Distribution');

subplot(1, 2, 2);
histogram(Xn(:), 50);
title('Normalized Distribution (Z-score)');

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

save('digits_normalized.mat', 'Xn', 'mean_d', 'std_d', '-v7.3');