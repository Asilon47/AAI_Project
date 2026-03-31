load('digits.mat');
X_raw = double(digits.image);
labels = digits.label;

[Xn, settings] = mapstd(X_raw);
X = Xn';

[coeff_pca, score_pca, ~, ~, explained, mu_pca] = pca(X);

K_features = 350; 
features_pca = score_pca(:, 1:K_features);

figure('Name', 'PCA 2D Projection', 'Position', [100, 100, 800, 600]);
gscatter(features_pca(:,1), features_pca(:,2), labels, jet(10), '.', 15);
title('2D Projection of MNIST Classes using PCA');
xlabel('Principal Component 1');
ylabel('Principal Component 2');
legend('0','1','2','3','4','5','6','7','8','9', 'Location', 'bestoutside');
grid on;

figure('Name', 'PCA 3D Projection', 'Position', [150, 150, 800, 600]);
scatter3(features_pca(:,1), features_pca(:,2), features_pca(:,3), 15, labels, 'filled');
colormap(jet(10));
cb = colorbar;
cb.Ticks = 0:9;
title('3D Projection of MNIST Classes using PCA');
xlabel('Principal Component 1');
ylabel('Principal Component 2');
zlabel('Principal Component 3');
grid on;

figure('Name', 'PCA Explained Variance', 'Position', [200, 200, 600, 400]);
plot(cumsum(explained), 'LineWidth', 0.1);
title('Cumulative Explained Variance by PCA Components');
xlabel('Number of Principal Components');
ylabel('Cumulative Variance (%)');
grid on;

k_components = [300, 100, 50, 25, 10, 3, 1];
target_numbers = 0:9;

figure('Name', 'PCA Digit Reconstruction', 'Position', [50, 50, 1400, 900]);
for i = 1:length(target_numbers)
    num = target_numbers(i);
    idx = find(labels == num, 1);
    
    original_vec = X(idx, :);
    digit_image_orig = reshape(original_vec, 28, 28)';
    
    subplot(10, 8, (i-1)*8 + 1);
    imshow(digit_image_orig, []);
    ylabel(['Digit ' num2str(num)], 'Visible', 'on', 'FontWeight', 'bold');
    set(gca, 'XTick', [], 'YTick', []);
    
    if i == 1
        title('Original');
    end
    
    for j = 1:length(k_components)
        K = k_components(j);
        reduced_score = score_pca(idx, 1:K);
        reduced_coeff = coeff_pca(:, 1:K)';
        reconstructed_vec = (reduced_score * reduced_coeff) + mu_pca;
        digit_image_recon = reshape(reconstructed_vec, 28, 28)';
        
        subplot(10, 8, (i-1)*8 + j + 1);
        imshow(digit_image_recon, []);
        
        if i == 1
            title(['K=' num2str(K)]);
        end
    end
end