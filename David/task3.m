load('digits.mat');
X_raw = double(digits.image);
labels = digits.label;

[Xn, settings] = mapstd(X_raw);
X = Xn';

classes = 0:9;
num_classes = length(classes);

[coeff_pca, score_pca, ~, ~, ~, mu_pca] = pca(X);
K_pca = 350; 
X_pca = score_pca(:, 1:K_pca);

mu_global = mean(X_pca);
Sw = zeros(K_pca, K_pca);
Sb = zeros(K_pca, K_pca);

for i = 1:num_classes
    c = classes(i);
    Xi = X_pca(labels == c, :);
    ni = size(Xi, 1);
    mu_class = mean(Xi);
    
    Xi_centered = Xi - mu_class;
    Sw = Sw + (Xi_centered' * Xi_centered);
    
    mean_diff = mu_class - mu_global;
    Sb = Sb + ni * (mean_diff' * mean_diff);
end

[V, D_val] = eig(Sw \ Sb);
[~, sort_idx] = sort(diag(D_val), 'descend');
max_components = num_classes - 1;
W_lda = real(V(:, sort_idx(1:max_components)));

features_lda = (X_pca - mu_global) * W_lda;

figure('Name', 'LDA 2D Projection', 'Position', [100, 100, 800, 600]);
gscatter(features_lda(:,1), features_lda(:,2), labels, jet(10), '.', 15);
title('2D Projection of MNIST Classes using PCA+LDA');
xlabel('LDA Component 1');
ylabel('LDA Component 2');
legend('0','1','2','3','4','5','6','7','8','9', 'Location', 'bestoutside');
grid on;

figure('Name', 'LDA 3D Projection', 'Position', [150, 150, 800, 600]);
scatter3(features_lda(:,1), features_lda(:,2), features_lda(:,3), 15, labels, 'filled');
colormap(jet(10));
cb = colorbar;
cb.Ticks = 0:9;
title('3D Projection of MNIST Classes using PCA+LDA');
xlabel('LDA Component 1');
ylabel('LDA Component 2');
zlabel('LDA Component 3');
grid on;