clear; clc; close all;

%% 1) Cargar datos
load('digits.mat');

X = double(digits.image);      % 784 x 10000
y = double(digits.label(:));   % 10000 x 1

%% 2) Separar datos en entrenamiento y prueba
cv = cvpartition(y, 'HoldOut', 0.30);

idxTrain = training(cv);
idxTest  = test(cv);

Xtrain_raw = X(:, idxTrain);
Xtest_raw  = X(:, idxTest);

ytrain = y(idxTrain);
ytest  = y(idxTest);

%% 3) Normalización usando SOLO entrenamiento
mean_d = mean(Xtrain_raw, 2);
std_d  = std(Xtrain_raw, 0, 2);
std_d(std_d == 0) = 1;

Xtrain_n = (Xtrain_raw - mean_d) ./ std_d;
Xtest_n  = (Xtest_raw  - mean_d) ./ std_d;

fprintf('Media global train normalizado: %.4f\n', mean(Xtrain_n(:)));
fprintf('Desv. estándar global train normalizado: %.4f\n', std(Xtrain_n(:)));

%% 4) Formato muestras x variables
Xtrain_full = Xtrain_n';
Xtest_full  = Xtest_n';

%% 5) Eliminar predictores con varianza cero dentro de alguna clase
classes = unique(ytrain);
numFeatures = size(Xtrain_full, 2);
keepFeature = true(1, numFeatures);

for j = 1:numFeatures
    for c = 1:length(classes)
        idxc = (ytrain == classes(c));
        if std(Xtrain_full(idxc, j)) == 0
            keepFeature(j) = false;
            break;
        end
    end
end

Xtrain_use = Xtrain_full(:, keepFeature);
Xtest_use  = Xtest_full(:, keepFeature);

fprintf('Predictores originales : %d\n', numFeatures);
fprintf('Predictores conservados: %d\n', sum(keepFeature));

%% 6) Configuración de reducción de dimensión
K_pca = 350;              % igual que tu compañero
methods = {'NONE', 'PCA', 'PCA_LDA'};

results = struct();
bestAcc = -inf;

%% 7) Probar Bayes con cada representación
for m = 1:length(methods)

    method = methods{m};

    switch method

        case 'NONE'
            Xtrain_final = Xtrain_use;
            Xtest_final  = Xtest_use;

            coeff_pca = [];
            mu_pca = [];
            explained = [];
            W_lda = [];
            mu_global_lda = [];

        case 'PCA'
            % Igual a la lógica de tu compañero
            [coeff_pca, score_pca, ~, ~, explained, mu_pca] = pca(Xtrain_use);

            Xtrain_final = score_pca(:, 1:K_pca);
            Xtest_centered = Xtest_use - mu_pca;
            Xtest_final = Xtest_centered * coeff_pca(:, 1:K_pca);

            W_lda = [];
            mu_global_lda = [];

        case 'PCA_LDA'
            % Primero PCA como en el código de tu compañero
            [coeff_pca, score_pca, ~, ~, explained, mu_pca] = pca(Xtrain_use);

            Xtrain_pca = score_pca(:, 1:K_pca);
            Xtest_centered = Xtest_use - mu_pca;
            Xtest_pca = Xtest_centered * coeff_pca(:, 1:K_pca);

            % Luego LDA como en el código de tu compañero
            classes_lda = unique(ytrain);
            num_classes = length(classes_lda);

            mu_global_lda = mean(Xtrain_pca);
            Sw = zeros(K_pca, K_pca);
            Sb = zeros(K_pca, K_pca);

            for i = 1:num_classes
                c = classes_lda(i);
                Xi = Xtrain_pca(ytrain == c, :);
                ni = size(Xi, 1);
                mu_class = mean(Xi);

                Xi_centered = Xi - mu_class;
                Sw = Sw + (Xi_centered' * Xi_centered);

                mean_diff = mu_class - mu_global_lda;
                Sb = Sb + ni * (mean_diff' * mean_diff);
            end

            [V, D_val] = eig(Sw \ Sb);
            [~, sort_idx] = sort(diag(D_val), 'descend');
            max_components = num_classes - 1;
            W_lda = real(V(:, sort_idx(1:max_components)));

            Xtrain_final = (Xtrain_pca - mu_global_lda) * W_lda;
            Xtest_final  = (Xtest_pca  - mu_global_lda) * W_lda;
    end

    %% Modelo 1: Prior uniforme
    nClasses = numel(unique(ytrain));
    prior_equal = ones(1, nClasses) / nClasses;

    bayMdl_equal = fitcnb(Xtrain_final, ytrain, 'Prior', prior_equal);

    yhat_train_equal = predict(bayMdl_equal, Xtrain_final);
    yhat_test_equal  = predict(bayMdl_equal, Xtest_final);

    acc_train_equal = mean(yhat_train_equal == ytrain) * 100;
    acc_test_equal  = mean(yhat_test_equal == ytest) * 100;

    err_train_equal = sum(yhat_train_equal ~= ytrain);
    err_test_equal  = sum(yhat_test_equal ~= ytest);

    %% Modelo 2: Prior proporcional
    class_labels = unique(ytrain);
    prior_prop = zeros(1, nClasses);

    for i = 1:nClasses
        prior_prop(i) = sum(ytrain == class_labels(i)) / numel(ytrain);
    end

    bayMdl_prop = fitcnb(Xtrain_final, ytrain, 'Prior', prior_prop);

    yhat_train_prop = predict(bayMdl_prop, Xtrain_final);
    yhat_test_prop  = predict(bayMdl_prop, Xtest_final);

    acc_train_prop = mean(yhat_train_prop == ytrain) * 100;
    acc_test_prop  = mean(yhat_test_prop == ytest) * 100;

    err_train_prop = sum(yhat_train_prop ~= ytrain);
    err_test_prop  = sum(yhat_test_prop ~= ytest);

    %% Elegir mejor prior dentro de este método
    if acc_test_equal >= acc_test_prop
        localBestMdl   = bayMdl_equal;
        localBestPred  = yhat_test_equal;
        localBestPrior = prior_equal;
        localBestType  = 'uniforme';
        localBestTrain = acc_train_equal;
        localBestTest  = acc_test_equal;
        localBestErrTr = err_train_equal;
        localBestErrTe = err_test_equal;
    else
        localBestMdl   = bayMdl_prop;
        localBestPred  = yhat_test_prop;
        localBestPrior = prior_prop;
        localBestType  = 'proporcional';
        localBestTrain = acc_train_prop;
        localBestTest  = acc_test_prop;
        localBestErrTr = err_train_prop;
        localBestErrTe = err_test_prop;
    end

    %% Guardar resultados del método
    results.(method).acc_train_equal = acc_train_equal;
    results.(method).acc_test_equal  = acc_test_equal;
    results.(method).acc_train_prop  = acc_train_prop;
    results.(method).acc_test_prop   = acc_test_prop;

    results.(method).err_train_equal = err_train_equal;
    results.(method).err_test_equal  = err_test_equal;
    results.(method).err_train_prop  = err_train_prop;
    results.(method).err_test_prop   = err_test_prop;

    results.(method).bestMdl   = localBestMdl;
    results.(method).bestPred  = localBestPred;
    results.(method).bestPrior = localBestPrior;
    results.(method).bestType  = localBestType;
    results.(method).bestTrain = localBestTrain;
    results.(method).bestTest  = localBestTest;
    results.(method).bestErrTr = localBestErrTr;
    results.(method).bestErrTe = localBestErrTe;

    results.(method).coeff_pca = coeff_pca;
    results.(method).mu_pca = mu_pca;
    results.(method).explained = explained;
    results.(method).W_lda = W_lda;
    results.(method).mu_global_lda = mu_global_lda;

    fprintf('\n=================================================\n');
    fprintf('METODO BAYES + : %s\n', method);
    fprintf('Prior uniforme     -> Train: %.2f %% | Test: %.2f %%\n', acc_train_equal, acc_test_equal);
    fprintf('Prior proporcional -> Train: %.2f %% | Test: %.2f %%\n', acc_train_prop, acc_test_prop);
    fprintf('Mejor en %s -> Prior %s | Test: %.2f %%\n', method, localBestType, localBestTest);

    %% Comparar contra el mejor global
    if localBestTest > bestAcc
        bestAcc = localBestTest;
        bestMethod = method;
        bestMdl = localBestMdl;
        bestPred = localBestPred;
        bestPrior = localBestPrior;
        bestType = localBestType;

        coeff_pca_best = coeff_pca;
        mu_pca_best = mu_pca;
        explained_best = explained;
        W_lda_best = W_lda;
        mu_global_lda_best = mu_global_lda;
    end
end

%% 8) Resumen final
fprintf('\n=================================================\n');
fprintf('MEJOR CONFIGURACION GLOBAL\n');
fprintf('Metodo BAYES + : %s\n', bestMethod);
fprintf('Prior : %s\n', bestType);
fprintf('Test accuracy: %.2f %%\n', bestAcc);
fprintf('=================================================\n');

figure;
confusionchart(ytest, bestPred);
title(['Matriz de confusión - Bayes (' bestMethod ' | ' bestType ')']);

%% 9) Guardar modelo final
save('bayes_model_digits.mat', ...
    'bestMdl', 'bestMethod', 'bestType', 'bestPrior', 'bestAcc', ...
    'mean_d', 'std_d', 'keepFeature', 'K_pca', ...
    'coeff_pca_best', 'mu_pca_best', 'explained_best', ...
    'W_lda_best', 'mu_global_lda_best', ...
    'results', '-v7.3');