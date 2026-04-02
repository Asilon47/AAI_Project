clear; clc; close all;
rng(42, 'twister');

out_dir = 'task8_outputs';
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

load('digits.mat');
X_raw = double(digits.image);
labels = digits.label(:);
N = length(labels);
num_classes = 10;

mu_norm = mean(X_raw, 2);
sigma_norm = std(X_raw, 0, 2);
sigma_norm(sigma_norm == 0) = 1;
Xn = ((X_raw - mu_norm) ./ sigma_norm)';

labels_cat = categorical(labels);

cv_hold = cvpartition(labels, 'HoldOut', 0.2);
idx_train = training(cv_hold);
idx_test = test(cv_hold);

X_train_full = Xn(idx_train, :);
Y_train_cat = labels_cat(idx_train);
Y_train_num = labels(idx_train);
X_test_full = Xn(idx_test, :);
Y_test_cat = labels_cat(idx_test);
Y_test_num = labels(idx_test);

[coeff_pca, score_train_pca, ~, ~, explained_pca, mu_pca] = pca(X_train_full);
score_test_pca = (X_test_full - mu_pca) * coeff_pca;

K_pre = 350;
X_tr_pre = score_train_pca(:, 1:K_pre);
X_te_pre = score_test_pca(:, 1:K_pre);
mu_g = mean(X_tr_pre);

Sw = zeros(K_pre);
Sb = zeros(K_pre);
for c = 0:9
    Xc = X_tr_pre(Y_train_num == c, :);
    nc = size(Xc, 1);
    mc = mean(Xc);
    Xc_c = Xc - mc;
    Sw = Sw + (Xc_c' * Xc_c);
    d = mc - mu_g;
    Sb = Sb + nc * (d' * d);
end
[V_lda, D_lda] = eig(Sw \ Sb);
[~, si] = sort(diag(D_lda), 'descend');
W_lda = real(V_lda(:, si(1:num_classes-1)));

sc_tr_lda = (X_tr_pre - mu_g) * W_lda;
sc_te_lda = (X_te_pre - mu_g) * W_lda;

pca_ks = [10, 25, 50, 100, 200, 350];
lda_ks = [3, 5, 7, 9];

reps = {};
reps{end+1} = struct('name','Raw-784','Xtr',X_train_full,'Xte',X_test_full,'dim',784,'family','Raw');
for k = pca_ks
    reps{end+1} = struct('name',sprintf('PCA-%d',k),'Xtr',score_train_pca(:,1:k), ...
        'Xte',score_test_pca(:,1:k),'dim',k,'family','PCA');
end
for k = lda_ks
    reps{end+1} = struct('name',sprintf('LDA-%d',k),'Xtr',sc_tr_lda(:,1:k), ...
        'Xte',sc_te_lda(:,1:k),'dim',k,'family','LDA');
end

save(fullfile(out_dir, 'task8_preprocessing.mat'), ...
    'mu_norm','sigma_norm','coeff_pca','mu_pca','W_lda','mu_g','K_pre', ...
    'reps','idx_train','idx_test', '-v7.3');
fprintf('Preprocessing saved.\n\n');

n_folds = 5;
n_reps = length(reps);
sweep_epochs = 30;
sweep_batch = 128;

architectures = {[64],[128],[256],[512],[128,64],[256,128],[512,256],[256,128,64],[512,256,128]};
arch_names = {'64','128','256','512','128-64','256-128','512-256','256-128-64','512-256-128'};
activations = {'relu','tanh','leakyrelu'};
learning_rates = [0.001, 0.005, 0.01, 0.05];

%% ========== EXPERIMENT 1: INPUT REPRESENTATION SWEEP ==========
chk1 = fullfile(out_dir, 'task8_exp1.mat');
if isfile(chk1)
    fprintf('Loading Experiment 1 from checkpoint...\n');
    load(chk1);
else
    fprintf('========== EXPERIMENT 1: Input Representation Sweep ==========\n');
    exp1_mean = zeros(1, n_reps);
    exp1_std = zeros(1, n_reps);
    fixed_arch = [128, 64];
    fixed_act = 'relu';
    fixed_lr = 0.01;

    for r = 1:n_reps
        fprintf('  [%d/%d] %s ... ', r, n_reps, reps{r}.name);
        fa = kfold_mlp(reps{r}.Xtr, Y_train_num, Y_train_cat, ...
            reps{r}.dim, fixed_arch, fixed_act, fixed_lr, sweep_epochs, sweep_batch, n_folds, num_classes);
        exp1_mean(r) = mean(fa);
        exp1_std(r) = std(fa);
        fprintf('%.2f%% +/- %.2f%%\n', exp1_mean(r)*100, exp1_std(r)*100);
    end

    [~, best_rep] = max(exp1_mean);
    save(chk1, 'exp1_mean','exp1_std','best_rep');
    fprintf('>> Best representation: %s (%.2f%%)\n', reps{best_rep}.name, exp1_mean(best_rep)*100);
    fprintf('Experiment 1 checkpoint saved.\n\n');
end

fig1 = figure('Name','Exp1: Representation Sweep','Position',[50,50,1200,500]);
b1 = bar(exp1_mean*100);
b1.FaceColor = 'flat';
for i = 1:n_reps
    switch reps{i}.family
        case 'Raw', b1.CData(i,:) = [0.20 0.40 0.80];
        case 'PCA', b1.CData(i,:) = [0.20 0.70 0.35];
        case 'LDA', b1.CData(i,:) = [0.90 0.35 0.15];
    end
end
hold on;
errorbar(1:n_reps, exp1_mean*100, exp1_std*100, 'k.','LineWidth',1.2);
set(gca,'XTick',1:n_reps,'XTickLabel',cellfun(@(r)r.name,reps,'Uni',0),'XTickLabelRotation',45);
ylabel('5-Fold CV Accuracy (%)');
title('MLP Accuracy vs Input Representation');
grid on;
legend({'Raw','PCA','LDA'},'Location','southeast');
saveas(fig1, fullfile(out_dir, 'fig1_representation_sweep.png'));
saveas(fig1, fullfile(out_dir, 'fig1_representation_sweep.fig'));

%% ========== EXPERIMENT 2: ARCHITECTURE SWEEP ==========
chk2 = fullfile(out_dir, 'task8_exp2.mat');
if isfile(chk2)
    fprintf('Loading Experiment 2 from checkpoint...\n');
    load(chk2);
else
    fprintf('========== EXPERIMENT 2: Architecture Sweep ==========\n');
    X_sw = reps{best_rep}.Xtr;
    D_sw = reps{best_rep}.dim;

    n_arch = length(architectures);
    n_act = length(activations);
    n_lr = length(learning_rates);
    total = n_arch * n_act * n_lr;

    exp2_mean = zeros(n_arch, n_act, n_lr);
    exp2_std_val = zeros(n_arch, n_act, n_lr);

    cnt = 0;
    for a = 1:n_arch
        for ac = 1:n_act
            for l = 1:n_lr
                cnt = cnt + 1;
                fprintf('  [%d/%d] %s | %s | lr=%.4f ... ', cnt, total, ...
                    arch_names{a}, activations{ac}, learning_rates(l));
                fa = kfold_mlp(X_sw, Y_train_num, Y_train_cat, ...
                    D_sw, architectures{a}, activations{ac}, learning_rates(l), ...
                    sweep_epochs, sweep_batch, n_folds, num_classes);
                exp2_mean(a,ac,l) = mean(fa);
                exp2_std_val(a,ac,l) = std(fa);
                fprintf('%.2f%%\n', exp2_mean(a,ac,l)*100);
            end
        end
    end

    [best_val, best_lin] = max(exp2_mean(:));
    [ba, bac, bl] = ind2sub(size(exp2_mean), best_lin);
    save(chk2, 'exp2_mean','exp2_std_val','ba','bac','bl');
    fprintf('>> Best: %s | %s | lr=%.4f -> %.2f%%\n', ...
        arch_names{ba}, activations{bac}, learning_rates(bl), best_val*100);
    fprintf('Experiment 2 checkpoint saved.\n\n');
end

n_arch = length(architectures);
n_act = length(activations);
n_lr = length(learning_rates);

for ac = 1:n_act
    fig_hm = figure('Name',sprintf('Exp2: Heatmap - %s',activations{ac}),'Position',[100+ac*50,100,900,500]);
    hm = squeeze(exp2_mean(:,ac,:)) * 100;
    imagesc(hm);
    colormap(parula); colorbar;
    set(gca,'XTick',1:n_lr,'XTickLabel',arrayfun(@(x)sprintf('%.4f',x),learning_rates,'Uni',0));
    set(gca,'YTick',1:n_arch,'YTickLabel',arch_names);
    xlabel('Learning Rate'); ylabel('Architecture');
    title(sprintf('5-Fold CV Accuracy (%%) — Activation: %s',activations{ac}));
    for i = 1:n_arch
        for j = 1:n_lr
            text(j,i,sprintf('%.1f',hm(i,j)),'HorizontalAlignment','center','FontSize',7);
        end
    end
    saveas(fig_hm, fullfile(out_dir, sprintf('fig2_heatmap_%s.png',activations{ac})));
    saveas(fig_hm, fullfile(out_dir, sprintf('fig2_heatmap_%s.fig',activations{ac})));
end

all_configs = [];
config_labels = {};
for a = 1:n_arch
    for ac = 1:n_act
        for l = 1:n_lr
            all_configs(end+1) = exp2_mean(a,ac,l);
            config_labels{end+1} = sprintf('%s|%s|%.3f',arch_names{a},activations{ac},learning_rates(l));
        end
    end
end
[sorted_acc, sort_i] = sort(all_configs, 'descend');
top_n = min(20, length(sorted_acc));

fig_top = figure('Name','Exp2: Top Configurations','Position',[200,200,1000,600]);
barh(flip(sorted_acc(1:top_n)*100));
set(gca,'YTickLabel',flip(config_labels(sort_i(1:top_n))));
xlabel('5-Fold CV Accuracy (%)');
title('Top 20 MLP Configurations');
grid on;
saveas(fig_top, fullfile(out_dir, 'fig2_top20_configs.png'));
saveas(fig_top, fullfile(out_dir, 'fig2_top20_configs.fig'));

%% ========== EXPERIMENT 3: FINAL EVALUATION ==========
chk3 = fullfile(out_dir, 'task8_exp3.mat');
if isfile(chk3)
    fprintf('Loading Experiment 3 from checkpoint...\n');
    load(chk3);
else
    fprintf('========== EXPERIMENT 3: Final Evaluation ==========\n');

    X_fin_tr = reps{best_rep}.Xtr;
    X_fin_te = reps{best_rep}.Xte;
    D_fin = reps{best_rep}.dim;

    layers_fin = build_mlp_layers(D_fin, architectures{ba}, activations{bac}, num_classes);
    opts_fin = trainingOptions('adam', ...
        'MaxEpochs', 80, ...
        'MiniBatchSize', sweep_batch, ...
        'InitialLearnRate', learning_rates(bl), ...
        'ValidationData', {X_fin_te, Y_test_cat}, ...
        'ValidationFrequency', 30, ...
        'Shuffle', 'every-epoch', ...
        'Plots', 'none', ...
        'Verbose', false, ...
        'OutputNetwork', 'best-validation-loss');

    tic;
    [net_mlp, info_mlp] = trainNetwork(X_fin_tr, Y_train_cat, layers_fin, opts_fin);
    mlp_train_time = toc;

    Y_pred_cat = classify(net_mlp, X_fin_te);
    Y_pred_num = str2double(cellstr(Y_pred_cat));
    final_acc = mean(Y_pred_cat == Y_test_cat);
    fprintf('Hold-out accuracy: %.2f%%\n', final_acc * 100);
    fprintf('Training time: %.1f s\n\n', mlp_train_time);

    cm = confusionmat(Y_test_num, Y_pred_num);
    precision_pc = diag(cm) ./ max(sum(cm,1)',1);
    recall_pc = diag(cm) ./ max(sum(cm,2),1);
    f1_pc = 2*(precision_pc .* recall_pc) ./ max(precision_pc + recall_pc, eps);
    macro_f1 = mean(f1_pc);

    fprintf('Per-class F1:\n');
    for c = 0:9
        fprintf('  Digit %d: P=%.3f R=%.3f F1=%.3f\n', c, precision_pc(c+1), recall_pc(c+1), f1_pc(c+1));
    end
    fprintf('Macro F1: %.4f\n\n', macro_f1);

    save(chk3, 'net_mlp','info_mlp','mlp_train_time','Y_pred_cat','Y_pred_num', ...
        'final_acc','cm','precision_pc','recall_pc','f1_pc','macro_f1', '-v7.3');
    fprintf('Experiment 3 checkpoint saved.\n\n');
end

fig_tc = figure('Name','Exp3: Training Curves','Position',[250,100,1000,450]);
subplot(1,2,1);
plot(info_mlp.TrainingLoss,'b','LineWidth',0.8); hold on;
val_iters = find(~isnan(info_mlp.ValidationLoss));
plot(val_iters, info_mlp.ValidationLoss(val_iters),'r-o','LineWidth',1.2,'MarkerSize',4);
xlabel('Iteration'); ylabel('Loss');
title('Training & Validation Loss');
legend('Train','Validation','Location','northeast'); grid on;
subplot(1,2,2);
plot(info_mlp.TrainingAccuracy,'b','LineWidth',0.8); hold on;
plot(val_iters, info_mlp.ValidationAccuracy(val_iters),'r-o','LineWidth',1.2,'MarkerSize',4);
xlabel('Iteration'); ylabel('Accuracy (%)');
title('Training & Validation Accuracy');
legend('Train','Validation','Location','southeast'); grid on;
saveas(fig_tc, fullfile(out_dir, 'fig3_training_curves.png'));
saveas(fig_tc, fullfile(out_dir, 'fig3_training_curves.fig'));

fig_cm = figure('Name','Exp3: Confusion Matrix','Position',[300,100,600,550]);
confusionchart(Y_test_cat, Y_pred_cat, 'Normalization','row-normalized','RowSummary','row-normalized');
title(sprintf('MLP Confusion Matrix — Accuracy: %.2f%%', final_acc*100));
saveas(fig_cm, fullfile(out_dir, 'fig3_confusion_matrix.png'));
saveas(fig_cm, fullfile(out_dir, 'fig3_confusion_matrix.fig'));

fig_f1 = figure('Name','Exp3: Per-class F1','Position',[350,150,800,400]);
bar(0:9, f1_pc);
xlabel('Digit'); ylabel('F1 Score');
title(sprintf('MLP Per-class F1 — Macro F1: %.4f', macro_f1));
set(gca,'XTick',0:9);
ylim([min(f1_pc)*0.9, 1.0]); grid on;
saveas(fig_f1, fullfile(out_dir, 'fig3_f1_per_class.png'));
saveas(fig_f1, fullfile(out_dir, 'fig3_f1_per_class.fig'));

%% ========== SAVE FINAL BUNDLE FOR TASK 9 ==========
mlp_results.accuracy = final_acc;
mlp_results.macro_f1 = macro_f1;
mlp_results.f1_pc = f1_pc;
mlp_results.precision_pc = precision_pc;
mlp_results.recall_pc = recall_pc;
mlp_results.cm = cm;
mlp_results.train_time = mlp_train_time;
mlp_results.best_rep_name = reps{best_rep}.name;
mlp_results.best_arch = arch_names{ba};
mlp_results.best_act = activations{bac};
mlp_results.best_lr = learning_rates(bl);
mlp_results.exp1_mean = exp1_mean;
mlp_results.exp1_std = exp1_std;
mlp_results.exp1_names = cellfun(@(r)r.name,reps,'Uni',0);

save('task8_results.mat', 'mlp_results', 'net_mlp', 'mu_norm', 'sigma_norm', ...
    'coeff_pca', 'mu_pca', 'W_lda', 'mu_g', 'K_pre', ...
    'reps', 'best_rep', 'ba', 'bac', 'bl', ...
    'architectures', 'activations', 'learning_rates', '-v7.3');

%% ========== TEST SET PREDICTION ==========
%{
load('Test_numbers.mat');
X_test_raw = double(Test_numbers.image);
X_test_norm = ((X_test_raw - mu_norm) ./ sigma_norm)';

switch reps{best_rep}.family
    case 'Raw'
        X_test_input = X_test_norm;
    case 'PCA'
        X_test_input = (X_test_norm - mu_pca) * coeff_pca(:, 1:reps{best_rep}.dim);
    case 'LDA'
        X_test_pca = (X_test_norm - mu_pca) * coeff_pca(:, 1:K_pre);
        X_test_input = (X_test_pca - mu_g) * W_lda(:, 1:reps{best_rep}.dim);
end

Y_pred_test = classify(net_mlp, X_test_input);
class = str2double(cellstr(Y_pred_test))';
name = {'Name1', 'Name2', 'Name3'};
save('Group##_mlp.mat', 'name', 'class');
%}

fprintf('Task 8 complete. All outputs in ./%s/\n', out_dir);

%% ========== LOCAL FUNCTIONS ==========

function layers = build_mlp_layers(D, hidden, act, nc)
    layers = [featureInputLayer(D, 'Normalization', 'none')];
    for i = 1:length(hidden)
        layers = [layers; fullyConnectedLayer(hidden(i))];
        switch act
            case 'relu',      layers = [layers; reluLayer];
            case 'tanh',      layers = [layers; tanhLayer];
            case 'leakyrelu', layers = [layers; leakyReluLayer(0.2)];
        end
    end
    layers = [layers; fullyConnectedLayer(nc); softmaxLayer; classificationLayer];
end

function fa = kfold_mlp(X, Ynum, Ycat, D, arch, act, lr, ep, bs, nf, nc)
    cv = cvpartition(Ynum, 'KFold', nf);
    fa = zeros(1, nf);
    for f = 1:nf
        Xtr = X(training(cv,f), :);
        Ytr = Ycat(training(cv,f));
        Xv  = X(test(cv,f), :);
        Yv  = Ycat(test(cv,f));
        layers = build_mlp_layers(D, arch, act, nc);
        opts = trainingOptions('adam', ...
            'MaxEpochs', ep, ...
            'MiniBatchSize', bs, ...
            'InitialLearnRate', lr, ...
            'Shuffle', 'every-epoch', ...
            'Verbose', false);
        net = trainNetwork(Xtr, Ytr, layers, opts);
        Yp = classify(net, Xv);
        fa(f) = mean(Yp == Yv);
    end
end