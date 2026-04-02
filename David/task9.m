clear; clc; close all;
rng(42, 'twister');

out_dir = 'task9_outputs';
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

load('digits.mat');
X_raw = double(digits.image);
labels = digits.label(:);
N = length(labels);
num_classes = 10;

mu_norm = mean(X_raw, 2);
sigma_norm = std(X_raw, 0, 2);
sigma_norm(sigma_norm == 0) = 1;
Xn = (X_raw - mu_norm) ./ sigma_norm;

X_images = permute(reshape(Xn, 28, 28, 1, []), [2 1 3 4]);

labels_cat = categorical(labels);

cv_hold = cvpartition(labels, 'HoldOut', 0.2);
idx_train = training(cv_hold);
idx_test = test(cv_hold);

X_train_img = X_images(:,:,:,idx_train);
Y_train_cat = labels_cat(idx_train);
Y_train_num = labels(idx_train);
X_test_img = X_images(:,:,:,idx_test);
Y_test_cat = labels_cat(idx_test);
Y_test_num = labels(idx_test);

n_folds = 5;
sweep_epochs = 20;
sweep_batch = 128;
sweep_lr = 0.001;

num_blocks_list = [1, 2, 3, 4];
filters_list = [8, 16, 32, 64];

n_blocks = length(num_blocks_list);
n_filters = length(filters_list);

%% ========== EXPERIMENT 1: DEPTH / FILTER SWEEP ==========
chk1 = fullfile(out_dir, 'task9_exp1.mat');
if isfile(chk1)
    fprintf('Loading Experiment 1 from checkpoint...\n');
    load(chk1);
else
    fprintf('========== EXPERIMENT 1: CNN Depth / Filter Sweep ==========\n');
    exp1_mean = zeros(n_blocks, n_filters);
    exp1_std = zeros(n_blocks, n_filters);

    total = n_blocks * n_filters;
    cnt = 0;
    for bi = 1:n_blocks
        for fi = 1:n_filters
            cnt = cnt + 1;
            fprintf('  [%d/%d] blocks=%d filters=%d ... ', cnt, total, ...
                num_blocks_list(bi), filters_list(fi));
            fa = kfold_cnn(X_train_img, Y_train_num, Y_train_cat, ...
                num_blocks_list(bi), filters_list(fi), 0, sweep_lr, 'adam', ...
                sweep_epochs, sweep_batch, n_folds, num_classes, false);
            exp1_mean(bi,fi) = mean(fa);
            exp1_std(bi,fi) = std(fa);
            fprintf('%.2f%% +/- %.2f%%\n', exp1_mean(bi,fi)*100, exp1_std(bi,fi)*100);
        end
    end

    [best_acc1, best_lin] = max(exp1_mean(:));
    [bb, bf] = ind2sub(size(exp1_mean), best_lin);
    save(chk1, 'exp1_mean','exp1_std','bb','bf');
    fprintf('>> Best: blocks=%d filters=%d -> %.2f%%\n', ...
        num_blocks_list(bb), filters_list(bf), best_acc1*100);
    fprintf('Experiment 1 checkpoint saved.\n\n');
end

best_blocks = num_blocks_list(bb);
best_filters = filters_list(bf);

fig1 = figure('Name','Exp1: Depth/Filter Heatmap','Position',[50,50,700,500]);
imagesc(exp1_mean*100);
colormap(parula); colorbar;
set(gca,'XTick',1:n_filters,'XTickLabel',arrayfun(@num2str,filters_list,'Uni',0));
set(gca,'YTick',1:n_blocks,'YTickLabel',arrayfun(@num2str,num_blocks_list,'Uni',0));
xlabel('Filters per Block'); ylabel('Number of Conv Blocks');
title('CNN 5-Fold CV Accuracy (%) — Depth vs Filters');
for i = 1:n_blocks
    for j = 1:n_filters
        text(j,i,sprintf('%.1f',exp1_mean(i,j)*100),'HorizontalAlignment','center','FontSize',9);
    end
end
saveas(fig1, fullfile(out_dir, 'fig1_depth_filter_heatmap.png'));
saveas(fig1, fullfile(out_dir, 'fig1_depth_filter_heatmap.fig'));

%% ========== EXPERIMENT 2: REGULARIZATION / TRAINING ABLATION ==========
chk2 = fullfile(out_dir, 'task9_exp2.mat');

dropout_rates = [0, 0.25, 0.5];
optimizers = {'sgdm', 'adam'};
use_augment = [false, true];

n_drop = length(dropout_rates);
n_opt = length(optimizers);
n_aug = length(use_augment);

if isfile(chk2)
    fprintf('Loading Experiment 2 from checkpoint...\n');
    load(chk2);
else
    fprintf('========== EXPERIMENT 2: Regularization Ablation ==========\n');
    total2 = n_drop * n_opt * n_aug;
    exp2_mean = zeros(n_drop, n_opt, n_aug);
    exp2_std_val = zeros(n_drop, n_opt, n_aug);

    cnt = 0;
    for di = 1:n_drop
        for oi = 1:n_opt
            for ai = 1:n_aug
                cnt = cnt + 1;
                aug_str = 'no-aug';
                if use_augment(ai), aug_str = 'aug'; end
                fprintf('  [%d/%d] drop=%.2f opt=%s %s ... ', cnt, total2, ...
                    dropout_rates(di), optimizers{oi}, aug_str);
                fa = kfold_cnn(X_train_img, Y_train_num, Y_train_cat, ...
                    best_blocks, best_filters, dropout_rates(di), sweep_lr, ...
                    optimizers{oi}, sweep_epochs, sweep_batch, n_folds, num_classes, use_augment(ai));
                exp2_mean(di,oi,ai) = mean(fa);
                exp2_std_val(di,oi,ai) = std(fa);
                fprintf('%.2f%%\n', exp2_mean(di,oi,ai)*100);
            end
        end
    end

    [best_acc2, best_lin2] = max(exp2_mean(:));
    [bd, bo, bau] = ind2sub(size(exp2_mean), best_lin2);
    save(chk2, 'exp2_mean','exp2_std_val','bd','bo','bau');
    fprintf('>> Best reg: drop=%.2f opt=%s aug=%d -> %.2f%%\n', ...
        dropout_rates(bd), optimizers{bo}, use_augment(bau), best_acc2*100);
    fprintf('Experiment 2 checkpoint saved.\n\n');
end

fig2 = figure('Name','Exp2: Regularization Ablation','Position',[100,100,1100,500]);
group_names = {};
group_acc = [];
group_err = [];
for di = 1:n_drop
    for oi = 1:n_opt
        for ai = 1:n_aug
            aug_str = 'no-aug';
            if use_augment(ai), aug_str = 'aug'; end
            group_names{end+1} = sprintf('d=%.2f\n%s\n%s', dropout_rates(di), optimizers{oi}, aug_str);
            group_acc(end+1) = exp2_mean(di,oi,ai)*100;
            group_err(end+1) = exp2_std_val(di,oi,ai)*100;
        end
    end
end
bar(group_acc); hold on;
errorbar(1:length(group_acc), group_acc, group_err, 'k.','LineWidth',1);
set(gca,'XTick',1:length(group_names),'XTickLabel',group_names,'XTickLabelRotation',45);
ylabel('5-Fold CV Accuracy (%)');
title(sprintf('Regularization Ablation (blocks=%d, filters=%d)', best_blocks, best_filters));
grid on;
saveas(fig2, fullfile(out_dir, 'fig2_regularization_ablation.png'));
saveas(fig2, fullfile(out_dir, 'fig2_regularization_ablation.fig'));

%% ========== EXPERIMENT 3: FINAL CNN EVALUATION ==========
chk3 = fullfile(out_dir, 'task9_exp3.mat');
if isfile(chk3)
    fprintf('Loading Experiment 3 from checkpoint...\n');
    load(chk3);
else
    fprintf('========== EXPERIMENT 3: Final CNN Evaluation ==========\n');

    layers_fin = build_cnn_layers(best_blocks, best_filters, dropout_rates(bd), num_classes);

    opts_fin = trainingOptions(optimizers{bo}, ...
        'MaxEpochs', 50, ...
        'MiniBatchSize', sweep_batch, ...
        'InitialLearnRate', sweep_lr, ...
        'ValidationData', {X_test_img, Y_test_cat}, ...
        'ValidationFrequency', 50, ...
        'Shuffle', 'every-epoch', ...
        'Plots', 'none', ...
        'Verbose', false, ...
        'OutputNetwork', 'best-validation-loss');

    if use_augment(bau)
        augmenter = imageDataAugmenter( ...
            'RandRotation', [-10 10], ...
            'RandXTranslation', [-2 2], ...
            'RandYTranslation', [-2 2]);
        auds = augmentedImageDatastore([28 28 1], X_train_img, Y_train_cat, ...
            'DataAugmentation', augmenter);
        tic;
        [net_cnn, info_cnn] = trainNetwork(auds, layers_fin, opts_fin);
        cnn_train_time = toc;
    else
        tic;
        [net_cnn, info_cnn] = trainNetwork(X_train_img, Y_train_cat, layers_fin, opts_fin);
        cnn_train_time = toc;
    end

    Y_pred_cnn = classify(net_cnn, X_test_img);
    Y_pred_cnn_num = str2double(cellstr(Y_pred_cnn));
    cnn_acc = mean(Y_pred_cnn == Y_test_cat);
    fprintf('Hold-out accuracy: %.2f%%\n', cnn_acc*100);
    fprintf('Training time: %.1f s\n\n', cnn_train_time);

    cm_cnn = confusionmat(Y_test_num, Y_pred_cnn_num);
    prec_cnn = diag(cm_cnn) ./ max(sum(cm_cnn,1)',1);
    rec_cnn = diag(cm_cnn) ./ max(sum(cm_cnn,2),1);
    f1_cnn = 2*(prec_cnn .* rec_cnn) ./ max(prec_cnn + rec_cnn, eps);
    macro_f1_cnn = mean(f1_cnn);

    fprintf('Per-class F1:\n');
    for c = 0:9
        fprintf('  Digit %d: P=%.3f R=%.3f F1=%.3f\n', c, prec_cnn(c+1), rec_cnn(c+1), f1_cnn(c+1));
    end
    fprintf('Macro F1: %.4f\n\n', macro_f1_cnn);

    save(chk3, 'net_cnn','info_cnn','cnn_train_time','Y_pred_cnn','Y_pred_cnn_num', ...
        'cnn_acc','cm_cnn','prec_cnn','rec_cnn','f1_cnn','macro_f1_cnn', '-v7.3');
    fprintf('Experiment 3 checkpoint saved.\n\n');
end

fig3a = figure('Name','Exp3: CNN Training Curves','Position',[150,100,1000,450]);
subplot(1,2,1);
plot(info_cnn.TrainingLoss,'b','LineWidth',0.8); hold on;
vi = find(~isnan(info_cnn.ValidationLoss));
plot(vi, info_cnn.ValidationLoss(vi),'r-o','LineWidth',1.2,'MarkerSize',4);
xlabel('Iteration'); ylabel('Loss');
title('CNN Training & Validation Loss');
legend('Train','Validation','Location','northeast'); grid on;
subplot(1,2,2);
plot(info_cnn.TrainingAccuracy,'b','LineWidth',0.8); hold on;
plot(vi, info_cnn.ValidationAccuracy(vi),'r-o','LineWidth',1.2,'MarkerSize',4);
xlabel('Iteration'); ylabel('Accuracy (%)');
title('CNN Training & Validation Accuracy');
legend('Train','Validation','Location','southeast'); grid on;
saveas(fig3a, fullfile(out_dir, 'fig3_training_curves.png'));
saveas(fig3a, fullfile(out_dir, 'fig3_training_curves.fig'));

fig3b = figure('Name','Exp3: CNN Confusion Matrix','Position',[200,100,600,550]);
confusionchart(Y_test_cat, Y_pred_cnn, 'Normalization','row-normalized','RowSummary','row-normalized');
title(sprintf('CNN Confusion Matrix — Accuracy: %.2f%%', cnn_acc*100));
saveas(fig3b, fullfile(out_dir, 'fig3_confusion_matrix.png'));
saveas(fig3b, fullfile(out_dir, 'fig3_confusion_matrix.fig'));

fig3c = figure('Name','Exp3: CNN Per-class F1','Position',[250,150,800,400]);
bar(0:9, f1_cnn);
xlabel('Digit'); ylabel('F1 Score');
title(sprintf('CNN Per-class F1 — Macro F1: %.4f', macro_f1_cnn));
set(gca,'XTick',0:9);
ylim([min(f1_cnn)*0.9, 1.0]); grid on;
saveas(fig3c, fullfile(out_dir, 'fig3_f1_per_class.png'));
saveas(fig3c, fullfile(out_dir, 'fig3_f1_per_class.fig'));

%% ========== EXPERIMENT 4: MLP vs CNN COMPARISON ==========
fprintf('========== EXPERIMENT 4: MLP vs CNN Comparison ==========\n');

if isfile('task8_results.mat')
    loaded = load('task8_results.mat', 'mlp_results');
    mlp = loaded.mlp_results;
    has_mlp = true;
    fprintf('MLP accuracy: %.2f%% | CNN accuracy: %.2f%%\n', mlp.accuracy*100, cnn_acc*100);
    fprintf('MLP macro-F1: %.4f | CNN macro-F1: %.4f\n', mlp.macro_f1, macro_f1_cnn);
    fprintf('MLP train time: %.1fs | CNN train time: %.1fs\n\n', mlp.train_time, cnn_train_time);
else
    has_mlp = false;
    fprintf('task8_results.mat not found — skipping MLP comparison plots.\n\n');
end

if has_mlp
    fig4a = figure('Name','Exp4: MLP vs CNN Per-digit','Position',[300,100,900,450]);
    bar(0:9, [mlp.f1_pc, f1_cnn]);
    xlabel('Digit'); ylabel('F1 Score');
    legend('MLP','CNN','Location','southeast');
    title('Per-digit F1: MLP vs CNN');
    set(gca,'XTick',0:9); grid on;
    saveas(fig4a, fullfile(out_dir, 'fig4_perdigit_f1_comparison.png'));
    saveas(fig4a, fullfile(out_dir, 'fig4_perdigit_f1_comparison.fig'));

    fig4b = figure('Name','Exp4: Confusion Difference','Position',[350,100,600,550]);
    cm_mlp_norm = mlp.cm ./ max(sum(mlp.cm,2),1);
    cm_cnn_norm = cm_cnn ./ max(sum(cm_cnn,2),1);
    diff_cm = cm_cnn_norm - cm_mlp_norm;
    imagesc(diff_cm); colormap(coolwarm_map()); colorbar;
    caxis([-max(abs(diff_cm(:))), max(abs(diff_cm(:)))]);
    set(gca,'XTick',1:10,'XTickLabel',0:9,'YTick',1:10,'YTickLabel',0:9);
    xlabel('Predicted'); ylabel('True');
    title('Confusion Difference (CNN - MLP): blue = CNN better');
    saveas(fig4b, fullfile(out_dir, 'fig4_confusion_difference.png'));
    saveas(fig4b, fullfile(out_dir, 'fig4_confusion_difference.fig'));

    fig4c = figure('Name','Exp4: Summary','Position',[400,100,600,400]);
    metrics = [mlp.accuracy, cnn_acc; mlp.macro_f1, macro_f1_cnn];
    bar(metrics);
    set(gca,'XTickLabel',{'Accuracy','Macro F1'});
    legend('MLP','CNN','Location','southeast');
    title('MLP vs CNN: Overall Metrics');
    ylim([min(metrics(:))*0.9, 1.0]); grid on;
    saveas(fig4c, fullfile(out_dir, 'fig4_summary_metrics.png'));
    saveas(fig4c, fullfile(out_dir, 'fig4_summary_metrics.fig'));

    fig4d = figure('Name','Exp4: Time vs Accuracy','Position',[450,150,500,400]);
    scatter(mlp.train_time, mlp.accuracy*100, 120, 'b', 'filled'); hold on;
    scatter(cnn_train_time, cnn_acc*100, 120, 'r', 'filled');
    text(mlp.train_time, mlp.accuracy*100+0.3, 'MLP','HorizontalAlignment','center');
    text(cnn_train_time, cnn_acc*100+0.3, 'CNN','HorizontalAlignment','center');
    xlabel('Training Time (s)'); ylabel('Accuracy (%)');
    title('Efficiency: Training Time vs Accuracy');
    grid on;
    saveas(fig4d, fullfile(out_dir, 'fig4_time_vs_accuracy.png'));
    saveas(fig4d, fullfile(out_dir, 'fig4_time_vs_accuracy.fig'));
end

%% ========== SAVE FINAL RESULTS ==========
cnn_results.accuracy = cnn_acc;
cnn_results.macro_f1 = macro_f1_cnn;
cnn_results.f1_pc = f1_cnn;
cnn_results.precision_pc = prec_cnn;
cnn_results.recall_pc = rec_cnn;
cnn_results.cm = cm_cnn;
cnn_results.train_time = cnn_train_time;
cnn_results.best_blocks = best_blocks;
cnn_results.best_filters = best_filters;
cnn_results.best_dropout = dropout_rates(bd);
cnn_results.best_optimizer = optimizers{bo};
cnn_results.best_augment = use_augment(bau);
cnn_results.exp1_mean = exp1_mean;
cnn_results.exp2_mean = exp2_mean;

save('task9_results.mat', 'cnn_results', 'net_cnn', 'mu_norm', 'sigma_norm', '-v7.3');

%% ========== TEST SET PREDICTION ==========
%{
load('Test_numbers.mat');
X_test_raw = double(Test_numbers.image);
X_test_norm = (X_test_raw - mu_norm) ./ sigma_norm;
X_test_images = permute(reshape(X_test_norm, 28, 28, 1, []), [2 1 3 4]);

Y_pred_test = classify(net_cnn, X_test_images);
class = str2double(cellstr(Y_pred_test))';
name = {'Name1', 'Name2', 'Name3'};
save('Group##_dln.mat', 'name', 'class');
%}

fprintf('Task 9 complete. All outputs in ./%s/\n', out_dir);

%% ========== LOCAL FUNCTIONS ==========

function layers = build_cnn_layers(nb, nf, drop, nc)
    layers = [imageInputLayer([28 28 1], 'Normalization', 'none')];
    for b = 1:nb
        layers = [layers;
            convolution2dLayer(3, nf, 'Padding', 'same')
            batchNormalizationLayer
            reluLayer
            maxPooling2dLayer(2, 'Stride', 2)];
    end
    if drop > 0
        layers = [layers; dropoutLayer(drop)];
    end
    layers = [layers;
        fullyConnectedLayer(128)
        reluLayer
        fullyConnectedLayer(nc)
        softmaxLayer
        classificationLayer];
end

function fa = kfold_cnn(X, Ynum, Ycat, nb, nf, drop, lr, opt, ep, bs, nfolds, nc, aug)
    cv = cvpartition(Ynum, 'KFold', nfolds);
    fa = zeros(1, nfolds);
    for f = 1:nfolds
        Xtr = X(:,:,:,training(cv,f));
        Ytr = Ycat(training(cv,f));
        Xv  = X(:,:,:,test(cv,f));
        Yv  = Ycat(test(cv,f));

        layers = build_cnn_layers(nb, nf, drop, nc);
        opts = trainingOptions(opt, ...
            'MaxEpochs', ep, ...
            'MiniBatchSize', bs, ...
            'InitialLearnRate', lr, ...
            'Shuffle', 'every-epoch', ...
            'Verbose', false);

        if aug
            augmenter = imageDataAugmenter( ...
                'RandRotation', [-10 10], ...
                'RandXTranslation', [-2 2], ...
                'RandYTranslation', [-2 2]);
            auds = augmentedImageDatastore([28 28 1], Xtr, Ytr, ...
                'DataAugmentation', augmenter);
            net = trainNetwork(auds, layers, opts);
        else
            net = trainNetwork(Xtr, Ytr, layers, opts);
        end
        Yp = classify(net, Xv);
        fa(f) = mean(Yp == Yv);
    end
end

function cmap = coolwarm_map()
    n = 128;
    r = [linspace(0.2,1,n), linspace(1,0.7,n)]';
    g = [linspace(0.2,1,n), linspace(1,0.2,n)]';
    b = [linspace(0.7,1,n), linspace(1,0.2,n)]';
    cmap = [r, g, b];
end