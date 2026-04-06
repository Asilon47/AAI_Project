clear; close all; clc;

load("digits.mat");
trainPercentage = 0.8;
images = reshape(digits.image, [28, 28, 1, 10000]) / 255;
labels = categorical(digits.label);
numImages = size(images, 4);
idx = randperm(numImages);
numTrain = round(trainPercentage * numImages);
trainIdx = idx(1:numTrain);
testIdx = idx(numTrain+1:end);
imagesTrain = images(:,:,:,trainIdx);
valuesTrain = labels(trainIdx);
imagesTest = images(:,:,:,testIdx);
valuesTest = labels(testIdx);

figure;
for i = 1:20
    subplot(2,10,i);
    imshow(imagesTrain(:,:,1,i), []);
    title(char(valuesTrain(i)), 'FontSize', 8);
end
sgtitle('Training Samples');

augmenter = imageDataAugmenter( ...
    'RandRotation', [-10 10], ...
    'RandXTranslation', [-3 3], ...
    'RandYTranslation', [-3 3], ...
    'RandXScale', [0.95 1.05], ...
    'RandYScale', [0.95 1.05]);

augDS = augmentedImageDatastore([28 28 1], imagesTrain, valuesTrain(:), ...
    'DataAugmentation', augmenter);

figure;
batch = read(augDS);
for i = 1:min(20, size(batch, 1))
    subplot(2,10,i);
    imshow(batch.input{i}, []);
    title(char(batch.response(i)), 'FontSize', 8);
end
sgtitle('Augmented Samples');

numAugPasses = 2;
allImages = imagesTrain;
allLabels = valuesTrain(:);
for p = 1:numAugPasses
    reset(augDS);
    augTable = readall(augDS);
    augImgs = cat(4, augTable.input{:});
    allImages = cat(4, allImages, augImgs);
    allLabels = [allLabels; augTable.response(:)];
end
fprintf('Original: %d | After augmentation: %d\n', numTrain, length(allLabels));

pcaDim = 150;
allFlat = reshape(allImages, [], size(allImages, 4))';
testFlat = reshape(imagesTest, [], size(imagesTest, 4))';
muTrain = mean(allFlat, 1);
[coeff, scoreAll] = pca(allFlat);
trainPCA = scoreAll(:, 1:pcaDim);
testPCA = (testFlat - muTrain) * coeff(:, 1:pcaDim);
fprintf('PCA: 784 -> %d dims\n', pcaDim);

layers = [
    featureInputLayer(pcaDim)
    fullyConnectedLayer(256)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.2)
    fullyConnectedLayer(128)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.2)
    fullyConnectedLayer(64)
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer
];

options = trainingOptions('adam', ...
    'ExecutionEnvironment', 'cpu', ...
    'InitialLearnRate', 0.001, ...
    'MaxEpochs', 50, ...
    'MiniBatchSize', 128, ...
    'Shuffle', 'every-epoch', ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.5, ...
    'LearnRateDropPeriod', 10, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

numModels = 5;
modelsMLP = cell(numModels, 1);
disp('Starting MLP Ensemble Training...');

for i = 1:numModels
    fprintf('Training Model %d of %d...\n', i, numModels);
    shuffleIdx = randperm(size(trainPCA, 1));
    modelsMLP{i} = trainNetwork(trainPCA(shuffleIdx,:), ...
        allLabels(shuffleIdx), layers, options);
end

totalProbs = zeros(size(imagesTest, 4), 10);
for i = 1:numModels
    totalProbs = totalProbs + predict(modelsMLP{i}, testPCA);
end
[~, YPred] = max(totalProbs, [], 2);

accuracy = mean(YPred == double(valuesTest));
fprintf('MLP Accuracy: %.2f%%\n', accuracy * 100);

trueLabels = double(valuesTest);
cm = confusionmat(trueLabels, YPred);
numClasses = size(cm, 1);
precision = zeros(numClasses, 1);
recall = zeros(numClasses, 1);
f1 = zeros(numClasses, 1);
for i = 1:numClasses
    tp = cm(i,i);
    fp = sum(cm(:,i)) - tp;
    fn = sum(cm(i,:)) - tp;
    if (tp+fp) > 0; precision(i) = tp/(tp+fp); end
    if (tp+fn) > 0; recall(i) = tp/(tp+fn); end
    if (precision(i)+recall(i)) > 0
        f1(i) = 2*precision(i)*recall(i)/(precision(i)+recall(i));
    end
end
fprintf('Macro Precision: %.4f\n', mean(precision));
fprintf('Macro Recall:    %.4f\n', mean(recall));
fprintf('Macro F1:        %.4f\n', mean(f1));

figure;
confusionchart(trueLabels, YPred);
title(sprintf('MLP Confusion Matrix — %.2f%%', accuracy*100));

figure;
bar(0:9, f1, 'FaceColor', [0.2 0.5 0.8]);
xlabel('Digit'); ylabel('F1');
title('MLP Per-class F1'); ylim([0.85 1]); grid on;

load("Test_numbers_HW1.mat");
X_test_raw = double(Test_numbers.image);
X_test_pca = (X_test_raw'/255 - muTrain) * coeff(:, 1:pcaDim);

totalProbsTest = zeros(10000, 10);
for i = 1:numModels
    totalProbsTest = totalProbsTest + predict(modelsMLP{i}, X_test_pca);
end
[~, testPred] = max(totalProbsTest, [], 2);
class = testPred' - 1;
name = {'Ruth', 'Jose', 'Asil'};

fprintf('\nTest distribution:\n');
for d = 0:9; fprintf('  %d: %d\n', d, sum(class==d)); end

figure;
rng(1); ri = randperm(10000, 30);
for i = 1:30
    subplot(3,10,i);
    imshow(reshape(X_test_raw(:,ri(i)),28,28)', []);
    title(num2str(class(ri(i))), 'FontSize', 8, 'Color', 'b');
end
sgtitle('MLP Test Predictions');

save('Group05_mlp.mat', 'name', 'class');
fprintf('Saved Group05_mlp.mat\n');