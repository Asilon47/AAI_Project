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
    imshow(imagesTrain(:,:,1,i)', []);
    title(char(valuesTrain(i)), 'FontSize', 8);
end
sgtitle('Training Samples');

layers = [
    imageInputLayer([28 28 1])

    convolution2dLayer(3, 32, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    convolution2dLayer(3, 32, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    dropoutLayer(0.25)

    convolution2dLayer(3, 64, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    convolution2dLayer(3, 64, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    dropoutLayer(0.25)

    convolution2dLayer(3, 128, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    dropoutLayer(0.25)

    fullyConnectedLayer(256)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.5)
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer
];

augmenter = imageDataAugmenter( ...
    'RandRotation', [-10 10], ...
    'RandXTranslation', [-3 3], ...
    'RandYTranslation', [-3 3], ...
    'RandXScale', [0.95 1.05], ...
    'RandYScale', [0.95 1.05]);

augimdsTrain = augmentedImageDatastore([28 28 1], imagesTrain, valuesTrain, ...
    'DataAugmentation', augmenter);

figure;
batch = read(augimdsTrain);
for i = 1:min(20, size(batch, 1))
    subplot(2,10,i);
    imshow(batch.input{i}, []);
    title(char(batch.response(i)), 'FontSize', 8);
end
sgtitle('Augmented Samples');
reset(augimdsTrain);

options = trainingOptions('adam', ...
    'ExecutionEnvironment', 'cpu', ...
    'InitialLearnRate', 0.001, ...
    'MaxEpochs', 30, ...
    'MiniBatchSize', 64, ...
    'Shuffle', 'every-epoch', ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.5, ...
    'LearnRateDropPeriod', 5, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

numModels = 3;
modelsCNN = cell(numModels, 1);
disp('Starting CNN Ensemble Training...');

for i = 1:numModels
    fprintf('Training Model %d of %d...\n', i, numModels);
    modelsCNN{i} = trainNetwork(augimdsTrain, layers, options);
end

totalProbs = zeros(size(imagesTest, 4), 10);
for i = 1:numModels
    totalProbs = totalProbs + predict(modelsCNN{i}, imagesTest);
end
[~, YPred] = max(totalProbs, [], 2);

accuracy = mean(YPred == double(valuesTest));
fprintf('CNN Accuracy: %.2f%%\n', accuracy * 100);

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
title(sprintf('CNN Confusion Matrix — %.2f%%', accuracy*100));

figure;
bar(0:9, f1, 'FaceColor', [0.8 0.3 0.2]);
xlabel('Digit'); ylabel('F1');
title('CNN Per-class F1'); ylim([0.9 1]); grid on;

% load("Test_numbers_HW1.mat");
% X_test_img = reshape(double(Test_numbers.image), [28, 28, 1, 10000]) / 255;
% 
% totalProbsTest = zeros(10000, 10);
% for i = 1:numModels
%     totalProbsTest = totalProbsTest + predict(modelsCNN{i}, X_test_img);
% end
% [~, testPred] = max(totalProbsTest, [], 2);
% class = testPred' - 1;
% name = {'Ruth', 'Jose', 'Asil'};
% 
% fprintf('\nTest distribution:\n');
% for d = 0:9; fprintf('  %d: %d\n', d, sum(class==d)); end
% 
% figure;
% rng(1); ri = randperm(10000, 30);
% for i = 1:30
%     subplot(3,10,i);
%     imshow(X_test_img(:,:,1,ri(i))', []);
%     title(num2str(class(ri(i))), 'FontSize', 8, 'Color', 'r');
% end
% sgtitle('CNN Test Predictions');
% 
% save('Group05_dln.mat', 'name', 'class');
% fprintf('Saved Group05_dln.mat\n');