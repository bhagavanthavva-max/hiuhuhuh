clc; clear; close all;

%% Image Data Load
datasetPath = 'WildFire';

imds = imageDatastore(datasetPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

disp("Class Labels:");
disp(unique(imds.Labels));

labelCount = countEachLabel(imds);
disp(labelCount);

%% Plot
figure;
bar(labelCount.Count);
set(gca, 'XTickLabel', labelCount.Label);
title('Number of Images per Class');

figure;
perm = randperm(numel(imds.Files), 6);
for i = 1:6
    subplot(2,3,i);
    imshow(readimage(imds, perm(i)));
    title(string(imds.Labels(perm(i))));
end

%% Preprocessing of data
inputSize = [128 128];
imds.ReadFcn = @(filename)imresize(imread(filename), inputSize);

%% Data Split
[imdsTrain, imdsTest] = splitEachLabel(imds, 0.8, 'randomized');

%% Process of Data
numTrain = numel(imdsTrain.Files);
features = [];

for i = 1:numTrain
    img = readimage(imdsTrain, i);
    
    if ndims(img) == 2
        grayImg = img;
        imgRGB = cat(3, img, img, img);
    else
        grayImg = rgb2gray(img);
        imgRGB = img;
    end
    
    hog = extractHOGFeatures(grayImg);
 
    hsvImg = rgb2hsv(imgRGB);
    colorFeat = mean(reshape(hsvImg, [], 3));
    
    features(i,:) = [hog colorFeat];
end

trainLabels = imdsTrain.Labels;

%% Data Normalization
featMean = mean(features, 1);
featStd  = std(features, 0, 1);
featStd(featStd == 0) = 1;

features = (features - featMean) ./ featStd;

%% Feature Engineering

sampleImg = readimage(imdsTrain,1);
if size(sampleImg,3)==3
    gray = rgb2gray(sampleImg);
else
    gray = sampleImg;
end

I = double(gray);

[Gx, Gy] = imgradientxy(I, 'sobel');
edgeMag = sqrt(Gx.^2 + Gy.^2);

figure; imshow(edgeMag, []);
title('Edge Magnitude (Sobel)');


glcm = graycomatrix(gray,'Offset',[0 1]);
stats_glcm = graycoprops(glcm);

disp('GLCM Texture Features:');
disp(stats_glcm);


F = fft2(I);
Fshift = fftshift(abs(F));

figure; imshow(log(1+Fshift),[]);
title('Fourier Transform Magnitude');

[CA,CH,CV,CD] = dwt2(I,'haar');

figure;
subplot(2,2,1); imshow(CA,[]); title('Approximation');
subplot(2,2,2); imshow(CH,[]); title('Horizontal');
subplot(2,2,3); imshow(CV,[]); title('Vertical');
subplot(2,2,4); imshow(CD,[]); title('Diagonal');

% PCA implementation
[coeff, score, latent] = pca(features);

figure;
plot(cumsum(latent)./sum(latent)*100,'LineWidth',2);
xlabel('Principal Components');
ylabel('Variance Explained (%)');
title('PCA Variance Explained');

%% Statistical Analysis

allPixels = [];

firePixels = [];
nofirePixels = [];

for i = 1:numel(imds.Files)
    img = readimage(imds,i);
    if size(img,3)==3
        img = rgb2gray(img);
    end
    
    pixels = double(img(:));
    allPixels = [allPixels; pixels];
    
    if imds.Labels(i) == "fire"
        firePixels = [firePixels; pixels];
    else
        nofirePixels = [nofirePixels; pixels];
    end
end

fprintf('\n--- Statistical Analysis ---\n');
fprintf('Mean: %.2f\n', mean(allPixels));
fprintf('Median: %.2f\n', median(allPixels));
fprintf('Std Dev: %.2f\n', std(allPixels));
fprintf('Skewness: %.2f\n', skewness(allPixels));
fprintf('Kurtosis: %.2f\n', kurtosis(allPixels));


[h,p] = ttest2(firePixels, nofirePixels);

fprintf('\n--- Hypothesis Test ---\n');
fprintf('p-value: %.5f\n', p);

if p < 0.05
    disp('H1: Significant difference between fire and nofire');
else
    disp('H0: No significant difference');
end


ci = mean(firePixels) + [-1 1]*1.96*std(firePixels)/sqrt(length(firePixels));
fprintf('Confidence Interval (Fire Mean): [%.2f %.2f]\n', ci(1), ci(2));

%% Visualization of data

figure;
histogram(firePixels,50,'Normalization','pdf');
hold on;
histogram(nofirePixels,50,'Normalization','pdf');
legend('Fire','NoFire');
title('Pixel Intensity Distribution');


figure;
boxplot([firePixels; nofirePixels], ...
    [ones(size(firePixels)); 2*ones(size(nofirePixels))]);
xticklabels({'Fire','NoFire'});
title('Boxplot Comparison');


figure;
[f1,xi1] = ksdensity(firePixels);
[f2,xi2] = ksdensity(nofirePixels);

plot(xi1,f1,'r','LineWidth',2); hold on;
plot(xi2,f2,'b','LineWidth',2);

xline(mean(firePixels),'r--','Mean Fire');
xline(median(firePixels),'r:','Median Fire');

xline(mean(nofirePixels),'b--','Mean NoFire');
xline(median(nofirePixels),'b:','Median NoFire');

title(['Density + Mean + Median (p = ', num2str(p), ')']);
legend('Fire Density','NoFire Density');


figure;
scatter3(score(:,1), score(:,2), score(:,3), 40, imdsTrain.Labels, 'filled');
title('3D PCA Feature Space');
xlabel('PC1'); ylabel('PC2'); zlabel('PC3');
grid on;

%% Model Training
SVMModel = fitcsvm(features, trainLabels);
KNNModel = fitcknn(features, trainLabels, 'NumNeighbors', 3, 'Standardize', 1);
LRModel  = fitclinear(features, trainLabels, 'Learner', 'logistic');

numTest = numel(imdsTest.Files);
testFeatures = [];

for i = 1:numTest
    img = readimage(imdsTest, i);
    
    if ndims(img) == 2
        grayImg = img;
        imgRGB = cat(3, img, img, img);
    else
        grayImg = rgb2gray(img);
        imgRGB = img;
    end
    
    hog = extractHOGFeatures(grayImg);
    
    hsvImg = rgb2hsv(imgRGB);
    colorFeat = mean(reshape(hsvImg, [], 3));
    
    testFeatures(i,:) = [hog colorFeat];
end

testFeatures = (testFeatures - featMean) ./ featStd;

testLabels = imdsTest.Labels;

% Predictive analysis
predSVM = predict(SVMModel, testFeatures);
predKNN = predict(KNNModel, testFeatures);
predLR  = predict(LRModel, testFeatures);

% Accuracy
accSVM = mean(predSVM == testLabels);
accKNN = mean(predKNN == testLabels);
accLR  = mean(predLR == testLabels);

fprintf('\nAccuracy:\n');
fprintf('SVM: %.2f%%\n', accSVM*100);
fprintf('KNN: %.2f%%\n', accKNN*100);
fprintf('Logistic Regression: %.2f%%\n', accLR*100);

classNames = categories(testLabels);

posClass = "fire";
% Determination of other metrices

[~,scoreSVM] = predict(SVMModel, testFeatures);
[~,scoreKNN] = predict(KNNModel, testFeatures);
[~,scoreLR]  = predict(LRModel, testFeatures);


classNames = categories(testLabels);
fireCol = find(classNames == posClass);

if size(scoreSVM,2) == 2
    scoreSVM = scoreSVM(:,fireCol);
    scoreKNN = scoreKNN(:,fireCol);
    scoreLR  = scoreLR(:,fireCol);
end


predList  = {predSVM, predKNN, predLR};
scoreList = {scoreSVM, scoreKNN, scoreLR};
names     = ["SVM","KNN","Logistic Regression"];

fprintf('\n--- Detailed Metrics ---\n');

for k = 1:3
    
    pred = predList{k};
    score = scoreList{k};
    
   
    TP = sum((pred == posClass) & (testLabels == posClass));
    TN = sum((pred ~= posClass) & (testLabels ~= posClass));
    FP = sum((pred == posClass) & (testLabels ~= posClass));
    FN = sum((pred ~= posClass) & (testLabels == posClass));
    
  
    Recall = TP / (TP + FN);
    Specificity = TN / (TN + FP);
    Precision = TP / (TP + FP);
    F1 = 2 * (Precision * Recall) / (Precision + Recall);
    
   
    [~,~,~,AUC] = perfcurve(testLabels, score, posClass);
    
    
    fprintf('\n%s:\n', names(k));
    fprintf('Sensitivity (Recall): %.4f\n', Recall);
    fprintf('Specificity: %.4f\n', Specificity);
    fprintf('Precision: %.4f\n', Precision);
    fprintf('F1-Score: %.4f\n', F1);
    fprintf('AUC: %.4f\n', AUC);
end

%% Confusion Matrices
figure; confusionchart(testLabels, predSVM); title('SVM');
figure; confusionchart(testLabels, predKNN); title('KNN');
figure; confusionchart(testLabels, predLR); title('LR');

%% Determination of best models
[bestAcc, idx] = max([accSVM accKNN accLR]);

if idx == 1
    bestModel = SVMModel; modelName = "SVM";
elseif idx == 2
    bestModel = KNNModel; modelName = "KNN";
else
    bestModel = LRModel; modelName = "LR";
end

fprintf('\nBest Model: %s (%.2f%% Accuracy)\n', modelName, bestAcc*100);

%% Testing

testFolder = 'WildFire_Tester';
testImds = imageDatastore(testFolder);

numNew = numel(testImds.Files);

predLabels = strings(numNew,1); 

figure('Name','All Test Image Predictions','NumberTitle','off');

cols = 4;
rows = ceil(numNew / cols);

for i = 1:numNew
    
    try
        img = imread(testImds.Files{i});
        img = imresize(img, inputSize);
    catch
        fprintf('Skipping corrupted file: %s\n', testImds.Files{i});
        continue;
    end
    
   
    if ndims(img) == 2
        imgRGB = cat(3, img, img, img);
        grayImg = img;
    else
        imgRGB = img;
        grayImg = rgb2gray(img);
    end
    
    % Test image prediction
    hog_full = extractHOGFeatures(grayImg);
    hsv_full = rgb2hsv(imgRGB);
    color_full = mean(reshape(hsv_full, [], 3));
    
    feat_full = [hog_full color_full];
    feat_full = reshape(feat_full, 1, []);
    feat_full = (feat_full - featMean) ./ featStd;
    
    finalLabel = predict(bestModel, feat_full);
    predLabels(i) = string(finalLabel);
    
    % fprintf('Image %d (%s): %s\n', i, testImds.Files{i}, finalLabel);
    
    subplot(rows, cols, i);
    imshow(imgRGB);
    title(string(finalLabel));
end


fireIdx = find(predLabels == "fire");
numFire = length(fireIdx);

for n = 1:numFire
    
    i = fireIdx(n);
    
    img = imread(testImds.Files{i});
    img = imresize(img, inputSize);
    
   
    if ndims(img) == 2
        imgRGB = cat(3, img, img, img);
        grayImg = img;
    else
        imgRGB = img;
        grayImg = rgb2gray(img);
    end
    
    windowSize = 32;
    stride = 16;
    heatmap = zeros(size(grayImg));
    
    for x = 1:stride:size(grayImg,1)-windowSize
        for y = 1:stride:size(grayImg,2)-windowSize
            
            patch = imgRGB(x:x+windowSize-1, y:y+windowSize-1, :);
            patch = imresize(patch, inputSize);
            
            if size(patch,3) == 3
                patchGray = rgb2gray(patch);
            else
                patchGray = patch;
                patch = cat(3, patch, patch, patch);
            end
            
            hog = extractHOGFeatures(patchGray);
            hsvImg = rgb2hsv(patch);
            colorFeat = mean(reshape(hsvImg, [], 3));
            
            feat = [hog colorFeat];
            feat = reshape(feat, 1, []);
            feat = (feat - featMean) ./ featStd;
            
            label = predict(bestModel, feat);
            
            if label == "fire"
                heatmap(x:x+windowSize-1, y:y+windowSize-1) = ...
                    heatmap(x:x+windowSize-1, y:y+windowSize-1) + 1;
            end
        end
    end
    
    heatmap = mat2gray(heatmap);
    
    bw = heatmap > 0.5;
    bw = bwareaopen(bw, 50);
    stats = regionprops(bw, 'BoundingBox');
    
    
    figure('Name', ['Fire Image ', num2str(i)], 'NumberTitle', 'off');
    
    % Original
    subplot(1,3,1);
    imshow(imgRGB);
    title('Original');
    
    % Heatmap
    subplot(1,3,2);
    imshow(imgRGB);
    hold on;
    h = imshow(heatmap);
    colormap jet;
    set(h, 'AlphaData', 0.5);
    title('Heatmap');
    
    % Detection of fire
    subplot(1,3,3);
    imshow(imgRGB);
    hold on;
    
    for k = 1:length(stats)
        rectangle('Position', stats(k).BoundingBox, ...
            'EdgeColor', 'r', 'LineWidth', 2);
    end
    
    title('Detection');
    
end
