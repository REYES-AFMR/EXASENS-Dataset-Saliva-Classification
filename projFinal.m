%% RESET
clear ; close all; clc

%% 1. LOADING DATA
% Exasens data already pre-processed in the matlab.mat file
% (removal of rows with missing data)
load;

% copy of table before altering for outlier removal
withOutliers = EXASENSdata;

% detection of outliers:
[noOut, reOrNot] = rmoutliers(EXASENSdata(:,2:8));

% variable values (not outliers) conversion to array
varVals = table2array(noOut);

% removal of outliters (to stay uniform with the variable values)
EXASENSdata(reOrNot,:) = [];

% ONE HOT ENCODING:
% extract smoking status
smoStatOg = EXASENSdata(:,"smoking status");
% create the table of one hot encoding
smoStat = onehotencode(smoStatOg,ClassNames=[1; 2; 3]);
% append it to the previous table
EXASENSdata = [EXASENSdata smoStat];
% delete old column
EXASENSdata = removevars(EXASENSdata, "smoking status");
% relabel
EXASENSdata.Properties.VariableNames(8) = "Non-smoker";
EXASENSdata.Properties.VariableNames(9) = "Ex-smoker";
EXASENSdata.Properties.VariableNames(10) = "Active smoker";

% removing the gender column since only one value remains
% after outlier removal
varVals(:,5) = [];
% normalization
exaValNorm = normalize(varVals);

% appending smoker status
exaValNorm(:, 6:8) = table2array(EXASENSdata(:, 8:10));

% extracting labels from the EXASENS file and converting it into
% a categorical data format
diagLabelsArray = table2array(EXASENSdata(:,1));
labelsCategorical = categorical(diagLabelsArray);

%///////////////////////////////////////////////////
% doing the same for the outliers-inclusive table:
% variable values (not outliers) conversion to array
varValsWOut = table2array(withOutliers(:, 2:8));

% ONE HOT ENCODING:
% extract smoking status
smoStatOgWOut = withOutliers(:,"smoking status");
% create the table of one hot encoding
smoStatwOut = onehotencode(smoStatOgWOut,ClassNames=[1; 2; 3]);
% append it to the previous table
withOutliers = [withOutliers smoStatwOut];
% delete old column
withOutliers = removevars(withOutliers, "smoking status");
% relabel
withOutliers.Properties.VariableNames(8) = "Non-smoker";
withOutliers.Properties.VariableNames(9) = "Ex-smoker";
withOutliers.Properties.VariableNames(10) = "Active smoker";

exaValNormWOut = normalize(varValsWOut);

% appending smoker status
exaValNormWOut(:, 7:9) = table2array(withOutliers(:, 8:10));

% extracting labels from the EXASENS file and converting it into
% a categorical data format
diagLabelsArrayWOut = table2array(withOutliers(:,1));
labelsCategoricalWOut = categorical(diagLabelsArrayWOut);

% TABLE GUIDE:
% exaValNorm: normalized values (min imaginary, ave. imaginary,
% min real, ave. real, age, one hot encoding table for smoking)

%% 2. SPLITTING INTO GROUPS
function [trainData, testData, trainLabels, testLabels] = splitData(data, labels, trainRatio, testRatio)
    % Function to split data into training, validation, and testing sets
    

    numSamples = size(data, 1);
    indices = randperm(numSamples);
    
    numTrain = round(trainRatio * numSamples);
    
    trainIdx = indices(1:numTrain);
    testIdx = indices(numTrain+1:end);
    
    trainData = data(trainIdx, :);
    testData = data(testIdx, :);
    
    trainLabels = labels(trainIdx);
    testLabels = labels(testIdx);
    
    fprintf('Training set size: %d\n', size(trainData, 1));
    fprintf('Testing set size: %d\n', size(testData, 1));
end

% Define the proportions for splitting
trainRatio = 0.7; % training
testRatio = 0.3; % testing

% Call the splitData function
[trainData, testData, trainLabels,testLabels] = splitData(exaValNorm, labelsCategorical, trainRatio, testRatio);

% Diagnosis classes
diagCategories = unique(labelsCategorical);
trainCompleteCategories = ismember(diagCategories, trainLabels);
testCompleteCategories = ismember(diagCategories, testLabels);

while ~(all(trainCompleteCategories)&&all(testCompleteCategories))
    disp("Not all categories represented in the training and" + ...
        " testing groups. Attempting to split again.")
    disp(" ")
    rng("shuffle"); % Set a random value
    [trainData, testData, trainLabels,testLabels] = splitData(exaValNorm, labelsCategorical, trainRatio, testRatio);
    trainCompleteCategories = ismember(diagCategories, trainLabels);
    testCompleteCategories = ismember(diagCategories, testLabels);
end


%% 3. KNN TRAINING AND TESTING

%% 3.1 All Data vs. Diagnosis

% Training Confusion Matrix for All Values
train_model_All = fitcknn(trainData, trainLabels, 'NumNeighbors', 3, 'BreakTies', 'nearest');
predTrain_All = predict(train_model_All, trainData);
figure;
plotconfusion(trainLabels, categorical(predTrain_All));
title('Training Confusion Matrix for All Values vs. Diagnosis');

% Testing Confusion Matrix for All Values
predTest_All = predict(train_model_All, testData);
figure;
plotconfusion(testLabels, categorical(predTest_All));
title('Testing Confusion Matrix for All Values vs. Diagnosis');

%% 3.2 Saliva Permittivity vs. Diagnosis

% Get the data of saliva permittivity for train
train_salivaPermittivityData = trainData(:, 1:4);

% Training for Saliva Permittivity 
model_SalivaPermittivity = fitcknn(train_salivaPermittivityData, trainLabels, 'NumNeighbors', 3, 'BreakTies', 'nearest');
predTrain_SalivaPermittivity = predict(model_SalivaPermittivity, train_salivaPermittivityData);
figure;
plotconfusion(trainLabels, categorical(predTrain_SalivaPermittivity));
title('Training Confusion Matrix for Saliva Permittivity (Imaginary and Real) vs. Diagnosis');

% Get the data of saliva permittivity for test
test_salivaPermittivityData = testData(:, 1:4);

% Testing Confusion Matrix for Saliva Permittivity 
predTest_SalivaPermittivity = predict(model_SalivaPermittivity, test_salivaPermittivityData);
figure;
plotconfusion(testLabels, categorical(predTest_SalivaPermittivity));
title('Testing Confusion Matrix for Saliva Permittivity (Imaginary and Real) vs. Diagnosis');

%% 3.3 Age vs. Diagnosis

train_AgeData = trainData(:,5);

% Training for Age vs. Diagnosis
model_Age = fitcknn(train_AgeData, trainLabels, 'NumNeighbors', 3, 'BreakTies', 'nearest');
predTrain_Age = predict(model_Age, train_AgeData);
figure;
plotconfusion(trainLabels, categorical(predTrain_Age));
title('Training Confusion Matrix: Age vs Diagnosis');

test_AgeData = testData(:, 5);

% Testing for Age vs. Diagnosis
predTest_Age = predict(model_Age, test_AgeData);
figure;
plotconfusion(testLabels, categorical(predTest_Age));
title('Testing Confusion Matrix: Age vs Diagnosis');

%% 3.4 Smoking Status vs Diagnosis

% Get the data of smoking status for train
train_smokingStatus = trainData(:, 6:8); 

% Training for Smoking Status 
model_Smoking= fitcknn(train_smokingStatus, trainLabels, 'NumNeighbors', 3, 'BreakTies', 'nearest');
predTrain_Smoking = predict(model_Smoking, train_smokingStatus);
figure;
plotconfusion(trainLabels, categorical(predTrain_Smoking));
title('Training Confusion Matrix for Smoking Status and Diagnosis');

% Get the data of smoking status for test
test_smokingStatus = testData(:, 6:8);
% Testing Confusion Matrix for Smoking Status
predTest_Smoking = predict(model_Smoking, test_smokingStatus);
figure;
plotconfusion(testLabels, categorical(predTest_Smoking));
title('Testing Confusion Matrix for Smoking Status and Diagnosis');

%% 3.5 Saliva Permittivity vs. Smoking Status

% Get the data of saliva (all) for train 
salivaData = exaValNorm(:, 1:4); 

% Extract Smoking Status One-Hot Encoding
nonSmoker = EXASENSdata{:,"Non-smoker"};
exSmoker = EXASENSdata{:,"Ex-smoker"};
activeSmoker = EXASENSdata{:,"Active smoker"};
% Convert One-Hot Encoding to Categorical Labels
smokingLabels = strings(size(nonSmoker, 1), 1);
smokingLabels(nonSmoker == 1) = "Non-smoker";
smokingLabels(exSmoker == 1) = "Ex-smoker";
smokingLabels(activeSmoker == 1) = "Active smoker";
% Convert to Categorical
smokingLabels = categorical(smokingLabels);

% Split into Training and Testing Sets
[train_SalivaData, test_SalivaData, trainSmokingLabels, testSmokingLabels] = splitData(salivaData, smokingLabels, trainRatio, testRatio);

% Training Confusion Matrix for Age vs. Smoking Status
model_SalivaSmoking = fitcknn(train_SalivaData, trainSmokingLabels, 'NumNeighbors', 3, 'BreakTies', 'nearest');
predTrain_Saliva = predict(model_SalivaSmoking, train_SalivaData);
figure;
plotconfusion(trainSmokingLabels, categorical(predTrain_Saliva));
title('Training Confusion Matrix: Saliva (all) vs Smoking Status');

% Testing Confusion Matrix for Age vs. Smoking Status
predTest_Saliva = predict(model_SalivaSmoking, test_SalivaData);
figure;
plotconfusion(testSmokingLabels, categorical(predTest_Saliva));
title('Testing Confusion Matrix: Saliva (all) vs Smoking Status');

%% 3.6 All parameters, no split, no outliters

figure;
model = fitcknn(exaValNorm, labelsCategorical, NumNeighbors=3, BreakTies="nearest");
predAllNoSplitNoOut = predict(model,exaValNorm);
predCat = categorical(predAllNoSplitNoOut);
plotconfusion(labelsCategorical, predCat)
title("All parameters, no split, no outliers")

%% 3.7 All parameters, no split, inclusive of outliers

figure;
model = fitcknn(exaValNormWOut, labelsCategoricalWOut, NumNeighbors=3, BreakTies="nearest");
predAllNoSplitWOut = predict(model,exaValNormWOut);
predCat = categorical(predAllNoSplitWOut);
plotconfusion(labelsCategoricalWOut, predCat)
title("All parameters, no split, inclusive of outliers")

%% 4. Predictor Significance using a Binary Decision Tree

btree = fitctree(exaValNorm, labelsCategorical)

imp = predictorImportance(btree)

figure;
bg = bar(imp);
bg.Labels = bg.YData;
title('Predictor Importance Estimates');
ylabel('Estimates');
xlabel('Predictors');
h = gca;
h.YLim = [0 max(imp)*1.25];
h.XTickLabel = EXASENSdata.Properties.VariableNames([2:5, 7:10]);
h.XTickLabelRotation = 45;
h.TickLabelInterpreter = 'none';

%% Checking Age as a Predictor

% heatmap trend
figure;
heatmap(EXASENSdata, 'age', 'Diagnosis', 'GridVisible', 'off','CellLabelColor','none')

% no split
figure;
model = fitcknn(exaValNorm(:,5), labelsCategorical, NumNeighbors=3, BreakTies="nearest");
predAllNoSplitNoOut = predict(model,exaValNorm(:,5));
predCat = categorical(predAllNoSplitNoOut);
plotconfusion(labelsCategorical, predCat)
title("Age, no split, no outliers")