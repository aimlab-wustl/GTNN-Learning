function procData = GTNNLearningDataPreprocess(trainx, Ytrain, valx, Yval, testx, Ytest, shuffleFlag, valData, scaleFlag, range)

[Ntrain, ~] = size(trainx);
if shuffleFlag == 1
   rng(10); ind = randperm(Ntrain);
   trainx = trainx(ind, :);
   Ytrain = Ytrain(ind, :);
end

if isempty(valx)
    trainxp = trainx; Ytrainp = Ytrain;
    Nt = round((1-valData)*Ntrain);
    trainx = trainxp(1:Nt, :);
    Ytrain = Ytrainp(1:Nt, :);
    valx = trainxp(Nt+1:end, :);
    Yval = Ytrainp(Nt+1:end, :);
end


[Ntrain, D] = size(trainx);
[Ntrainy, M] = size(Ytrain);
[Nval, Dv] = size(valx);
[Nvaly, Mv] = size(Yval);
[Ntest, Dt] = size(testx);
[Ntesty, Mt] = size(Ytest);


%% Sanity check
if Ntrain~=Ntrainy
   error('Dimension mismatch between training set and train labels'); 
end

if Nval~=Nvaly
   error('Dimension mismatch between validation set and validation labels'); 
end

if Ntest~=Ntesty
   error('Dimension mismatch between test set and test labels'); 
end

if numel(unique([D, Dv, Dt]))>1
    error('Dimension mismatch between train, validation and test sets'); 
end

if numel(unique([M, Mv, Mt]))>1
    error('Class mismatch between train, validation and test sets'); 
end


if scaleFlag==1
   trainxp = trainx;
   trainx = (range(2) - range(1))*(trainx - min(trainxp))./(max(trainxp) - min(trainxp)) + range(1);
   valx = (range(2) - range(1))*(valx - min(trainxp))./(max(trainxp) - min(trainxp)) + range(1);
   testx = (range(2) - range(1))*(testx - min(trainxp))./(max(trainxp) - min(trainxp)) + range(1);
end

procData.trainx = trainx; procData.Ytrain = Ytrain;
procData.valx = valx; procData.Yval = Yval;
procData.testx = testx; procData.Ytest = Ytest;