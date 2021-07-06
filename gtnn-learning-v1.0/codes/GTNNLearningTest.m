function testResults = GTNNLearningTest(network, hyperparams, trainedNetwork, procData)

testxp = procData.testx; Ytestp = procData.Ytest;
[Ntest, ~] = size(testxp);
[~, M] = size(Ytestp);
Q = trainedNetwork.Q;
Nt = sum(network.N);
mask = trainedNetwork.mask;
poss_labels = eye(M, M);
poss_labels(poss_labels==0) = -1;

%% Inference on test set
conf_mat_test = zeros(M, M);
accuracy_test = 0;
output_spikes = zeros(Ntest, M);
metric_epoch = zeros(1, Ntest);

fprintf('\n Running inference on test set...');
for i = 1:Ntest
    
    X = testxp(i, :)';
    Y = Ytestp(i, :)';
    
    for m = 1:M
        
       [~, ~, spikes, ~, ~] = GTNNLearningWeightAdapt(Q, X, poss_labels(m, :)', zeros(Nt, 1), 0, network, hyperparams, mask, -0.1*ones(Nt, 1), 0);
       output_spikes(i, m) = (network.last_layer == 0)*sum(spikes(:)) + (network.last_layer == 1)*sum(sum(spikes(Nt-network.N(end)+1:Nt, :)));
       if isequal(Y, poss_labels(m, :)')
           metric_epoch(1, i) = sum(mean(spikes, 2));
       end
        
    end
    
    [~, ind] = min(output_spikes(i, :));
    [~, indtrue] = max(Y);
    accuracy_test = accuracy_test + (ind == indtrue)*1;
    conf_mat_test(ind, indtrue) = conf_mat_test(ind, indtrue) + 1;
    
end
fprintf('\n...done');
metric_test = mean(metric_epoch)/Nt;
accuracy_test = (accuracy_test/Ntest)*100;
fprintf('\n Final test accuracy is %.2f', accuracy_test);
fprintf('\n Test energy metric = %.4f', metric_test);


testResults.accuracy_test = accuracy_test;
testResults.metric_test = metric_test;