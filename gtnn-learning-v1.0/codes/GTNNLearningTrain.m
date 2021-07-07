function [network, trainedNetwork, trainResults] = GTNNLearningTrain(network, hyperparams, procData, flag)

trainxp = procData.trainx; Ytrainp = procData.Ytrain;
valxp = procData.valx; Yvalp = procData.Yval;

[Ntrain, D] = size(trainxp);
[~, M] = size(Ytrainp);
[Nval, ~] = size(valxp);

%% Initialize connection matrices for the layers
Q = cell(network.num_layers,1);
mask = cell(network.num_layers,1);
N = zeros(network.num_layers, 1);
for k = 1:network.num_layers
    
    if k<network.num_layers
        N(k) = 2*(D + M*network.include_labels(k))*network.num_sub(k);
    else
        N(k) = 2*((N(k-1)/2) + M*network.include_labels(k))*network.num_sub(k);
    end
    Qk = GTNNLearningGenerateQ(N(k)/2, network.density(k), k, network.network_type(k));
    Q{k, 1} = Qk;
    mask{k, 1} = (Qk~=0)*1;
    
end
Q0 = Q;
Nt = sum(N);
network.N = N;

poss_labels = eye(M, M);
poss_labels(poss_labels==0) = -1;


% Store parameters
cost = [];
accuracy_train = [];
accuracy_val = [];
metric_train = [];
metric_val = [];
bestQ = Q;


%% Training phase
count = 1; ind_val_max = 1; val_max = 0;
while (count-ind_val_max)<=hyperparams.improv_epochs
    
    rng(count); ind = randperm(Ntrain);
    trainx = trainxp(ind, :);
    Ytrain = Ytrainp(ind, :);
    cost_epoch = zeros(network.num_layers, Ntrain);
    fprintf('\n Epoch %d of training...', count-1);
    tic
    
    %% Incremental training
    for i = 1:Ntrain
        
        X = trainx(i, :)';
        Y = Ytrain(i, :)';
        
        % Train GTNN
        [~, ~, spikes, ~, Q] = GTNNLearningWeightAdapt(Q, X, Y, zeros(Nt, 1), (count>1)*1, network, hyperparams, mask, -0.1*ones(Nt, 1), count-1);
        cost_epoch(2, i) = sum(mean(spikes, 2));
        
    end
    
    cost = [cost, mean(cost_epoch, 2)];
    
    
    %% Inference on training and validation data
    for t = 1:2
        
        if t == 1
            Ntotal = Ntrain;
            feats = trainxp;
            labels = Ytrainp;
        else
            Ntotal = Nval;
            feats = valxp;
            labels = Yvalp;
        end
        
        
        accuracy = 0;
        metric_epoch = zeros(1, Ntotal);
        for i = 1:Ntotal
            
            X = feats(i, :)';
            Y = labels(i, :)';
            output_spikes = zeros(1, M);
            
            for m = 1:M
                
                
                [~, ~, spikes, ~, Q] = GTNNLearningWeightAdapt(Q, X, poss_labels(m, :)', zeros(Nt, 1), 0, network, hyperparams, mask, -0.1*ones(Nt, 1), 0);
                output_spikes(1, m) = (network.last_layer == 0)*sum(spikes(:)) + (network.last_layer == 1)*sum(sum(spikes(Nt-N(end)+1:Nt, :)));
                if isequal(Y, poss_labels(m, :)')
                    metric_epoch(1, i) = sum(mean(spikes, 2));
                end
                
            end
            
            [~, ind] = min(output_spikes(1, :));
            [~, indtrue] = max(Y);
            accuracy = accuracy + (ind == indtrue)*1;
            
            
        end
        accuracy = (accuracy/Ntotal)*100;
        
        if t==1
            accuracy_train = [accuracy_train, accuracy];
            metric_train = [metric_train, mean(metric_epoch)/sum(N)];
        else
            fprintf('\n Validation accuracy = %.2f', accuracy);
            if accuracy > val_max
                bestQ = Q;
                ind_val_max = count;
                val_max = accuracy;
            end
            accuracy_val = [accuracy_val, accuracy];
            metric_val = [metric_val, mean(metric_epoch)/sum(N)];
        end
        
    end
    
    toc
    count = count + 1;
    if t==2 && accuracy==100
        break
    end
    
end
fprintf('\n...Training done');
indices = find(accuracy_val==max(accuracy_val));
ind = indices(end);

fprintf('\n Training accuracy = %.2f', accuracy_train(ind));
fprintf('\n Train sparsity metric = %.4f', metric_train(ind));

fprintf('\n Highest validation accuracy = %.2f', accuracy_val(ind));
fprintf('\n Validation sparsity metric = %.4f', metric_val(ind));


if flag.plotFlag == 1
    
    set(figure,'defaultAxesColorOrder',[0.75 0 0; 0 0 0]);
    set(gca, 'FontSize', 12);
    xaxis = 0:count-2;
    yyaxis left;
    plot(xaxis, accuracy_train, 'LineWidth', 3);
    ylabel('Training accuracy (%)');
    yyaxis right;
    plot(xaxis, metric_train, 'LineWidth', 3);
    ylabel('Training metric');
    box on; grid on;
    title('Training plots');
    hold off
    
    set(figure,'defaultAxesColorOrder',[0.75 0 0; 0 0 0]);
    set(gca, 'FontSize', 12);
    xaxis = 0:count-2;
    yyaxis left;
    plot(xaxis, accuracy_val, 'LineWidth', 3);
    ylabel('Validation accuracy (%)');
    yyaxis right;
    plot(xaxis, metric_val, 'LineWidth', 3);
    ylabel('Validation metric');
    box on; grid on;
    title('Validation plots');
    hold off
    
end

trainedNetwork.Q = bestQ;
trainedNetwork.mask = mask;

trainResults.accuracy_train = accuracy_train;
trainResults.metric_train = metric_train;
trainResults.accuracy_val = accuracy_val;
trainResults.metric_val = metric_val;
trainResults.cost = cost;