function GTNNLearningContour(network, hyperparams, trainedNetwork, procData, range, step)

trainxp = procData.trainx; Ytrainp = procData.Ytrain;
[~, M] = size(Ytrainp);
Nt = sum(network.N);
poss_labels = eye(M, M);
poss_labels(poss_labels==0) = -1;

u1 = range(1); u2 = range(2);
[XX, YY] = meshgrid(u1:step:u2, u1:step:u2);
[s1, s2] = size(XX);
ZZ = zeros(s1, s2);

for i = 1:s1
    
    for j = 1:s2
        
        X = [XX(i,j) YY(i,j)]';
        output_spikes = zeros(1, M);
        
        for m = 1:M
            
            [~, ~, spikes, ~, ~] = GTNNLearningWeightAdapt(trainedNetwork.Q, X, poss_labels(m, :)', zeros(Nt, 1), 0, hyperparams.maxiter, hyperparams.eta, trainedNetwork.mask, -0.1*ones(Nt, 1), network.N, network.num_sub);
            output_spikes(1, m) = (network.last_layer == 0)*sum(spikes(:)) + (network.last_layer == 1)*sum(sum(spikes(Nt-network.N(end)+1:Nt, :)));
            
        end
        
        [~, ind] = min(output_spikes);
        if ind == 1
            ZZ(i, j, :) = 0;  
        else
            ZZ(i, j, :) = 1;  
        end
        
    end
    
end

figure; hold on;
cmap = jet(2);
cmap(1, :) = 1.1*[0.65 0.65 0.65];
cmap(2, :) = 1.1*[0.66 0.85 0.92];
cmap(cmap>1) = 1;
colormap(cmap);
h = pcolor(XX, YY, ZZ); 
set(h, 'EdgeColor', 'none');
sz = 10;
hold on
plot(trainxp(Ytrainp(:, 1)>0, 1), trainxp(Ytrainp(:, 1)>0, 2), 'ok', 'MarkerSize', sz, 'MarkerFaceColor', [0.65 0.65 0.65]);
plot(trainxp(Ytrainp(:, 1)<0, 1), trainxp(Ytrainp(:, 1)<0, 2), 'ok', 'MarkerSize', sz, 'MarkerFaceColor', [0.66 0.85 0.92]);
set(gca,'XTickLabel',[]);
set(gca,'YTickLabel',[]);
hold off