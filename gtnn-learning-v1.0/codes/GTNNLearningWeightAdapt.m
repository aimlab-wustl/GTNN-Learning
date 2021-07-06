function [dpcheckf, dpf, indf, psif, Q] = GTNNLearningWeightAdapt(Q, X, Y, thr, Qadapt, network, hyperparams, mask, dp0, epoch)

N = network.N;
Nt = sum(N);
num_sub = network.num_sub;
num_layers = length(N);
Fac = 20*ones(Nt,1);
dp = dp0;
sec = 3*ones(Nt,1);
dec = 0*ones(Nt,1);
a_iter = 0.5*ones(Nt, 1);
v_c = 1;

% Store iteration variables
maxiter = hyperparams.maxiter;
dpcheckf = zeros(Nt, maxiter);
dpf = zeros(Nt, maxiter);
indf = zeros(Nt, maxiter);
psif = zeros(Nt, maxiter);

for iter = 1:maxiter
    
    ind = (dp > 0);
    indf(:, iter) = ind;
    dp(dp > 0) = 0;
    quant = sec.*(ind) - dec.*(dp <= 0);
    psif(:, iter) = quant;
    dpcheckf(:, iter) = dp + 1*(ind);
    dpf(:, iter) = dp;
    dpprev = dp;
    start = 1;
    
    %% Layer-wise updates
    for k = 1:num_layers
        
        nidx = start: start+N(k)-1;
        quantk = quant(nidx);
        thrk = thr(nidx);
        maskk = mask{k, 1};
        Qk = Q{k, 1};
        indfk = indf(nidx);

        
        %% Neuron updates
        add_input = Qk(1:N(k)/2, 1:N(k)/2)*thrk(1:N(k)/2, 1);
        b = zeros(N(k), 1);
        if network.include_labels == 1
            if k==1
                b(1:N(k)/2, 1) = repmat([X; Y], num_sub, 1);
            else
                b(1:(N(k)/2), :) = [dpf(nidxprev(1:N(k-1)/2), iter); Y];
            end
        else
            if k==1
                b(1:N(k)/2, 1) = repmat(X, num_sub, 1);
            elseif k<num_layers
                b(1:(N(k)/2), :) = dpf(nidxprev(1:N(k-1)/2), iter);
            else
                b(1:(N(k)/2), :) = [dpf(nidxprev(1:N(k-1)/2), iter); Y];
            end
        end
        b((N(k)/2)+1:N(k), :) = - b(1:N(k)/2, :);
        G = -Qk*dp(nidx, 1) + b - [add_input; -add_input] - quantk;
        dp(nidx) = v_c*(G*v_c + Fac(nidx).*dp(nidx))./(dp(nidx).*G + Fac(nidx)*v_c);
        dp(nidx) = a_iter(nidx).*dp(nidx) + (1-a_iter(nidx)).*dpprev(nidx);
        
        
        %% Weight updates using instantaneous values
        if Qadapt == 1
            
            if k<num_layers && (epoch==0 || epoch>hyperparams.trainEpochs)
                eta = 0;
            else
                eta = hyperparams.eta(k);
            end
            
            dpk = dp(nidx);
            delQ = (- (quantk(1:(N(k)/2), 1)>0) + (quantk(N(k)/2+1:N(k), 1)>0))*(dpk(1:N(k)/2, 1) - dpk((N(k)/2)+1:N(k), 1) + thr(1:N(k)/2, 1))';
            %check = (mean(indfk(1:N(k)/2, 1:iter), 2)<1) & (mean(indf((N(k)/2)+1:N(k), 1:iter), 2)<1); % 1:non-saturated, 0:saturated
            %delQ = delQ.*check;  % only update for non-saturated neuron pairs
            Qnew = Qk(1:N(k)/2, 1:N(k)/2) - eta.*maskk(1:N(k)/2, 1:N(k)/2).*delQ;
            Qk = [Qnew -Qnew; -Qnew Qnew];
            
            for i = 1:N(k)
                Qk(i, i) = 1;
            end
            Qk = Qk.*maskk;
            
        end
        
        Q{k, 1} = Qk;
        start = start + N(k);
        nidxprev = nidx;
        
    end
    
end