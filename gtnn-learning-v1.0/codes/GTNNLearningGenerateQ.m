function Q = GTNNLearningGenerateQ(N, density, seed, network_type)

Q = zeros(2*N, 2*N);

switch network_type
    
    case 1 % Fully-connected network
        rng(seed); Q(1:N, 1:N) = 0.5*randn(N, N); 
        for i=1:N
            Q(i, i) = 0;
        end
        conn_mask = zeros(N, N);
        conn_mask(randperm(numel(conn_mask), ceil(density*numel(conn_mask)))) = 1;
        Q(1:N, 1:N) = Q(1:N, 1:N).*conn_mask;
        
    case 2 % Feed-forward network
        rng(seed); Q(1, 2:N) = 0.5*rand(1, N-1);
    
end
 
Q(N+1:2*N, N+1:2*N) = Q(1:N, 1:N);
Q(1:N, N+1:2*N) = - Q(1:N, 1:N);
Q(N+1:2*N, 1:N) = - Q(1:N, 1:N);
Q = Q + eye(2*N, 2*N);