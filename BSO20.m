% The source code of the paper "BSO20: Efficient Brain Storm Optimization for Real-Parameter Numerical Optimization"

function [bestfit] = BSO20(fname, func_num, runs, D)

max_FEs = 1e4*D;          % Termination condition
NP = 4*D;                % population size
max_iter = max_FEs/NP; 

lb = -100;
ub = 100;

prob_one_cluster = 0.1;                     % probability for select one cluster to form new individual; 
Sr = 20;                                    % the cluster size of RGS
k = NP / Sr;

step_size = ones(1,D);                      % effecting the step size of generating new individuals by adding random values
% -----------

pop = lb + (ub - lb) * rand(NP,D);          % initialize the population of individuals
% pop = zeros(NP, D);
% -------------

iters = 0;                                  % current iteration number
FEs = 0;                                    % current used evaluations

% initialize cluster probability to be zeros
prob = zeros(k, 1);
% centers = zeros(NC,D);

fit = feval(fname, pop', func_num);         % calculate fitness for each individual in the initialized population
bestfit = min(fit);
temp_pop = zeros(NP, D);               % store temperary individual

FEs = FEs + NP;

while FEs < max_FEs
    % ---------------- Step start ----------------
    kr = ceil(k * (1-iters/max_iter));  % the cluster number of RGS
    N_nbc = NP - kr * Sr;
    kn = k - kr;
    
    species = struct();
    
    if kn > 0
        [fit, sidx] = sort(fit);
        pop = pop(sidx, :);
        pop_nbc = pop(1 : N_nbc, :);
        fit_nbc = fit(1 : N_nbc);
        matdis = pdist2(pop_nbc, pop_nbc);
        [species, leaders, cluster_idx] = NBC(matdis, fit_nbc, pop_nbc, kn);
    end
    
    N_rg = NP - N_nbc;
    rp = N_nbc + randperm(N_rg);
    start_idx = 1;
    for j = kn+1 : k
        if j == k
            species(j).idx = rp(start_idx : end);
        else
            species(j).idx = rp(start_idx : start_idx+Sr-1);
        end
        species(j).len = length(species(j).idx);
        cluster_idx(species(j).idx) = j;
        start_idx = start_idx + Sr;
    end
    for i = N_nbc+1 : NP
        c = cluster_idx(i);
        temp = find(fit(species(c).idx) < fit(i));
        if isempty(temp)
            leaders(i) = i;
        else
            leaders(i) = species(c).idx(randi(length(temp)));
        end
    end
    % ----------------- Step end -----------------
    
    % ----------------- generate new individual start ----------------
    % generate NP new individuals by adding Gaussian random values
    for i = 1 : NP
%         j = sum(rand > prob) + 1;
%         temp_idx = species(j).idx(randi(species(j).len));
        temp_idx = randi(NP);
        if rand < prob_one_cluster          % select one cluster 
            r = rand(1, D);
            temp_pop(i, :) = (1 - r) .* pop(temp_idx, :) + r .* pop(leaders(temp_idx), :);
        else                                % select two clusters
            c1 = randi(k);
            i1 = species(c1).idx(randi(species(c1).len));
            c2 = randi(k);
            i2 = species(c2).idx(randi(species(c2).len));
            
            r1 = rand(1, D);
            r2 = rand(1, D);
            
            temp_pop(i, :) = (1-r1-r2) .* pop(temp_idx, :) + r1 .* pop(i1, :) + r2 .* pop(i2, :);
        end                
        step_size = logsig(((0.5*max_iter - iters)/20)) * rand(1,D);
%         step_size = rand(1, D) .* exp(1 - (max_iter)/(max_iter - iters + 1));
        rn = normrnd(0,1,1,D);
        rn(find(rn) > ub) = ub;
        rn(find(rn) < lb) = lb;
        
        temp_pop(i, :) = temp_pop(i, :) + step_size .* rn;
    end
    % ----------------- generate new individual end ------------------
    
    % update the current population
    
    reflect = find(temp_pop > ub);
    temp_pop(reflect) = ub - mod((temp_pop(reflect) - ub), (ub - lb));
    
    reflect = find(temp_pop < lb);
    temp_pop(reflect) = lb + mod((lb - temp_pop(reflect)), (ub - lb));
    
    temp_fit = feval(fname, temp_pop', func_num);
    FEs = FEs + NP;
    is_update = temp_fit < fit;
    fit(is_update) = temp_fit(is_update);
    pop(is_update, :) = temp_pop(is_update, :);
    
    iters = iters + 1;    
    % record the best fitness in each iteration
    bestfit = min(fit);
    bestfit = bestfit - 100*func_num;
    if mod(iters, 100) == 0
        fprintf('%2d.%2d| Gen: %d, Func Val: %f\n', func_num, runs, iters, bestfit);
    end
end

end

function [species, leaders, cluster_idx] = NBC(matdis, fit, pop, k)
factor=2; % fai

n=length(matdis); 
leader_node = cell(n, 1);
% follower_node = cell(n, 1);
cluster_idx = zeros(n, 1);
nbc=zeros(n,3); 
nbc(1:n,1)=1:n;
nbc(1,2)=-1;
nbc(1,3)=-1;
for i=2:n
    [u,v]=min(matdis(i,1:i-1));
    nbc(i,2)=v;
    nbc(i,3)=u;
end

[~, sidx] = sort(nbc(:, 3), 'descend');
divid = sidx(1 : k-1);
nbc(divid, 2) = -1;
nbc(divid, 3) = -1;   

% meandis=factor*mean(nbc(2:n,3));
% nbc(nbc(:,3)>meandis,2)=-1;
% nbc(nbc(:,3)>meandis,3)=0;   
seeds=nbc(nbc(:,2)==-1,1);
m=zeros(n,2); 
m(1:n,1)=1:n;
for i=1:n
    j = nbc(i,2);
    k = j;
    while j ~= -1
        if j ~= -1
            leader_node{i} = [j, leader_node{i}];
        end
        k=j;
        j=nbc(j,2);
    end
    if k==-1
        m(i,2)=i;
    else
        m(i,2)=k;
    end
end 

% construct the result
    species = struct();
    for i=1:length(seeds)
       species(i).seed_idx = seeds(i);
       species(i).idx = m(m(:, 2) == seeds(i), 1);
       species(i).len = length(species(i).idx);
       species(i).seed = pop(seeds(i), :);
       species(i).seed_fit = fit(seeds(i));
       cluster_idx(species(i).idx) = i;
    end
    
    for i = 1 : n
        temp = leader_node{i};
        if ~isempty(temp)
            leaders(i) = leaders(randi(length(temp)));
        else
            leaders(i) = i;
        end
        
    end
    
end