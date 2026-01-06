%% PoC pipeline (1D, 10 points): in-sample f + out-of-sample phi + f_local(x) sweep
% Steps:
% 1) Given (X, y_true)
% 2) Run clustering algorithm: y = A_theta(X)
% 3) Choose geometry: d_X (Euclidean) or G(X) (kNN graph)
% 4) Evaluate in-sample: f(X, y_true, y)
% 5) Define out-of-sample assignment phi
% 6) For x_new: yhat_new = phi(X, y, x_new)
% 7) Evaluate local correctness f_local(x_new) (optional)
% 8) Study smoothness by sweeping x_new (and optionally perturbing X)

clear; clc; close all;

%% 1) Given (X, y_true)
X = [-1.2; -1.0; -0.8; -0.6; -0.4;  0.4;  0.6;  0.8;  1.0;  1.2];
y_true = [ 0;   0;   0;   0;   0;   1;   1;   1;   1;   1];
N = numel(X);

%% 2) Run clustering algorithm: y = A_theta(X)
% Simple placeholder clustering algorithm: k-means in 1D with K=2
% (You can swap this with any clustering method)
K = 2;
y = kmeans1D(X, K);

%% 3) Choose geometry: d_X or G(X)
% (A) Euclidean distance is implicit in this 1D example
% (B) A simple kNN graph G(X) (used in graph-based phi)
k = 2;
W = buildKNNGraph1D(X, k); % adjacency weights = |xi-xj| on edges, Inf otherwise

%% 4) Evaluate in-sample: f(X, y_true, y)
f_global = idealCVI(y_true, y);
fprintf('In-sample global f(X,y_true,y): %.4f\n', f_global);

%% 5) Define out-of-sample assignment phi
% We'll compare two phi choices:
%   phi_nn: nearest-neighbor label
%   phi_graph: attach to kNN graph and use shortest-path distance to clusters

%% 6) Sweep x_new: yhat_new = phi(X,y,x_new)
x_grid = linspace(min(X)-0.5, max(X)+0.5, 401)';

yhat_nn    = zeros(size(x_grid));
yhat_graph = zeros(size(x_grid));
f_loc_nn    = zeros(size(x_grid));
f_loc_graph = zeros(size(x_grid));

% Truth field pi(x): logistic around boundary at 0 (cluster 0 left, cluster 1 right)
r0 = 0.0;
beta = 12.0;
pi0 = @(x) 1 ./ (1 + exp(beta*(x - r0)));         % ~1 on left, ~0 on right
pi  = @(x) [pi0(x); 1 - pi0(x)];                  % 2x1 probability vector

tau = 0.15; % temperature for converting distances->probabilities in rho(x)

for t = 1:numel(x_grid)
    x_new = x_grid(t);

    % NN phi
    [yhatA, rhoA] = phi_nn(X, y, x_new, tau);
    yhat_nn(t) = yhatA;

    % Graph phi
    [yhatB, rhoB] = phi_graph(X, y, x_new, W, tau);
    yhat_graph(t) = yhatB;

    % 7) Local correctness (optional): f_local(x_new)
    f_loc_nn(t)    = f_local_js(pi(x_new), rhoA);
    f_loc_graph(t) = f_local_js(pi(x_new), rhoB);
end

%% 8) Study smoothness/sensitivity by varying x_new (plots)
figure('Color','w','Name','Local f_{local}(x) sweep');
plot(x_grid, f_loc_nn, 'LineWidth', 1.5); hold on;
plot(x_grid, f_loc_graph, 'LineWidth', 1.5);
grid on;
xlabel('x_{new}');
ylabel('f_{local}(x_{new})');
title('Local correctness vs x_{new}');
legend({'\phi = NN', '\phi = graph'}, 'Location', 'best');
ylim([-0.05, 1.05]);

figure('Color','w','Name','Out-of-sample hard labels');
plot(x_grid, yhat_nn, '.', 'MarkerSize', 7); hold on;
plot(x_grid, yhat_graph, '.', 'MarkerSize', 7);
grid on;
xlabel('x_{new}');
ylabel('\hat{y}_{new}');
title('Out-of-sample assignment vs x_{new}');
legend({'\phi = NN', '\phi = graph'}, 'Location', 'best');
yticks([0 1]); ylim([-0.25 1.25]);

%% (Optional) Smoothness metric: finite-difference sensitivity of f_local
df_nn    = abs(diff(f_loc_nn)) ./ abs(diff(x_grid));
df_graph = abs(diff(f_loc_graph)) ./ abs(diff(x_grid));

figure('Color','w','Name','Sensitivity |df/dx|');
plot(x_grid(1:end-1), df_nn, 'LineWidth', 1.5); hold on;
plot(x_grid(1:end-1), df_graph, 'LineWidth', 1.5);
grid on;
xlabel('x_{new}');
ylabel('|d f_{local} / d x| (finite diff)');
title('Local sensitivity proxy');
legend({'\phi = NN', '\phi = graph'}, 'Location', 'best');

%% =========================
% Local functions
%% =========================

function y = kmeans1D(X, K)
% Minimal 1D k-means (K=2 recommended for this PoC). Returns labels {0,1,...,K-1}.
    X = X(:);
    N = numel(X);

    % init centroids: pick two extremes
    [~, iMin] = min(X);
    [~, iMax] = max(X);
    mu = [X(iMin); X(iMax)];
    if K ~= 2
        % simple general init: random samples
        rng(1);
        mu = X(randperm(N, K));
    end

    y = zeros(N,1);
    for it = 1:50
        % assign
        D = abs(X - mu');        % NxK (implicit expansion works in newer MATLAB)
        if size(D,2) ~= K
            % fallback for older MATLAB (no implicit expansion)
            D = zeros(N,K);
            for k = 1:K
                D(:,k) = abs(X - mu(k));
            end
        end
        [~, idx] = min(D, [], 2);
        y_new = idx - 1; % 0..K-1

        if isequal(y_new, y)
            break;
        end
        y = y_new;

        % update
        for k = 1:K
            if any(y == (k-1))
                mu(k) = mean(X(y == (k-1)));
            end
        end
    end
end

function score01 = idealCVI(y_true, y_pred)
% Chance-corrected pairwise agreement normalized to [0,1] using permutation baseline.
    y_true = y_true(:); y_pred = y_pred(:);
    N = numel(y_true);
    if numel(y_pred) ~= N, error('y_pred must have same length as y_true.'); end
    if N < 2, score01 = 1; return; end

    A = pairwiseAgreement(y_true, y_pred);
    nperm = 1000; % lower for speed; raise for precision
    Aexp = expectedAgreementPermutation(y_true, y_pred, nperm);

    denom = (1 - Aexp);
    if denom <= 0, score = 1; else, score = (A - Aexp) / denom; end
    score01 = min(1, max(0, score));
end

function A = pairwiseAgreement(y_true, y_pred)
    N = numel(y_true);
    totalPairs = N*(N-1)/2;
    agree = 0;
    for i = 1:N-1
        for j = i+1:N
            sameT = (y_true(i) == y_true(j));
            sameP = (y_pred(i) == y_pred(j));
            if sameT == sameP, agree = agree + 1; end
        end
    end
    A = agree / totalPairs;
end

function Aexp = expectedAgreementPermutation(y_true, y_pred, nperm)
    N = numel(y_true);
    acc = 0;
    for t = 1:nperm
        yp = y_pred(randperm(N));
        acc = acc + pairwiseAgreement(y_true, yp);
    end
    Aexp = acc / nperm;
end

function W = buildKNNGraph1D(X, k)
% Symmetric kNN graph adjacency matrix W with weights = |xi-xj| on edges, Inf otherwise.
    X = X(:); N = numel(X);
    W = Inf(N,N);
    for i = 1:N
        W(i,i) = 0;
        d = abs(X - X(i));
        [~, idx] = sort(d, 'ascend');
        nn = idx(2:min(k+1,N)); % exclude itself
        for j = nn(:)'
            W(i,j) = abs(X(i)-X(j));
            W(j,i) = W(i,j);
        end
    end
end

function [yhat, rho] = phi_nn(X, y, x_new, tau)
% Out-of-sample assignment by nearest neighbor label + soft rho from cluster distances.
    X = X(:); y = y(:);
    labels = unique(y);
    if numel(labels) ~= 2, error('phi_nn assumes 2 clusters.'); end

    [~, nn] = min(abs(X - x_new));
    yhat = y(nn);

    d0 = min(abs(X(y==labels(1)) - x_new));
    d1 = min(abs(X(y==labels(2)) - x_new));
    rho = soft2_from_dist(d0, d1, tau); % 2x1
end

function [yhat, rho] = phi_graph(X, y, x_new, W, tau)
% Out-of-sample assignment by attaching x_new to graph and using shortest-path to clusters.
    X = X(:); y = y(:);
    N = numel(X);
    labels = unique(y);
    if numel(labels) ~= 2, error('phi_graph assumes 2 clusters.'); end

    % Attach to k nearest points (Euclidean) for this PoC
    kAttach = 2;
    d = abs(X - x_new);
    [~, idx] = sort(d, 'ascend');
    nn = idx(1:kAttach);

    Waug = Inf(N+1, N+1);
    Waug(1:N,1:N) = W;
    Waug(N+1,N+1) = 0;

    for j = nn(:)'
        Waug(N+1,j) = abs(x_new - X(j));
        Waug(j,N+1) = Waug(N+1,j);
    end

    dist = dijkstra_dense(Waug, N+1); % distances from x_new node to all nodes

    d0 = min(dist(y==labels(1)));
    d1 = min(dist(y==labels(2)));
    rho = soft2_from_dist(d0, d1, tau);

    [~, idxMax] = max(rho);
    yhat = labels(idxMax);
end

function rho = soft2_from_dist(d0, d1, tau)
% Softmax(-d/tau) for two classes
    a0 = exp(-d0 / max(eps,tau));
    a1 = exp(-d1 / max(eps,tau));
    s = a0 + a1;
    rho = [a0; a1] / max(eps, s);
end

function f = f_local_js(p, q)
% Local correctness score in [0,1] using chance-normalized JS divergence:
% f = 1 - JS(p||q) / JS(p||u), u=[0.5;0.5]
    p = p(:); q = q(:);
    p = p / max(eps,sum(p));
    q = q / max(eps,sum(q));
    u = [0.5; 0.5];

    js_pq = js_div(p, q);
    js_pu = js_div(p, u);

    if js_pu < 1e-12
        f = 1 - min(1, js_pq / 1e-12);
    else
        f = 1 - (js_pq / js_pu);
    end
    f = min(1, max(0, f));
end

function js = js_div(p, q)
    m = 0.5*(p+q);
    js = 0.5*kl_div(p, m) + 0.5*kl_div(q, m);
end

function kl = kl_div(p, q)
    mask = (p > 0);
    kl = sum(p(mask) .* log(p(mask) ./ max(eps, q(mask))));
end

function dist = dijkstra_dense(W, src)
% Dijkstra for dense adjacency matrix W (Inf=no edge). Returns distances from src.
    n = size(W,1);
    visited = false(n,1);
    dist = Inf(n,1);
    dist(src) = 0;

    for it = 1:n
        dtmp = dist; dtmp(visited) = Inf;
        [~, u] = min(dtmp);
        if isinf(dist(u)), break; end
        visited(u) = true;

        for v = 1:n
            if ~visited(v) && isfinite(W(u,v))
                alt = dist(u) + W(u,v);
                if alt < dist(v)
                    dist(v) = alt;
                end
            end
        end
    end
end
