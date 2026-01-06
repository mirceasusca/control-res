%% PoC skeleton: global f(X,y_true,y_pred) + local f_local(x_new) with phi (NN vs graph)
% 1D, 2 clusters, includes:
%   - global f: chance-corrected pairwise agreement (external-style)
%   - local f_local(x): compares truth pi(x) vs predicted rho(x) with JS-divergence (chance-normalized)
%   - phi out-of-sample assignment:
%         (A) nearest-neighbor label
%         (B) graph-based label (attach to kNN graph, pick cluster with shortest-path / bottleneck score)
%
% Compatible with older MATLAB (avoid string objects).

clear; clc; close all;

%% ---------------------------
% 1) Ground-truth dataset
%% ---------------------------
X = [-1.2; -1.0; -0.8; -0.6; -0.4;  0.4;  0.6;  0.8;  1.0;  1.2];
y_true = [ 0;   0;   0;   0;   0;   1;   1;   1;   1;   1];

N = numel(X);

%% ---------------------------
% 2) Example "clustering algorithm output" (y_pred)
%    Replace this with your algorithm's labels
%% ---------------------------
y_pred = y_true;
y_pred(5) = 1; % one_flip example

%% ---------------------------
% 3) Global score f(X,y_true,y_pred)
%% ---------------------------
f_global = idealCVI(y_true, y_pred);
fprintf('Global f (idealCVI, [0,1]): %.4f\n', f_global);

%% ---------------------------
% 4) Truth field pi(x): logistic boundary
%    You can tune r0 and beta.
%    Here, boundary at 0 (since our 1D clusters are separated by 0).
%% ---------------------------
r0 = 0.0;        % boundary location
beta = 12.0;     % sharpness (larger = sharper)
% pi0(x) = sigmoid(beta*(r0 - x)) makes "left side" cluster 0, right side cluster 1.
% We'll define: cluster 0 for x < 0, cluster 1 for x > 0.
pi0 = @(x) 1 ./ (1 + exp(beta*(x - r0)));  % ~1 on left, ~0 on right
pi  = @(x) [pi0(x); 1 - pi0(x)];

%% ---------------------------
% 5) Build a kNN graph on X for graph-based phi
%% ---------------------------
k = 2; % in 1D, k=2 is usually enough (neighbors left/right)
W = buildKNNGraph1D(X, k); % adjacency matrix with edge weights = |xi-xj|

%% ---------------------------
% 6) Sweep x_new and compute local scores for two phi variants
%% ---------------------------
x_grid = linspace(min(X)-0.5, max(X)+0.5, 401)';

methodNames = {'NN', 'Graph'};
f_local_NN    = zeros(size(x_grid));
f_local_Graph = zeros(size(x_grid));
yhat_NN       = zeros(size(x_grid));
yhat_Graph    = zeros(size(x_grid));

tau = 0.15; % temperature for rho(x) soft assignment from distances (smaller => harder)

for t = 1:numel(x_grid)
    x_new = x_grid(t);

    % --- (A) phi: nearest neighbor hard label (then soften into rho)
    [yhatA, rhoA] = phi_nn(X, y_pred, x_new, tau);

    % --- (B) phi: graph-based: attach to kNN graph, use shortest path distances to clusters
    [yhatB, rhoB] = phi_graph(X, y_pred, x_new, W, tau);

    % --- local score f_local(x): compare pi(x) vs rho(x) with chance-normalized JS
    f_local_NN(t)    = f_local_js(pi(x_new), rhoA);
    f_local_Graph(t) = f_local_js(pi(x_new), rhoB);

    yhat_NN(t)    = yhatA;
    yhat_Graph(t) = yhatB;
end

%% ---------------------------
% 7) Plot local f_local(x) along the sweep
%% ---------------------------
figure('Color','w','Name','Local f_{local}(x) sweep');
plot(x_grid, f_local_NN, 'LineWidth', 1.5); hold on;
plot(x_grid, f_local_Graph, 'LineWidth', 1.5);
grid on;
xlabel('x_{new}');
ylabel('f_{local}(x)');
title('Local chance-normalized score vs x_{new}');
legend(methodNames, 'Location', 'best');
ylim([-0.05, 1.05]);

%% ---------------------------
% 8) Plot predicted hard labels along x_new (for intuition)
%% ---------------------------
figure('Color','w','Name','Predicted labels along sweep');
plot(x_grid, yhat_NN, '.', 'MarkerSize', 8); hold on;
plot(x_grid, yhat_Graph, '.', 'MarkerSize', 8);
grid on;
xlabel('x_{new}');
ylabel('\hat{y}_{new}');
title('Out-of-sample hard assignment along x_{new}');
legend(methodNames, 'Location', 'best');
yticks([0 1]);
ylim([-0.25 1.25]);

%% =========================
% Local functions
%% =========================

function score01 = idealCVI(y_true, y_pred)
% Chance-corrected pairwise agreement normalized to [0,1] (perm baseline).

    y_true = y_true(:);
    y_pred = y_pred(:);
    N = numel(y_true);

    if numel(y_pred) ~= N
        error('y_pred must have same length as y_true.');
    end
    if N < 2
        score01 = 1;
        return;
    end

    A = pairwiseAgreement(y_true, y_pred);

    nperm = 2000;
    Aexp = expectedAgreementPermutation(y_true, y_pred, nperm);

    denom = (1 - Aexp);
    if denom <= 0
        score = 1;
    else
        score = (A - Aexp) / denom;
    end

    % clamp
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
            if sameT == sameP
                agree = agree + 1;
            end
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
% Build symmetric kNN graph adjacency matrix for 1D points.
% W(i,j) = |Xi-Xj| if edge exists, else Inf. Diagonal is 0.

    X = X(:);
    N = numel(X);
    W = Inf(N,N);
    for i = 1:N
        W(i,i) = 0;
        d = abs(X - X(i));
        [~, idx] = sort(d, 'ascend');
        nn = idx(2:min(k+1,N)); % exclude itself (idx(1)=i)
        for j = nn(:)'
            W(i,j) = abs(X(i)-X(j));
            W(j,i) = W(i,j);
        end
    end
end

function [yhat, rho] = phi_nn(X, y_pred, x_new, tau)
% Nearest-neighbor out-of-sample assignment, plus soft rho via distance-to-cluster.

    X = X(:);
    y_pred = y_pred(:);

    % hard label: nearest sample label
    [~, nn] = min(abs(X - x_new));
    yhat = y_pred(nn);

    % soft rho: use distance to nearest point in each predicted cluster
    labels = unique(y_pred);
    if numel(labels) ~= 2
        error('phi_nn assumes 2 clusters for this PoC.');
    end
    d0 = min(abs(X(y_pred==labels(1)) - x_new));
    d1 = min(abs(X(y_pred==labels(2)) - x_new));

    % softmax on distances (smaller distance => larger prob)
    rho = soft2_from_dist(d0, d1, tau);
end

function [yhat, rho] = phi_graph(X, y_pred, x_new, W, tau)
% Graph-based out-of-sample assignment:
% 1) attach x_new to its k nearest nodes (based on existing graph degree implied by W)
% 2) compute shortest-path distances from x_new to each cluster (min over nodes in cluster)
% 3) convert those distances into soft rho, and hard label is argmax(rho)

    X = X(:);
    y_pred = y_pred(:);
    N = numel(X);

    % infer k from W by counting finite neighbors (excluding self)
    finiteCounts = sum(isfinite(W), 2) - 1;
    k = max(1, round(median(finiteCounts(finiteCounts>0))));
    k = min(k, N-1);

    % build augmented adjacency for N+1 nodes (last one is x_new)
    Waug = Inf(N+1, N+1);
    Waug(1:N,1:N) = W;
    Waug(N+1,N+1) = 0;

    % connect x_new to its k nearest points in Euclidean 1D (for PoC)
    d = abs(X - x_new);
    [~, idx] = sort(d, 'ascend');
    nn = idx(1:k);

    for j = nn(:)'
        Waug(N+1,j) = abs(x_new - X(j));
        Waug(j,N+1) = Waug(N+1,j);
    end

    % shortest paths from x_new node to all nodes
    dist = dijkstra_dense(Waug, N+1);

    labels = unique(y_pred);
    if numel(labels) ~= 2
        error('phi_graph assumes 2 clusters for this PoC.');
    end

    % cluster distance = min shortest-path distance to any node in that cluster
    d0 = min(dist(y_pred==labels(1)));
    d1 = min(dist(y_pred==labels(2)));

    rho = soft2_from_dist(d0, d1, tau);
    [~, idxMax] = max(rho);
    yhat = labels(idxMax);
end

function rho = soft2_from_dist(d0, d1, tau)
% Convert two nonnegative distances into 2-class soft probs via softmax(-d/tau)
    a0 = exp(-d0 / max(eps,tau));
    a1 = exp(-d1 / max(eps,tau));
    s = a0 + a1;
    rho = [a0; a1] / max(eps, s);
end

function f = f_local_js(p, q)
% Local score in [0,1] via chance-normalized Jensen-Shannon divergence:
%   f = 1 - JS(p||q)/JS(p||u), u=[0.5;0.5]
% Uses log base e; scaling cancels in ratio.

    p = p(:); q = q(:);
    p = p / max(eps,sum(p));
    q = q / max(eps,sum(q));

    u = [0.5; 0.5];

    js_pq = js_div(p, q);
    js_pu = js_div(p, u);

    if js_pu < 1e-12
        % truth is essentially uniform (maximally ambiguous); define as perfect if q ~ uniform too
        f = 1 - min(1, js_pq / 1e-12);
    else
        f = 1 - (js_pq / js_pu);
    end

    % clamp
    f = min(1, max(0, f));
end

function js = js_div(p, q)
% Jensen-Shannon divergence between two discrete distributions p and q.
    m = 0.5*(p+q);
    js = 0.5*kl_div(p, m) + 0.5*kl_div(q, m);
end

function kl = kl_div(p, q)
% KL(p||q) with safe handling of zeros.
    mask = (p > 0);
    kl = sum(p(mask) .* log(p(mask) ./ max(eps, q(mask))));
end

function dist = dijkstra_dense(W, src)
% Simple Dijkstra for dense adjacency matrix W (Inf = no edge).
% Returns distances from src to all nodes.

    n = size(W,1);
    visited = false(n,1);
    dist = Inf(n,1);
    dist(src) = 0;

    for it = 1:n
        % pick unvisited node with smallest dist
        dtmp = dist;
        dtmp(visited) = Inf;
        [~, u] = min(dtmp);
        if isinf(dist(u))
            break;
        end
        visited(u) = true;

        % relax edges u->v
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
