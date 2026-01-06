%% PoC: 1D toy dataset (10 points, 2 clusters) + "idealCVI" (chance-corrected pairwise agreement)
% Compatible with older MATLAB versions (no string objects needed).

clear; clc; close all;

%% 1) Data (column vectors)
X = [-1.2; -1.0; -0.8; -0.6; -0.4;  0.4;  0.6;  0.8;  1.0;  1.2];
y_true = [ 0;   0;   0;   0;   0;   1;   1;   1;   1;   1];

N = numel(X);

%% 2) A few candidate labelings
Y = cell(0,1);
Y_name = cell(0,1);

% Correct labeling
Y{1} = y_true;
Y_name{1} = 'y_true';

% Random labeling (example)
rng(1);
Y{2} = randi([0 1], N, 1);
Y_name{2} = 'random';

% Split by threshold at 0 (same as true here, but useful if you change X)
Y{3} = double(X >= 0);
Y_name{3} = 'threshold@0';

% Wrong: flip one point near boundary
Y{4} = y_true;
Y{4}(5) = 1;   % move x=-0.4 into cluster 1
Y_name{4} = 'one_flip';

% Wrong: alternating labels
Y{5} = mod((1:N)', 2);
Y_name{5} = 'alternating';

%% 3) Plot labelings
figure('Color','w','Name','1D clustering labelings');
tiledlayout(numel(Y), 1, 'TileSpacing', 'compact');

for k = 1:numel(Y)
    nexttile;
    yk = Y{k};

    idx0 = (yk == 0);
    idx1 = (yk == 1);

    plot(X(idx0), zeros(sum(idx0),1), 'o', 'MarkerSize', 7, 'LineWidth', 1.5); hold on;
    plot(X(idx1), zeros(sum(idx1),1), 's', 'MarkerSize', 7, 'LineWidth', 1.5);
    yline(0, ':');
    grid on;

    xlim([min(X)-0.2, max(X)+0.2]);
    ylim([-0.5, 0.5]);
    yticks([]);
    title(Y_name{k}, 'Interpreter','none');
    legend({'label 0','label 1'}, 'Location','eastoutside');
end

%% 4) Compute scores with idealCVI()
fprintf('--- Scores (idealCVI: chance-corrected pairwise agreement, normalized to [0,1]) ---\n');
for k = 1:numel(Y)
    score = idealCVI(y_true, Y{k});  % X not needed for this first external-style CVI
    fprintf('%-12s : %.4f\n', Y_name{k}, score);
end

%% ===== Local functions =====

function score01 = idealCVI(y_true, y_pred)
% idealCVI  Chance-corrected pairwise agreement normalized to [0,1].
%
% This is essentially:
%   agreement = fraction of pairs (i<j) for which [same/different] matches between y_true and y_pred
%   expected  = expected agreement under random permutation of y_pred labels (fixed cluster sizes)
%   score     = (agreement - expected) / (1 - expected)
%   score01   = max(0, min(1, score))  % clamp to [0,1] for convenience
%
% Properties:
% - score01(y_true, y_true) = 1
% - E_perm[ score ] = 0 (before clamping) when y_pred is random permutation w/ same counts
% - invariant to label permutations

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

    % Observed agreement on pairwise same/different
    A = pairwiseAgreement(y_true, y_pred);

    % Expected agreement under random permutation of y_pred (fixed label counts)
    % (Monte Carlo estimate; increase nperm for more precision.)
    nperm = 2000;
    Aexp = expectedAgreementPermutation(y_true, y_pred, nperm);

    % Chance-corrected normalization
    denom = (1 - Aexp);
    if denom <= 0
        % Degenerate case (shouldn't happen unless Aexp==1)
        score = 1;
    else
        score = (A - Aexp) / denom;
    end

    % Normalize to [0,1] by clamping
    if score < 0
        score01 = 0;
    elseif score > 1
        score01 = 1;
    else
        score01 = score;
    end
end

function A = pairwiseAgreement(y_true, y_pred)
% Fraction of pairs (i<j) whose co-membership matches between true and pred.

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
% Expected pairwise agreement when y_pred is randomly permuted (fixed cluster sizes).

    N = numel(y_true);
    acc = 0;

    for t = 1:nperm
        yp = y_pred(randperm(N));
        acc = acc + pairwiseAgreement(y_true, yp);
    end

    Aexp = acc / nperm;
end
