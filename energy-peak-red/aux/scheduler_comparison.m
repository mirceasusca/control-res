%% ------------------------------------------------------------------------
% 7. Random scheduling comparisons
% -------------------------------------------------------------------------

numRandomTests = 50;  % number of random schedules

P_rand_all = zeros(numRandomTests,1);     % random: all appliances fully random
P_rand_pairs = zeros(numRandomTests,1);   % random: schedule appliances in pairs

for trial = 1:numRandomTests
    
    %% --- Random schedule (all appliances independently randomized)
    x_rand = zeros(numConsumers, T);
    for i = 1:numConsumers
        d = d_slots(i);
        start_idx = randi([1, T-d+1]);
        x_rand(i, start_idx:start_idx+d-1) = 1;
    end
    
    total_load_rand = L(:)' + sum(A_kW .* x_rand,1);
    P_rand_all(trial) = max(total_load_rand);
    
    
    %% --- Random pairwise scheduling (consumers shuffled)
    x_rand_pair = zeros(numConsumers, T);
    perm = randperm(numConsumers);
    
    for idx = 1:2:numConsumers
        i = perm(idx);
        d = d_slots(i);
        start_idx = randi([1, T-d+1]);
        x_rand_pair(i, start_idx:start_idx+d-1) = 1;
        
        if idx+1 <= numConsumers
            j = perm(idx+1);
            d2 = d_slots(j);
            start_idx2 = randi([1, T-d2+1]);
            x_rand_pair(j, start_idx2:start_idx2+d2-1) = 1;
        end
    end
    
    total_load_pair = L(:)' + sum(A_kW .* x_rand_pair,1);
    P_rand_pairs(trial) = max(total_load_pair);
end

%% ------------------------------------------------------------------------
% Display statistics
% -------------------------------------------------------------------------
fprintf('\nRandom scheduling stats:\n');
fprintf('  Random-all mean peak   : %.3f kW\n', mean(P_rand_all));
fprintf('  Random-all min peak    : %.3f kW\n', min(P_rand_all));
fprintf('  Random-pairs mean peak : %.3f kW\n', mean(P_rand_pairs));
fprintf('  Random-pairs min peak  : %.3f kW\n', min(P_rand_pairs));
fprintf('  MILP-scheduled peak    : %.3f kW\n', P_sched);


%% ------------------------------------------------------------------------
% 8. Plot improvement vs random schedules
% -------------------------------------------------------------------------

figure;
hold on; grid on;

% Boxplots for random schedule peaks
boxplot([P_rand_all P_rand_pairs], ...
        'Labels',{'Random All','Random Pairs'});

% Overlay MILP result as horizontal line
yline(P_sched,'r-','LineWidth',2);
yline(P_unsched,'k--','LineWidth',1.5);

title('Peak Comparison: Random Schedules vs. MILP');
ylabel('Peak load [kW]');

legend('','',sprintf('MILP Peak = %.2f',P_sched), ...
             sprintf('Unscheduled Peak = %.2f',P_unsched), ...
             'Location','NorthEast');

hold off;