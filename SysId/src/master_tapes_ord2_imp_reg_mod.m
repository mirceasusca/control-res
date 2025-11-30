t = scope130(:,1);
t = t-t(1);  %% CHIAR FACE DIFERENTA!
u = scope130(:,2);
y = scope130(:,4);

% plot(t, [u, y]); shg

figure
C = orderedcolors("gem");
colororder(C([1,5,2],:))

clf
plot(t, [u, y]); shg
xlabel('Timp[s]','FontSize',16)
ylabel('u/y','FontSize',16)
grid

% Extract indeces for K
i1 = 411;
i2 = 464;
% Evaluate the stationary values
ust = mean(u(i1:i2));
yst = mean(y(i1:i2));

K = yst/ust


%% Extract T1
% extract indeces for linear regression
i3 = 100;
i4 = 220;

t_log = t(i3:i4);
y_log = y(i3:i4);
figure
% plot(t_log, log(y_log-yst)); shg
plot(t_log, log(y_log-yst),'x','LineWidth',1.5,'MarkerSize',6); shg

% apply linear regression to extract T1
a = [sum(t_log.^2) sum(t_log); sum(t_log) length(t_log)];
b = [sum(log(y_log-yst).*t_log); sum(log(y_log-yst))];
sol = a\b;
T1 = -1/sol(1)
hold on
plot(t_log,sol(1)*t_log+sol(2),'LineWidth',2)
xlabel('Timp[s]','FontSize',16)
ylabel('ln(y(t)-y(\infty))','FontSize',16)
grid

%% Extract T2
% extract indeces for linear regression
i5 = 180;
i6 = 200;
hold on
xline(t(i5),'k--')
xline(t(i6),'k--')

%%

t_log2 = t(i5:i6);
y_log2 = y(i5:i6)-K*exp(-t_log2/T1);
figure
% plot(t_log2, y_log2); shg
plot(t_log2, log(y_log2-yst),'x','LineWidth',1.5,'MarkerSize',12); shg

% apply linear regression to extract T1
a = [sum(t_log2.^2) sum(t_log2); sum(t_log2) length(t_log2)];
b = [sum(log(y_log2-yst).*t_log2); sum(log(y_log2-yst))];
sol = a\b;
T2 = -1/sol(1)
hold on
plot(t_log2,sol(1)*t_log2+sol(2),'LineWidth',2)
xlabel('Timp[s]','FontSize',16)
ylabel('ln(y(t)-y(\infty))','FontSize',16)
grid

%%
% simulations
A = [0 1; -1/(T1*T2), -(1/T1 + 1/T2)];
B = [0; K/(T1*T2)];
C = [1, 0]; 
D = 0;
ysim = lsim(A,B,C,D,u,t,[y(1) 0]);
figure
plot(t,[u, y, ysim]);

eMPN = norm(y(1:502)-ysim(1:502))/norm(y(1:502)-mean(y(1:502)))
