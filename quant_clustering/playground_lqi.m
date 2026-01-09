%% 2D SISO DT system
clear; clc; close all;

Ts = 0.1;

% A, B controllable
A = [1.2,0.4;
    -0.3,0.9];
B = [1;0.5];
C = [1,1];
D = 0;

n = size(A,1);
m = size(B,2); 
p = size(C,1);

sys = ss(A,B,C,D,Ts);
pole(sys)
zero(sys)
% figure,step(sys,10)

%% LQI design
Q = diag([10,2,20])/10;
R = 0.5;
N = zeros(n+p,m);
[Klqi,S,e] = lqi(sys,Q,R,N);
K = -Klqi;
lqrStruct = struct('Q',Q,'R',R,'N',N,'K',K);

%% Closed-loop system (ideal)
Kx = K(1:n);
Kz = K(n+1:end);

Acl = [A+B*Kx,B*Kz;
    -Ts*C,eye(p)];
Bcl_ref = [zeros(n,m);Ts*eye(m)];
Ccl = [C,zeros(p)];
Dcl = zeros(p);

syscl = ss(Acl,Bcl_ref,Ccl,Dcl,Ts);
figure,step(syscl)

%%
x_tilde_0 = [1,0,0];
tfin = 30;
ref = @(t) 0.2;
% ref = @(t) 0;
% ref = @(t) t>=2;
qStruct = struct(...
    'qU',@(x)qIdeal(x),...
    'qX',@(x)qIdeal(x),...
    'qY',@(x)qIdeal(x)...
    );
eqStruct = struct;  % initial gol, doar sa faca simularea
info = sim_lqi(sys,lqrStruct,x_tilde_0,tfin,ref,qStruct,eqStruct);

%
t = (0:size(info.XK,2)-1)*Ts;

figure
subplot(221)
% plot(t,info.RK,t,info.XK)
plot(t,info.XK)
ylabel('X')
subplot(222)
plot(t,info.RK,t,info.YK)
ylabel('R/Y')
subplot(223)
plot(t,info.UK)
ylabel('U')
subplot(224)
plot(t,info.ZK)
ylabel('Z')
shg

% acum avem echilibru care se poate folosi mai incolo
xi_eq = [info.XK(:,end);info.ZK(:,end)];
u_eq = info.UK(:,end);
y_eq = info.YK(:,end);
%
eqStruct = struct('xi_eq',xi_eq,...
    'u_eq',u_eq,'y_eq',y_eq);

info.J

%% Monte Carlo dataset creation for Kmeans clustering
Nexp = 200;

% domeniul de variatie al conditiilor initiale
x0min = [-2,-2];
x0max = [2,2];

Xdataset = [];

for k = 1:Nexp
    x10 = x0min(1)+(x0max(1)-x0min(1))*rand(1);
    x20 = x0min(2)+(x0max(2)-x0min(2))*rand(1);
    x_tilde_0 = [x10;x20;0];
    info = sim_lqi(sys,lqrStruct,x_tilde_0,tfin,ref,qStruct,eqStruct);
    Xdataset = [Xdataset; info.XK'];
    info.J
end

%% Precision and Kmeans optimization
range_x1 = [min(Xdataset(:,1)),max(Xdataset(:,1))];
range_x2 = [min(Xdataset(:,2)),max(Xdataset(:,2))];

n_bits = 7;
num_clusters_K1 = 2^n_bits;
num_clusters_K2 = 2^n_bits;

qUnif_x1 = diff(range_x1)/2^n_bits;  % profit de intregul domeniu parcurs
qUnif_x2 = diff(range_x2)/2^n_bits;

opts = statset('MaxIter', 500, 'Display', 'final');
[~, C1] = kmeans(Xdataset(:,1), num_clusters_K1, 'Replicates', 7, 'Options', opts);
[~, C2] = kmeans(Xdataset(:,2), num_clusters_K2, 'Replicates', 7, 'Options', opts);
C1 = sort(C1); C2 = sort(C2);

%% learned kmeans grid
figure(Name="Kmeans grid")
subplot(211)
stem(C1,ones(size(C1))),
ylabel('x1 grid')
subplot(212)
stem(C2,ones(size(C2)))
ylabel('x2 grid')

%% uniform quantizer grid
figure(Name="Uniform grid")
subplot(211)
stem([range_x1(1):qUnif_x1:range_x1(2)],ones(size([range_x1(1):qUnif_x1:range_x1(2)]))),
ylabel('x1 grid')
subplot(212)
stem([range_x2(1):qUnif_x2:range_x2(2)],ones(size([range_x2(1):qUnif_x2:range_x2(2)]))),
ylabel('x2 grid')

%%
% testat mai riguros pe mai multe stari initiale, nu doar schimbat manual
x_tilde_0 = [0;0.75;0]; % trei stari initiale, (x1,x2,z1=0), z0=0 la integratoare, de obicei

qStructKmeans = struct(...
    'qU',@(x)qIdeal(x),...
    'qX',@(x)qCentroid2D(x,C1,C2),...
    'qY',@(x)qIdeal(x)...
    );
qStructIdeal = struct(...
    'qU',@(x)qIdeal(x),...
    'qX',@(x)qIdeal(x),...
    'qY',@(x)qIdeal(x)...
    );
qStructUniform = struct(...
    'qU',@(x)qIdeal(x),...
    'qX',@(x)qUniform2D(x,qUnif_x1,qUnif_x2),...
    'qY',@(x)qIdeal(x)...
    );
info_kmeans = sim_lqi(sys,lqrStruct,x_tilde_0,tfin,ref,qStructKmeans,eqStruct);
info_ideal = sim_lqi(sys,lqrStruct,x_tilde_0,tfin,ref,qStructIdeal,eqStruct);
info_uniform = sim_lqi(sys,lqrStruct,x_tilde_0,tfin,ref,qStructUniform,eqStruct);

%
t = (0:size(info_ideal.XK,2)-1)*Ts;
figure
subplot(221)
plot(t,info_ideal.XK,t,info_kmeans.XK,t,info_uniform.XK)
% plot(t,info_ideal.XK(1,:),t,info_kmeans.XK(1,:),t,info_uniform.XK(1,:))
legend('ideal','kmeans','uniform')
ylabel('X')
subplot(222)
plot(t,info_ideal.YK,t,info_kmeans.YK,t,info_uniform.YK)
legend('ideal','kmeans','uniform')
yline(ref(0)) % linie suplimentara la referinta, dupa colorare
ylabel('R/Y')
subplot(223)
plot(t,info_ideal.UK,t,info_kmeans.UK,t,info_uniform.UK)
legend('ideal','kmeans','uniform')
ylabel('U')
subplot(224)
plot(t,info_ideal.ZK,t,info_kmeans.ZK,t,info_uniform.ZK)
legend('ideal','kmeans','uniform')
ylabel('Z')
shg

J_experiments = [
    info_ideal.J;
    info_kmeans.J;
    info_uniform.J
]

% ideal ce am vrea: J_ideal < J_kmeans < J_altceva
% ar trebui ca nimic sa nu depaseasca J_ideal; daca se intampla, inseamna
% ca ceva nu e corect construit ca principiu (problema de referinta).