close all 
% Load data from file
t = treapta0(:,1);
u = treapta0(:,2);
y = treapta0(:,4);
% plot data
plot(t,[u, y]); grid ; shg

% extract the values 
    % indeces for stationary values
    i1 = 436; % start index for initial value
    i2 = 498; % stop index for initial value
    j1 = 764; % start index for stationary value
    j2 = 829; % stop index for stationary value
    imax = 581; % first maximum index
    i3 = 504; % settling time indeces  
    i4 = 670;
    
u0 = mean(u(i1:i2));
y0 = mean(y(i1:i2));
ust = mean(u(j1:j2)); 
yst = mean(y(j1:j2));
rng(1)
k = 2.2;
zeta = 0.4;
wn = 1.2e3; 
A = [0 1; -wn^2 -2*zeta*wn]; B = [0; k*wn^2]; C = [1 0]; D = 0;

y1 = lsim(ss(A,B,C,D),u-ust,t)+k*ust+0.5*rand(length(t),1)-0.25;
plot(t,u,'LineWidth',2); hold on
plot(t,y1,'LineWidth',2);
% plot(t(559),y1(559),'xk','MarkerSize',15,'MarkerFaceColor','r','LineWidth',3); 
xline(t(501),'--','LineWidth',2);
xline(t(559),'--','LineWidth',2);
grid minor; shg
hold on 
% plot(t(501),y1(501),'Marker','o','MarkerSize',2,'MarkerFaceColor','r')
% plot(t, ust*ones(1,length(t)),'--r','LineWidth',2);
% plot(t, u0*ones(1,length(t)),'--r','LineWidth',2);
% plot(t, k*ust*ones(1,length(t)),'--k','LineWidth',2);
% plot(t, k*u0*ones(1,length(t)),'--k','LineWidth',2);
% legend('u','y','u_{st}','u_0','y_{st}','y_0');
% legend('u','y','y_{max}','y_{st}','y_0');
legend('u','y');
xlabel('Timp[s]'); ylabel('u/y');
xlim([t(1) t(end)])
%%
% ysim = lsim(tf(k*wn^2,[1 2*zeta*wn wn^2]),u,t);
ysim = lsim(ss(A,B,C,D),u-ust,t)+k*ust;
figure;
plot(t,u,t,y1,t,ysim,'LineWidth',2); grid minor; shg
xlabel('Timp[s]'); ylabel('u/y/y_{sim}');
legend('u','y','y_{sim}');
xlim([t(1) t(end)])
%%

ymax = y(imax);
% settling time
tr = t(i4)-t(i3);
% overshoot
sigma = (ymax-yst)/(yst-y0);

% gain factor
k = (yst-y0)/(ust-u0);
% damping factor
zeta = -log(sigma)/sqrt(pi^2+log(sigma)^2);
% natural frequency
wn = 4/zeta/tr;

% construct the tf
H = tf(k*wn^2,[1 2*zeta*wn wn^2]);
ysim = lsim(H,u,t);
figure
plot(t,[u,y,ysim]); 

% simulate from nonzero initial conditions
A = [0 1; -wn^2 -2*zeta*wn]; B = [0; k*wn^2]; C = [1 0]; D = 0;
ysim1 = lsim(A,B,C,D,u,t,[y(1) 0]);
figure
plot(t,[u,y,ysim1], 'LineWidth', 1.5); shg; grid minor

xlim([-0.005 0.01])
legend('u','y','y_{sim}')
xlabel('Time[s]'); ylabel('u/y')

% evaluate the results
J = norm(y-ysim1)
e_MPN = norm(y-ysim1)/norm(y-mean(y))