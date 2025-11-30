R = 2.2; 
L = 1e-4;
C = 4.7e-6; 

% declararea functiei de transfer
H = tf([1/R/C 0],[1 1/R/C 1/L/C]);

% valoarea pulsatiei medii 
w0 = 1/sqrt(L*C);

% raspunsul sistemului la intrare sinusolidala de pulsatie w0/10, w0 si
% w0*10 (o decada in stanga si o decada in dreapta pulsatiei centrale)

% simulare pe 10 perioade
T1 = 2*pi/(w0/10); 
t1 = 0:T1/100:10*T1;
u1 = sin(w0/10*t1);
y1 = lsim(H,u1,t1);

T2 = 2*pi/w0; 
t2 = 0:T2/100:10*T2;
u2 = sin(w0*t2);
y2 = lsim(H,u2,t2);

T3 = 2*pi/(10*w0); 
t3 = 0:T3/100:10*T3;
u3 = sin(w0*10*t3);
y3 = lsim(H,u3,t3);

figure 

subplot(311); plot(t1,y1,'LineWidth',1.5); hold on
yline(sqrt(2)/2,'r','LineWidth',1.5), ylim([-1,1])
title('Raspunsul la intrarea u(t)=sin(w_0/10*t)')
xlabel('Timp [s]'), ylabel('u_c [V]')

subplot(312); plot(t2,y2,'LineWidth',1.5); hold on
yline(sqrt(2)/2,'r','LineWidth',1.5), ylim([-1,1])
title('Raspunsul la intrarea u(t)=sin(w_0*t)')
xlabel('Timp [s]'), ylabel('u_c [V]')

subplot(313); plot(t3,y3,'LineWidth',1.5); hold on
yline(sqrt(2)/2,'r','LineWidth',1.5), ylim([-1,1])
title('Raspunsul la intrarea u(t)=sin(w_0*10*t)')
xlabel('Timp [s]'), ylabel('u_c [V]')

