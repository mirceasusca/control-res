num=[1, 11, 30]; den=[1, 9, 26, 24];
b0=1;b1=11;b2=30;a0=9;a1=26;a2=24;
[A,b,c,d]=tf2ss(num,den);
%
sistem=ss(A,b,c,d);
x0=[0,0,0];
t=0:0.01:5; u=ones(1,length(t)); % semnal treapta
[y,t,x]=lsim(sistem,u,t,x0); % raspunsul sistemului afisat grafic
subplot(121);plot(t,y); legend('y');grid;
subplot(122);plot(t,x); legend('x_1','x_2','x_3');grid

%%
num_ext=[num, zeros(1,3)]; % first three Markov parameters
[gv,~]=deconv(num_ext,den)

gamma = [1 0 0; a1 1 0; a2 a1 1]\[b1;b2;b3]
g1=gamma(1);g2=gamma(2);g3=gamma(3);
 
%%
ord = length(den)-1;
n = ord-1;
Ac = [-den(2:end); [eye(2), zeros(n,1)]];
Bc = [1; zeros(n,1)];
Cc = num;
Dc = 0;
sys_c= ss(Ac,Bc,Cc,Dc);

%%
Te = 0.1;
[numd,dend] = c2dm(num,den,Te,'tustin');
sys_d = tf2ss(numd,dend);

%%
H2 = tf([1,1,1,1],[1,1,3,1]);
[num2,den2]=tfdata(H2,'v');
[Q,R]=deconv(num2,den2);
%
Ac=[-1,-3,-1; 1 0 0; 0 1 0]; Bc = [1; 0; 0]; Cc =[0, -2, 0]; Dc=1;
sys_2 = ss(Ac,Bc,Cc,Dc);
step(sys_2,H2,'d')