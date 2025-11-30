wn=1e3; zeta1=1e-3; zeta2=0.5;
Hc=tf([1,2*zeta1*wn,wn^2],[1,2*zeta2*wn,wn^2]);
Te=500e-6;
Hd1=c2d(Hc,Te,'zoh');
Hd2=c2d(Hc,Te,'tustin');
% utilizarea metodei Tustin cu prewarping
opt = c2dOptions('Method','tustin','prewarpfrequency',wn);
Hd3=c2d(Hc,Te,opt);
figure
bode(Hc,Hd1,Hd2,Hd3); 
legend('H(s)','zoh','tustin','prewarp');