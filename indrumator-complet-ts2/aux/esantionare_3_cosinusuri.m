% fie trei semnale de frecventa diferita
f1=1;f2=9;f3=11;
Te=0.1; % perioada de esantionare configurabila
t=0:Te:1;
x1=cos(2*pi*f1*t);
x2=cos(2*pi*f2*t);
x3=cos(2*pi*f3*t);
% se reprezinta grafic semnalele
plot(t,x1,'-o',t,x2,'-o',t,x3,'-o','linewidth',1.5,'markersize',5);shg;
legend(['T_{1,max}/T_e=',num2str(1/f1/Te)],...
	['T_{2,max}/T_e=',num2str(1/f2/Te)],...
	['T_{2,max}/T_e=',num2str(1/f3/Te)])
xlabel('Timp [s]')
ylabel('x(t) = cos(2 \pi f t)'), grid minor
