Hc = tf(500*[1,410],conv([1,210],[1,320]));
Te = min([1/210,1/320])/10;
format long
Hd = c2d(Hc,Te,'zoh');
[num,den]=tfdata(Hd,'v');
Hd5=tf([0.15331,-0.13487],[1,-1.84131,0.84736],Te);
Hd4=tf([0.1533,-0.1348],[1,-1.8413,0.8473],Te);
Hd3=tf([0.153,-0.134],[1,-1.841,0.847],Te);
step(Hd,Hd5,Hd4,Hd3,0.04),shg % timpul de simulare de 0.04[s]
legend('15z','5z','4z','3z');
