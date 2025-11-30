% perioada de esantionare
Te=0.2;
% functia de transfer cu zoh
Hd=c2d(tf(1,[1 3 2]),Te,'zoh');
% extragere numarator si numitor
[num,den]=tfdata(Hd,'v');
% extragerea parametrilor pentru implementare
[Res,Pole,Dir] = residue(num,den);
r1 = Res(1); r2 = Res(2);
z1 = Pole(1); z2 = Pole(2);
% conditii initiale
y1(1)=0;
y2(1)=0;
y(1) = y1(1)+y2(1);
% numar de esantioane
n=30;
% semnalul de intrare
u=ones(1,n);
% implementarea sistemului numeric
for k=2:n
    y1(k)=z1*y1(k-1)+r1*u(k-1);
    y2(k)=z2*y2(k-1)+r2*u(k-1);
    y(k)=y1(k)+y2(k);
end
% timpul de simulare
te=0:Te:(n-1)*Te;
% comparatie intre ce s-a calculat si ce returneaza MATLAB
stairs(te,y,'r*');hold;step(Hd)