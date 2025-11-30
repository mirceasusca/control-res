% perioada de esantionare
Te=0.2;
% functia de transfer cu zoh
Hd=c2d(tf(1,[1 3 2]),Te,'zoh');
% extragere numarator si numitor
[num,den]=tfdata(Hd,'v');
% extragerea parametrilor pentru implementare
a2=den(3); a1=den(2);
b2=num(3); b1=num(2);
% conditii initiale
y(1)=0;
y(2)=b1;
% numar de esantioane
n=30;
% semnalul de intrare
u=ones(1,n);
% implementarea sistemului numeric
for k=3:n
    y(k)=b2*u(k-2)+b1*u(k-1)-a2*y(k-2)-a1*y(k-1);
end
% timp de simulare
te=0:Te:(n-1)*Te;
% comparatie intre ce s-a calculat si ce returneaza MATLAB
stairs(te,y,'r*');hold;step(Hd)