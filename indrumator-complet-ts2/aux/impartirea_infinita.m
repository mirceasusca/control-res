% perioada de esantionare
Te=0.2;
% functia de transfer cu ZOH
Hd=c2d(tf(1,[1 3 2]),Te,'zoh');
% extragere numarator si numitor
[num,den]=tfdata(Hd,'v');
% numarul termenilor pentru aproximare
N = 10; 
% extragerea parametrilor
[h,~] = deconv([num zeros(1,N)],den);
% numar de esantioane
n=30;
% initializarea iesirii
y = zeros(1,n);
% semnalul de intrare
u=[zeros(1,N) ones(1,n-N)]; 
for k=N+1:n
    for j=1:N
        y(k)=y(k)+h(j)*u(k-j);
    end
end
% timpul de simulare
t=0:Te:(n-N-2)*Te;
% comparatie intre ce s-a calculat si ce returneaza MATLAB
stairs(t,y(N+2:n),'r*');hold;step(Hd)