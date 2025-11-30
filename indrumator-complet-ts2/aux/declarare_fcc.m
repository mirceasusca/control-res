num=[1, 11, 30]; den=[1, 9, 26, 24];
b0=1;b1=11;b2=30;a0=9;a1=26;a2=24;
%
[A,b,c,d]=tf2ss(num,den); % FCC
sys=ss(A,b,c,d); 
