H = tf([-1,2],[1,1,1,0])
nyquist(H)
shg

%%
H = tf([1,0.5],[1,1,1,0])
% H = tf([1,1],[1,20,0])
bode(H)

%%
H = tf([2,0,1],[1,1,1])
nyquist(H)

%%
% H = tf([1,4],[2,2],'iodelay',0.2);
% nyquist(H,logspace(-2,2,1000))
% shg

% H = tf([4],[2,2],'iodelay',0.2);
% figure, nyquist(H,logspace(-2,2,1000))
% shg

% H = tf([1,1],[1,20,0],'iodelay',0.5);
H = tf([1,7],[1,2,0]);
% figure, nyquist(H,logspace(-2,2,1000))
hold on, bode(H)
shg