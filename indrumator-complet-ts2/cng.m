H = tf([-1,2],[1,4,4,0])


nyquist(H,1/0.374973002245484*H,5*H)

%%
figure
rlocus(H,logspace(-5,3,10000))