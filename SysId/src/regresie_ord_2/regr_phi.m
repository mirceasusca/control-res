slope = 1.2;
offset = -2;
rndnoise = 0.9;
clf, [tk,xk] = generate_noisy_line(100,slope,offset,rndnoise);
plot(tk,xk,'o'), hold on, 
plot(tk,slope*tk+offset), 
xlabel('t'), ylabel('y(t)'), grid

% 
N = length(tk);
Phi = ones(N,2);
for k=1:N
    Phi(k,1) = tk(k);
end
Y = xk';

theta = linsolve(Phi,Y)

% theta_star = inv(Phi'*Phi)*Phi'*Y