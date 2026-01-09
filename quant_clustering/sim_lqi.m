function info = sim_lqi(sys,lqrStruct,x_tilde_0,tfin,ref,qStruct,eqStruct)

info = struct();

Ts = sys.Ts;

A = sys.a;
B = sys.b;
C = sys.c;
D = sys.d;

K = lqrStruct.K;

n = size(A,1);
m = size(B,2); 
p = size(C,1);

Kx = K(1:n);
Kz = K(n+1:end);

N = floor(tfin/Ts);

x_tilde_0 = x_tilde_0(:);

xk = x_tilde_0(1:n);
zk = x_tilde_0(n+1:end);

UK = zeros(m,N);
RK = zeros(m,N);
XK = zeros(n,N);
ZK = zeros(p,N);
YK = zeros(p,N);

for k = 1:N
    uk = Kx*qStruct.qX(xk)+Kz*zk;
    rk = ref(k*Ts);
    xknew = A*xk+B*qStruct.qU(uk);
    yk = C*xk;
    zknew = zk + Ts*(rk-qStruct.qY(yk));
    
    XK(:,k) = xk;
    ZK(:,k) = zk;
    xk = xknew;
    zk = zknew;
    RK(:,k) = rk;
    UK(:,k) = uk;
    YK(:,k) = yk;
end

% compute J = sum(x'*Q*x + u'*R*u + 2*x'*N*u)
% compensate for equilibrium point (for static step reference)
if isempty(fieldnames(eqStruct))
    eqStruct.xi_eq = zeros(n+p,1);
    eqStruct.u_eq = zeros(m,1);
end
J = 0;
for k = 1:N
    J = J + ([XK(:,k);ZK(:,k)]-eqStruct.xi_eq)'*lqrStruct.Q*([XK(:,k);ZK(:,k)]-eqStruct.xi_eq) + ...
        (UK(:,k)-eqStruct.u_eq)'*lqrStruct.R*(UK(:,k)-eqStruct.u_eq) + ...
        2*([XK(:,k);ZK(:,k)]-eqStruct.xi_eq)'*lqrStruct.N*(UK(:,k)-eqStruct.u_eq); 
end

info.XK = XK;
info.ZK = ZK;
info.UK = UK;
info.RK = RK;
info.YK = YK;
info.J = J;

end