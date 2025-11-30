function tau = Gamma_d(xk, A, B, K, P, lambda)
%GAMMA_D compute next execution time

u = -K*xk;
Vk = xk' * P * xk; % functia Lyapunov la momentul tk
tau = 0;
step = 1e-3; % pas de cautare

maxTau = 1; % parametru de design. Vad daca ma mai joc cu el

while tau < maxTau
    % prezicem starea viitoare
    Nt = 40;
    s = linspace(0, tau, Nt);
    integ = zeros(2, 1);

    for j = 1:Nt
        ds = s(2) - s(1);
        integ = integ + expm(A*(tau-s(j)))*B*u*ds;
    end
    x_tau = expm(A*tau)*xk + integ;

    % % prezicem starea la timp tau cu ode45
    % xdot = @(t, x)(A*x + B*u);
    % [~, x_ode] = ode45(xdot, [0, tau], xk);
    % x_tau = x_ode(end, :)';   % starea la t = tau
    
    if (x_tau'*P*x_tau) > Vk*exp(-lambda*tau)
        break; % s-a incalcat conditia Lyapunov
    end

    tau = tau + step;
end

end

