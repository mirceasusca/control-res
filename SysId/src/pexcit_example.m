t = double(data.Scope_109.X.Data)';
u = double(data.Scope_109.Y(3).Data)';
w = double(data.Scope_109.Y(2).Data)';
th = double(data.Scope_109.Y(1).Data)';

plot(t,[u])

dat2 = iddata(w,u,t(2)-t(1));

u2 = [zeros(round(length(t)/2),1);ones(round(length(t)/2),1)];

dat3 = iddata(w,u2,t(2)-t(1));

Ped = pexcit(dat2)