clf, nichols(Hdes), grid, hold on
Mdb = 5.28; % modulul evidentiat
M = 10^(Mdb/20);
Ts = 0.01;

% jumatatea din dreapta de cerc
t = (pi+Ts):Ts:(2*pi);
X = -M^2/(M^2-1); 
Y = 0; 
R = abs(M/(M^2-1));
% cercurile din diagrama Nyquist
x = X+R*cos(t); y = Y+R*sin(t);
% echivalentul lor in diagrama Nichols
mag = db(sqrt(x.^2+y.^2));
ph = atan2(y,x);
plot(rad2deg(ph),mag,'r')  % poate trebuie adaugate/scazute cercuri!

% jumatatea din stanga de cerc
t = Ts:Ts:(pi);
X = -M^2/(M^2-1); 
Y = 0; 
R = abs(M/(M^2-1));
% cercurile din diagrama Nyquist
x = X+R*cos(t); y = Y+R*sin(t);
% echivalentul cercurilor din diagrama Nichols
mag = db(sqrt(x.^2+y.^2));
ph = atan2(y,x);
plot(-360+rad2deg(ph),mag,'r')  % poate trebuie adaugate/scazute cercuri!
shg
