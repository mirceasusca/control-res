Hdes = tf(1295.65, [0.1198, 1, 0]);
clf, nichols(Hdes), grid, hold on

%%
H0 = feedback(Hdes,1);
figure
bodemag(H0,{10,1e4}), grid
% step(H0), grid

%%
hold on

%%
yline(-3,'r')
yline(0,'k')

%%
Mdb = 0.00001; % modulul evidentiat
% Mdb = 21.9; % modulul evidentiat
% Mdb = -3; % modulul evidentiat
M = 10^(Mdb/20);
Ts = 0.0001;

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

%%
H = tf(-2*[1,-2],[1,6,0])
margin(H)

%%
H = tf(8*[1,9],[1,50,0],'iodelay',2)
margin(H)

%%
w = 0.825;
f = pi/2+atan(w/9)-atan(w/50)-2*w

%%
% R1 = 5.5; 
% R2 = 5.5;
% C1 = 10;
% C2 = 10;
% H = tf(1,[R1*R2*C1*C2,(R1*C1+R2*C2+R2*C1),1])
% zpk(H)

% Km = 334.89;
% Ki = 3.869;
% Tm = 0.1198;
% Te = 0.0025;
% H = tf(Km*Ki,conv([Te,1],[Tm,1,0]))
% zpk(H)

H = tf(3.333*[1,0,3167],conv([1,2.864,49.85],[1,17.14,3388]))
zpk(H)


subplot(2,2,[1,3])
bode(H), grid

subplot(2,2,2)
nyquist(H)

subplot(2,2,4)
nichols(H)