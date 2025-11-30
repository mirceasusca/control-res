H = tf([-1,2],[1,1,1],'iodelay',0.1)
% nyquist(H)
Hd = c2d(H,0.1,'zoh')
step(H,Hd)
shg

%%
% Parametrii filtrului
Fs = 1000;            % Frecventa de esantionare [Hz]
FN = Fs/2;            % Frecventa Nyquist
N = 4;                % Ordinul filtrului
wc = 150/FN;          % Frecventa de taiere normalizata (150 Hz)

% Proiectarea filtrului Butterworth trece jos
[B, A] = butter(N, wc, 'low');

% Vizualizarea răspunsului
freqz(B, A, 1024, Fs);
title('Raspuns in frecventa - filtru trece jos Butterworth');

%%
% Parametrii de bază
fs = 2000;           % Frecvența de eșantionare în Hz
nyq = fs/2;          % Frecventa Nyquist

% Ordine și parametri de filtrare
n = 8;               % Ordinul filtrului
Wp = [300 700]/nyq;  % Benzi de trecere - nefolosit, dar necesar pentru definit
Ws = [250 750]/nyq;  % Benzi de oprire (stop-band)

rp = 1;              % Ondulație maximă în banda de trecere (dB)
rs = 60;             % Rejeție minimă (dB) în banda de oprire

% Proiectarea filtrului elliptic stop-band
[B, A] = ellip(n, rp, rs, Ws, 'stop');

% Vizualizarea răspunsului în frecvență
figure;
freqz(B, A, 1024, fs);
title('Răspuns în frecvență - filtru elliptic band-stop');