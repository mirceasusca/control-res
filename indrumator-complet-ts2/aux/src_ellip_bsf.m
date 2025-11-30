% Parametrii filtrului
Fe = 2000;          % frecventa de eșantionare [Hz]
FN = Fe/2;          % frecventa Nyquist
N = 8;              % ozrdinul filtrului
Wp = [300 700]/FN;  % banda de trecere (bandpass)
Ws = [250 750]/FN;  % benzi de oprire (bandstop)
rp = 1;    % ondulatie maxima un banda de trecere [dB]
rs = 60;   % atenuare minima [dB] in banda de oprire
[B, A] = ellip(N, rp, rs, Ws, 'stop');
% Vizualizarea răspunsului în frecvență
freqz(B, A, 1024, Fe);
title('Diagrama Bode, FOB eliptic');

H = tf(B,A,1/Fe)
