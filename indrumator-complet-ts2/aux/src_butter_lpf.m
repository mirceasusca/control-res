% Parametrii filtrului
Fe = 1000;   % frecventa de esantionare [Hz]
FN = Fe/2;   % frecventa Nyquist
N = 4;       % ordinul filtrului
Fc = 150/FN; % Frecventa de taiere normalizata (150 Hz)
[B, A] = butter(N, Fc, 'low');

% Vizualizarea raspunsului
freqz(B, A, 1024, Fe);
title('Diagrama Bode, FTJ Butterworth');

H = tf(B,A,1/Fe)

