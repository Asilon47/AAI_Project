clear; clc; close all;
load('digits.mat');
X = double(digits.image);
[D,N] = size(X);
% media por cada píxel
mean_d = mean(X, 2);    
% desviación por cada píxel
std_d = std(X, 0, 2);   
% evitar división por cero
std_d(std_d == 0) = 1;
% X_normalizado
Xn = (X - mean_d) ./ std_d;

fprintf('Media global después de normalizar: %.4f\n', mean(Xn(:)));
fprintf('Desviación estándar global después de normalizar: %.4f\n', std(Xn(:)));


% guardar datos, la media y desviación
save('digits_normalized.mat','Xn','mean_d','std_d','-v7.3');