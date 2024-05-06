clc;
clear all;
close all;
warning off;
%% 参数设置
c = 3e8;
f=77e9;
lambda = c / f;
d = lambda/2;
%% Set Parameters
N = 11; % azi
M = 7; % ele
D_azi = (N-1)*lambda/2;
D_ele = (M-1)*lambda/2;
rho_azi = 1.22 * lambda / D_azi/pi*180;
rho_ele = 1.22 * lambda / D_ele/pi*180;

% TI-AWR2243
P = zeros(N,M);
P(1:8,1) = 1;
P(2:9,2) = 1;
P(3:10,5) = 1;
P(4:11,7) = 1;
locP = find(P==1);
NP = length(locP);
%% Frequencies
SNR = 20;
K = 4; 
theta = [60,80,90,130]/180*pi;
phi = [120,80,90,100]/180*pi;
f_r = (cos(theta).*sin(phi))/2;
f_t = cos(phi)/2;
c = [1,1,2,2];
%% Generate Array Manifolds
v_M = [0:(M - 1)]';
v_N = [0:(N - 1)]';
A_r = [];
A_t = [];

for ii = 1:K
    A_r = [A_r, exp(1i * 2 * pi * f_r(ii) * v_N)];
    A_t = [A_t, exp(1i * 2 * pi * f_t(ii) * v_M)];
end

H = A_r * diag(c) * A_t';

HW = awgn(H, SNR);
W = HW - H;
sigma = sqrt(sum(abs(W(:)).^2) / length(W(:)));
HW = P.*HW;
HWvec = HW(locP);

[Yf,Xf]= meshgrid(v_M,v_N);
v = [Xf(locP),Yf(locP)];
%% Decoupled ANM
tic; 
[f] = GMANM_ADMM(P.*HW, P, K, sigma,1e-5);
toc;
ef_r = f(1, :);
ef_t = f(2, :);

ephi = acosd(2*ef_t);
etheta = acosd(2*ef_r./sind(ephi));
figure(1);
scatter(etheta,ephi,'rv','filled','LineWidth',1.5);hold on;
scatter(theta/pi*180,phi/pi*180,'ko','LineWidth',1.5);hold off;
xlim([40,140]);ylim([40,140]);
xlabel('Azimuth angle, °');ylabel('Pitch angle, °');
legend('Estimates', 'Ground truth','location', 'southeast');
set(gca,'fontsize',15);
box on;
