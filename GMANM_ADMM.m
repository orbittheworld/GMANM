function [f] = GMANM_ADMM(Y, P, K, sigma,tolerance,TuMethod)
    % Written by Silin Gao
    % 2024.3
    % Formulation:
    % Y=P.*(X+N)=P.*(AxCAz^H+N)
    % 1/2*||P.*X-Y||_F^2+tau/2/sqrt(NM)*(trace(Tu_x)+trace(Tu_y))
    % (Tu,X;X',V)>=0
    % INPUT:
    % Y : N*M 不完整观测值（SRA）
    % P : N*M 部分观测的采样矩阵，1的地方是布置了sensor，0的地方代表missing sensor
    % X : N*M X=Ax*C*Az^H 完整观测信号（URA）
    % Ax: N*K 水平向阵列流形
    % C : K*K
    % Az: M*K 垂直向阵列流形
    % tolerance : 可接受的误差水平

    %size of mesurements
    [N, M] = size(Y);

    if sigma > 0
        tau = sigma * (1 + 1 / log(M + N)) * sqrt((M + N) * log(M + N) + (M + N) * log(4 * pi * log(M + N)));
    end

    %parameters
    alpha = 10;
    mu = 2;
    rho = 1.0; % penalty parameter in augmented Lagrangian
    tol_abs = tolerance;% 1E-5
    tol_rel = tolerance;
    maxiter = 1e5;

    %initialization
    X0 = Y;%N*M
    Tu_x0 = X0 * X0';%N*N
    Tu_y0 = X0' * X0;%M*M

    ZOld = [Tu_x0, X0; X0', Tu_y0]; %actually the Z matrix
    Lambda = zeros(N+M); %Lagrangian multiplier

    for iter = 1:maxiter

        Zs = ZOld(1:N, N + 1:N+M);
        Zy = ZOld(N + 1:N + M, N + 1:N + M);
        Zx = ZOld(1:N, 1:N);
        Lambdas = Lambda(1:N, N + 1:N+M);
        Lambday = Lambda(N + 1:N + M, N + 1:N + M);
        Lambdax = Lambda(1:N, 1:N);

        %update the variables X,Tux and Tuy
        X = (2 * rho * Zs + 2 * Lambdas + Y)./(2 * rho * ones(N,M) + P);
        Tu_x = Tproj(Zx + 1 / rho * Lambdax - tau / 2 / rho / sqrt(N*M) * eye(N));
        Tu_y = Tproj(Zy + 1 / rho * Lambday - tau / 2 / rho / sqrt(N*M) * eye(M));

        %Ztemp is the matrix that should be psd.
        Ztemp = [Tu_x, X; X', Tu_y];

        %projection of Q onto the semidefinite cone
        Q = Ztemp - 1 / rho * Lambda;
        Z = PSD(Q);

        Tu_x = Z(1:N, 1:N);
        Tu_y = Z(N + 1:N + M, N + 1:N + M);
        %update the variables Lambda
        Lambda = Lambda + rho * (Z - Ztemp);
        
        % stopping criterion (Boyd's rule)
        res_prim = Z - Ztemp;
        Z_dif = Z - ZOld;
        res_dual = rho * Z_dif;
        Lambda_dual = Lambda;
    
        err_prim = norm(res_prim, 'fro');
        err_dual = norm(res_dual,'fro');
    
        tol_prim = (M+N)*tol_abs + tol_rel*max(norm(Z,'fro'), norm(Ztemp,'fro'));
        tol_dual = (M+N)*tol_abs + tol_rel*norm(Lambda_dual,'fro');
    
        if err_prim < tol_prim && err_dual < tol_dual
            break;
        end
%         if norm(Tu-ZT,'fro')/ norm(ZT,'fro')< 1e-3
%             break;
%         end

        ZOld = Z;

        % a varying penalty parameter scheme
        if err_prim > alpha * err_dual
            rho = rho * mu;
        elseif err_prim < err_dual / alpha
            rho = rho / mu;
        end

    end

    %% Vandermonde Decomposition
    if strcmp(TuMethod, 'music')
        fx_e = rootMUSIC(Tu_x, K); %(-0.5,0.5)
        fy_e = rootMUSIC(Tu_y, K); %(-0.5,0.5)
    elseif strcmp(TuMethod, 'prony')
        fx_e = PRONY(Tu_x, K);
        fy_e = PRONY(Tu_x, K);
    else
        fx_e = MatrixPencil(Tu_x, K);
        fy_e = MatrixPencil(Tu_y, K);
    end

    %% Pairing
    v_N = [0:(N - 1)]';
    v_M = [0:(M - 1)]';
    A_x = [];

    for index = 1:K
        A_x = [A_x, exp(1i * 2 * pi * fx_e(index) * v_N)];
    end

    Dx = diag(diag(abs(pinv(A_x) * Tu_x * pinv(A_x)')));
    V = inv(Dx) * pinv(A_x) * X;
    fx_sort = zeros(1, K);

    for index = 1:K
        c = abs(V * exp(1i * 2 * pi * fy_e(index) * v_M));
        [~, ii] = max(c);
        fx_sort(index) = fx_e(ii);
    end

    f = [fx_sort; fy_e];

end
