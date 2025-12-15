function [L, S] = bridging_rpca(M, r, opts)
%BRIDGING_RPCA Alternating minimization for nonconvex RPCA (Zhang et al., 2021).
%   [L, S] = BRIDGING_RPCA(M, r, opts) solves the nonconvex RPCA problem
%   with rank-r factors using the alternating minimization scheme from
%   Algorithm 1 of "Bridging convex and nonconvex optimization in robust
%   PCA: Stability and optimality" (Annals of Statistics, 2021).
%
%   opts.lambda_reg - Tikhonov regularization on factors (default: 1e-2)
%   opts.tau        - soft-threshold level for sparse term (default: 0.1)
%   opts.step_size  - gradient step size for factor updates (default: 5e-3)
%   opts.max_iter   - maximum iterations (default: 1000)
%   opts.tol        - relative tolerance for convergence (default: 1e-6)
%   opts.verbose    - enable per-iteration logging (default: false)
%
%   The defaults follow the tuned hyperparameters recommended in the paper
%   and accompanying implementation notes.

    if nargin < 3 || isempty(opts)
        opts = struct();
    end
    if ~isfield(opts, 'lambda_reg') || isempty(opts.lambda_reg)
        opts.lambda_reg = 1e-2;
    end
    if ~isfield(opts, 'tau') || isempty(opts.tau)
        opts.tau = 0.1;
    end
    if ~isfield(opts, 'step_size') || isempty(opts.step_size)
        opts.step_size = 5e-3;
    end
    if ~isfield(opts, 'max_iter') || isempty(opts.max_iter)
        opts.max_iter = 1000;
    end
    if ~isfield(opts, 'tol') || isempty(opts.tol)
        opts.tol = 1e-6;
    end
    if ~isfield(opts, 'verbose') || isempty(opts.verbose)
        opts.verbose = false;
    end

    [m, n] = size(M);
    r = min(r, min(m, n));
    normM = norm(M, 'fro');
    if normM == 0
        L = zeros(size(M));
        S = zeros(size(M));
        return;
    end

    % Initialization: remove outliers and take rank-r SVD of the cleaned matrix.
    S = soft_threshold(M, opts.tau);
    [U, Sigma, V] = truncated_svd(M - S, r);
    singular_vals = diag(Sigma);
    sqrt_s = sqrt(max(singular_vals, 0));
    X = U * diag(sqrt_s);
    Y = V * diag(sqrt_s);

    for iter = 1:opts.max_iter
        L_prev = X * Y';
        S_prev = S;

        % Update sparse component first (proximal step).
        S = soft_threshold(M - L_prev, opts.tau);

        % Gradient step on low-rank factors.
        R = (L_prev + S) - M;
        gradX = R * Y + opts.lambda_reg * X;
        gradY = R' * X + opts.lambda_reg * Y;
        X = X - opts.step_size * gradX;
        Y = Y - opts.step_size * gradY;

        % Convergence check based on relative change.
        rel_change = (norm(X * Y' - L_prev, 'fro') + norm(S - S_prev, 'fro')) / normM;
        if opts.verbose
            fprintf('Iter %d: rel change = %.3e\n', iter, rel_change);
        end
        if rel_change < opts.tol
            break;
        end
    end

    L = X * Y';
end

function S = soft_threshold(A, tau)
    S = sign(A) .* max(abs(A) - tau, 0);
end

function [U, Sigma, V] = truncated_svd(A, r)
    try
        [U, Sigma, V] = svds(A, r);
    catch
        [U, Sigma, V] = svd(A, 'econ');
        r = min(r, size(Sigma, 1));
        U = U(:, 1:r);
        Sigma = Sigma(1:r, 1:r);
        V = V(:, 1:r);
    end
end
