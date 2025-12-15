function [L, S] = fast_rpca_factorized(M, r, rhoS, opts)
%FAST_RPCA_FACTORIZED Implements the Fast RPCA algorithm from Cherapanamjeri et al. (2016).
%   [L, S] = FAST_RPCA_FACTORIZED(M, r, rhoS, opts) splits M into low-rank
%   and sparse parts using the gradient-descent algorithm described in
%   https://arxiv.org/pdf/1605.07784 with the paper's recommended parameters.
%
%   opts.alpha      - fraction of entries to keep when thresholding (default: min(2*rhoS, 0.49))
%   opts.step_size  - gradient step size (default: 0.8 as suggested in the paper)
%   opts.lambda     - Frobenius regularization on factors (default: 0)
%   opts.max_iter   - maximum iterations (default: 500)
%   opts.tol        - relative tolerance for convergence (default: 1e-6)
%   opts.verbose    - display per-iteration progress (default: false)

    if nargin < 4 || isempty(opts)
        opts = struct();
    end
    if ~isfield(opts, 'alpha') || isempty(opts.alpha)
        opts.alpha = min(2 * rhoS, 0.49);
    end
    if ~isfield(opts, 'step_size') || isempty(opts.step_size)
        opts.step_size = 0.8;
    end
    if ~isfield(opts, 'lambda') || isempty(opts.lambda)
        opts.lambda = 0;
    end
    if ~isfield(opts, 'max_iter') || isempty(opts.max_iter)
        opts.max_iter = 500;
    end
    if ~isfield(opts, 'tol') || isempty(opts.tol)
        opts.tol = 1e-6;
    end
    if ~isfield(opts, 'verbose') || isempty(opts.verbose)
        opts.verbose = false;
    end

    [m, n] = size(M);
    normM = norm(M, 'fro');
    if normM == 0
        L = zeros(size(M));
        S = zeros(size(M));
        return;
    end

    % Initialization: hard threshold and truncated SVD.
    S = hard_threshold_fraction(M, opts.alpha);
    [U0, Sigma0, V0] = truncated_svd(M - S, r);
    U = U0 * sqrt(Sigma0);
    V = V0 * sqrt(Sigma0);

    for iter = 1:opts.max_iter
        L_prev = U * V';
        S_prev = S;

        % Sparse update via hard thresholding.
        S = hard_threshold_fraction(M - L_prev, opts.alpha);

        % Gradient step on factors.
        resid = (L_prev + S) - M;
        gradU = resid * V + opts.lambda * U;
        gradV = resid' * U + opts.lambda * V;

        % Use per-iteration Lipschitz estimates to keep the step sizes
        % stable; otherwise the factor norms can explode and produce NaNs.
        normV2 = max(eps, norm(V, 2)^2) + opts.lambda;
        normU2 = max(eps, norm(U, 2)^2) + opts.lambda;
        stepU = opts.step_size / normV2;
        stepV = opts.step_size / normU2;

        U = U - stepU * gradU;
        V = V - stepV * gradV;

        % Convergence check.
        rel_change = (norm(U * V' - L_prev, 'fro') + norm(S - S_prev, 'fro')) / normM;
        if opts.verbose
            fprintf('Iter %d: rel change = %.3e\n', iter, rel_change);
        end
        if rel_change < opts.tol
            break;
        end
    end

    L = U * V';
end

function S = hard_threshold_fraction(A, alpha)
%HARD_THRESHOLD_FRACTION Keeps the largest |A| entries by fraction alpha.
    alpha = max(0, min(alpha, 1));
    if alpha == 0
        S = zeros(size(A));
        return;
    end

    absA = abs(A(:));
    k = max(1, ceil(alpha * numel(absA)));
    sorted_vals = sort(absA, 'descend');
    tau = sorted_vals(k);
    mask = abs(A) >= tau;
    S = A .* mask;
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
