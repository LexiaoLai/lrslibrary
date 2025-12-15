function [L, S] = fast_rpca_factorized(M, r, rhoS, opts)
%FAST_RPCA_FACTORIZED Implements the Fast RPCA algorithm from Shen and Sanghavi (2016).
%   [L, S] = FAST_RPCA_FACTORIZED(M, r, rhoS, opts) returns low-rank and
%   sparse components of M. The algorithm alternates between updating the
%   sparse support via the operator T_alpha and refining low-rank factors.
%
%   opts.alpha   - corruption estimator (default: min(2*rhoS, 0.5))
%   opts.alpha0  - slack to avoid trimming true inliers (default: 0.05)
%   opts.eta     - step size for the V update (default: 1.25)
%   opts.max_iter- maximum iterations (default: 200)
%   opts.tol     - relative tolerance for convergence (default: 1e-6)
%   opts.verbose - display per-iteration progress

    if nargin < 4 || isempty(opts)
        opts = struct();
    end
    if ~isfield(opts, 'alpha') || isempty(opts.alpha)
        opts.alpha = min(2 * rhoS, 0.5);
    end
    if ~isfield(opts, 'alpha0') || isempty(opts.alpha0)
        opts.alpha0 = 0.05;
    end
    if ~isfield(opts, 'eta') || isempty(opts.eta)
        opts.eta = 1.25;
    end
    if ~isfield(opts, 'max_iter') || isempty(opts.max_iter)
        opts.max_iter = 200;
    end
    if ~isfield(opts, 'tol') || isempty(opts.tol)
        opts.tol = 1e-6;
    end
    if ~isfield(opts, 'verbose') || isempty(opts.verbose)
        opts.verbose = false;
    end

    [m, n] = size(M);
    [U0, Sigma0, V0] = truncated_svd(M, r);
    U_t = U0 * sqrt(Sigma0);
    V_t = V0 * sqrt(Sigma0);
    S_t = zeros(m, n);

    normM = norm(M, 'fro');
    if normM == 0
        L = zeros(size(M));
        S = zeros(size(M));
        return;
    end

    for iter = 1:opts.max_iter
        prev_L = U_t * V_t';
        prev_S = S_t;

        S_t = T_alpha(M - U_t * V_t', opts.alpha + opts.alpha0);
        A = truncated_svd(M - S_t, r);
        V_gram = V_t' * V_t;
        U_t = A * (V_t / (V_gram + 1e-8 * eye(size(V_gram))));

        trimmed_M = T_alpha(M, opts.alpha + opts.alpha0);
        V_t = opts.eta * V_t - opts.eta * (U_t' * (U_t * V_t - trimmed_M));

        rel_change = (norm(U_t * V_t' - prev_L, 'fro') + norm(S_t - prev_S, 'fro')) / normM;
        if opts.verbose
            fprintf('Iter %d: rel change = %.3e\n', iter, rel_change);
        end
        if rel_change < opts.tol
            break;
        end
    end

    L = U_t * V_t';
    S = S_t;
end

function S = T_alpha(A, alpha)
%T_ALPHA Keep the largest-magnitude entries row- and column-wise.
    alpha = max(0, min(alpha, 1));
    if alpha == 0
        S = zeros(size(A));
        return;
    end
    [m, n] = size(A);
    absA = abs(A);

    row_thresh = zeros(m, 1);
    for i = 1:m
        row_thresh(i) = frac_threshold(absA(i, :), alpha);
    end
    col_thresh = zeros(1, n);
    for j = 1:n
        col_thresh(j) = frac_threshold(absA(:, j), alpha);
    end

    row_mask = bsxfun(@ge, absA, row_thresh);
    col_mask = bsxfun(@ge, absA, col_thresh);
    mask = row_mask | col_mask;
    S = A .* mask;
end

function tau = frac_threshold(vec, alpha)
    vec = abs(vec(:));
    if isempty(vec)
        tau = 0;
        return;
    end
    k = max(1, ceil(alpha * numel(vec)));
    sorted_vals = sort(vec, 'ascend');
    tau = sorted_vals(max(numel(sorted_vals) - k + 1, 1));
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

    if nargout == 1
        U = U * Sigma * V';
    end
end
