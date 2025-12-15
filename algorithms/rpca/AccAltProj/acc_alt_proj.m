function [L, S] = acc_alt_proj(M, r, opts)
%ACC_ALT_PROJ Simplified implementation of Accelerated Alternating Projections for RPCA.
%   [L, S] = ACC_ALT_PROJ(M, r, opts) decomposes M into a rank-r matrix L
%   and a sparse matrix S. The implementation follows the alternating
%   projection intuition in Cai et al. (2019) and uses magnitude-based
%   thresholding to promote sparsity.

    if nargin < 3 || isempty(opts)
        opts = struct();
    end
    if ~isfield(opts, 'max_iter') || isempty(opts.max_iter)
        opts.max_iter = 200;
    end
    if ~isfield(opts, 'tol') || isempty(opts.tol)
        opts.tol = 1e-6;
    end
    if ~isfield(opts, 'sparsity') || isempty(opts.sparsity)
        opts.sparsity = 0.1;
    end
    if ~isfield(opts, 'verbose') || isempty(opts.verbose)
        opts.verbose = false;
    end

    normM = norm(M, 'fro');
    if normM == 0
        L = zeros(size(M));
        S = zeros(size(M));
        return;
    end

    S = magnitude_threshold(M, opts.sparsity);
    L = zeros(size(M));

    for iter = 1:opts.max_iter
        L_prev = L;
        S_prev = S;

        L = truncated_svd(M - S, r);
        R = M - L;
        S = magnitude_threshold(R, opts.sparsity);

        rel_change = (norm(L - L_prev, 'fro') + norm(S - S_prev, 'fro')) / normM;
        if opts.verbose
            fprintf('Iter %d: rel change = %.3e\n', iter, rel_change);
        end
        if rel_change < opts.tol
            break;
        end
    end
end

function L = truncated_svd(A, r)
    try
        [U, Sigma, V] = svds(A, r);
    catch
        [U, Sigma, V] = svd(A, 'econ');
        r = min(r, size(Sigma, 1));
        U = U(:, 1:r);
        Sigma = Sigma(1:r, 1:r);
        V = V(:, 1:r);
    end
    L = U * Sigma * V';
end

function S = magnitude_threshold(X, fraction)
    if fraction <= 0
        S = zeros(size(X));
        return;
    end
    if fraction >= 1
        S = X;
        return;
    end

    absX = abs(X(:));
    k = max(1, ceil(fraction * numel(absX)));
    sorted_vals = sort(absX, 'ascend');
    threshold = sorted_vals(max(numel(sorted_vals) - k + 1, 1));
    mask = abs(X) >= threshold;
    S = X .* mask;
end
