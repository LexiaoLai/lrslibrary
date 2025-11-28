function [L, S, output] = rpca_sgd(M, opts)
%RPCA_SGD Robust PCA via stochastic subgradient descent on factors.
%   [L, S, output] = RPCA_SGD(M, opts) decomposes the input matrix M into
%   a low-rank component L and sparse component S by minimizing
%   sum(abs(X*Y' - M)) / (m*n) with respect to factors X in R^{m x k} and
%   Y in R^{n x k}. The routine uses mini-batch stochastic subgradients
%   over random entries of M.
%
%   opts is an optional struct with fields:
%     - rank:         target factor rank k (default: min(10, min(m,n)))
%     - max_iter:     maximum iterations (default: 1000)
%     - batch_size:   number of sampled entries per iteration (default:
%                     min(numel(M), ceil(0.01 * numel(M))))
%     - init_step:    initial step size (default: 1 / sqrt(m*n))
%     - step_schedule:'sqrt' (default) for eta = init_step/sqrt(t) or
%                     'constant' to keep eta fixed
%     - eval_freq:    compute full objective every eval_freq iterations
%                     (default: 10)
%     - init_mode:    'random' (default) or 'svd' for initialization
%     - init_scale:   scale for random initialization (default: 1e-6)
%     - X0, Y0:       custom initial factors (override init_mode)
%     - random_seed:  seed for reproducibility
%
%   output is a struct containing:
%     - X, Y:         final factors
%     - hist_obj:     objective values at evaluation iterations
%     - eval_iters:   iteration numbers corresponding to hist_obj
%     - final_step:   final step size
%
%   Example:
%     [L, S] = rpca_sgd(M, struct('rank', 5, 'batch_size', 5000));
%
%   This implementation mirrors the adaptive subgradient factorization but
%   relies on stochastic batches for scalability.

if nargin < 2 || isempty(opts)
    opts = struct();
end

[m, n] = size(M);
num_entries = numel(M);

k = get_opt(opts, 'rank', min(10, min(m, n)));
max_iter = get_opt(opts, 'max_iter', 1000);
batch_size = get_opt(opts, 'batch_size', min(num_entries, ceil(0.01 * num_entries)));
init_step = get_opt(opts, 'init_step', 1 / sqrt(m * n));
step_schedule = get_opt(opts, 'step_schedule', 'sqrt');
eval_freq = get_opt(opts, 'eval_freq', 10);
init_mode = get_opt(opts, 'init_mode', 'random');
init_scale = get_opt(opts, 'init_scale', 1e-6);

if isfield(opts, 'random_seed')
    rng(opts.random_seed);
end

% Initialization
if isfield(opts, 'X0') && isfield(opts, 'Y0')
    X = opts.X0;
    Y = opts.Y0;
else
    switch lower(init_mode)
        case 'svd'
            r = min(k, min(m, n));
            [U, Svals, V] = svds(M, r);
            sqrtS = sqrt(Svals);
            X = U * sqrtS;
            Y = V * sqrtS;
            if r < k
                X = [X, zeros(m, k - r)];
                Y = [Y, zeros(n, k - r)];
            end
        otherwise % 'random'
            X = (2 * rand(m, k) - 1) * init_scale;
            Y = (2 * rand(n, k) - 1) * init_scale;
    end
end

hist_obj = nan(max_iter, 1);
eval_iters = zeros(max_iter, 1);
eta = init_step;

for t = 1:max_iter
    % Sample mini-batch of entries
    idx = randi(num_entries, batch_size, 1);
    [rows, cols] = ind2sub([m, n], idx);

    preds = sum(X(rows, :) .* Y(cols, :), 2);
    residual = preds - M(idx);
    G = sign(residual);

    grad_X = zeros(m, k);
    grad_Y = zeros(n, k);
    for j = 1:k
        grad_X(:, j) = accumarray(rows, G .* Y(cols, j), [m, 1], @sum, 0);
        grad_Y(:, j) = accumarray(cols, G .* X(rows, j), [n, 1], @sum, 0);
    end

    step = get_step(step_schedule, eta, t);
    scale = step / batch_size;
    X = X - scale * grad_X;
    Y = Y - scale * grad_Y;

    if mod(t, eval_freq) == 0 || t == 1 || t == max_iter
        R_full = X * Y' - M;
        f_val = sum(abs(R_full), 'all') / (m * n);
        hist_obj(t) = f_val;
        eval_iters(t) = t;
    end
end

L = X * Y';
S = M - L;

hist_obj = hist_obj(~isnan(hist_obj));
eval_iters = eval_iters(eval_iters ~= 0);

output.X = X;
output.Y = Y;
output.hist_obj = hist_obj;
output.eval_iters = eval_iters;
output.final_step = get_step(step_schedule, eta, max_iter);
end

function val = get_opt(opts, name, default)
%GET_OPT Fetch option with default.
    if isfield(opts, name) && ~isempty(opts.(name))
        val = opts.(name);
    else
        val = default;
    end
end

function step = get_step(schedule, init_step, iter)
%GET_STEP Step-size rule dispatcher.
    switch lower(schedule)
        case 'constant'
            step = init_step;
        otherwise % 'sqrt'
            step = init_step / sqrt(iter);
    end
end
