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
%     - shrink:       shrink factor for step size (default: 0.5)
%     - patience:     number of non-decreasing iterations before shrinking
%                     the step size (default: 10)
%     - decrease_tol: minimum decrease to be counted as improvement
%                     (default: 1e-10)
%     - stop_tol:     stop when consecutive objective change is below this
%                     threshold (default: 1e-8)
%     - min_step:     lower bound on the step size (default: 1e-10)
%     - init_mode:    'random' (default) or 'svd' for initialization
%     - init_scale:   scale for random initialization (default: 1e-6)
%     - X0, Y0:       custom initial factors (override init_mode)
%     - random_seed:  seed for reproducibility
%
%   output is a struct containing:
%     - X, Y:         final factors
%     - hist_obj:     objective values per iteration
%     - step_history: step size used at each iteration
%     - final_step:   final step size
%
%   Example:
%     [L, S] = rpca_sgd(M, struct('rank', 5, 'batch_size', 5000));
%
%   This implementation mirrors the adaptive subgradient factorization while
%   relying on stochastic batches for scalability.

if nargin < 2 || isempty(opts)
    opts = struct();
end

[m, n] = size(M);
num_entries = numel(M);

k = get_opt(opts, 'rank', min(10, min(m, n)));
max_iter = get_opt(opts, 'max_iter', 1000);
batch_size = get_opt(opts, 'batch_size', min(num_entries, ceil(0.01 * num_entries)));
init_step = get_opt(opts, 'init_step', 1 / sqrt(m * n));
shrink_factor = get_opt(opts, 'shrink', 0.5);
patience = get_opt(opts, 'patience', 10);
decrease_tol = get_opt(opts, 'decrease_tol', 1e-10);
stop_tol = get_opt(opts, 'stop_tol', 1e-8);
min_step = get_opt(opts, 'min_step', 1e-10);
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

eta = init_step;
hist_obj = zeros(max_iter, 1);
step_history = zeros(max_iter, 1);
best_f = inf;
no_improve_count = 0;
prev_f = inf;

for t = 1:max_iter
    R_full = X * Y' - M;
    f_val = sum(abs(R_full), 'all') / (m * n);
    hist_obj(t) = f_val;
    step_history(t) = eta;

    if t > 1 && abs(f_val - prev_f) < stop_tol
        hist_obj = hist_obj(1:t);
        step_history = step_history(1:t);
        break;
    end

    if f_val < best_f - decrease_tol
        best_f = f_val;
        no_improve_count = 0;
    else
        no_improve_count = no_improve_count + 1;
        if no_improve_count >= patience
            eta = max(eta * shrink_factor, min_step);
            no_improve_count = 0;
        end
    end

    idx = randi(num_entries, batch_size, 1);
    [rows, cols] = ind2sub([m, n], idx);
    residual_batch = R_full(idx);
    G = sign(residual_batch);

    grad_X = zeros(m, k);
    grad_Y = zeros(n, k);
    for j = 1:k
        grad_X(:, j) = accumarray(rows, G .* Y(cols, j), [m, 1], @sum, 0);
        grad_Y(:, j) = accumarray(cols, G .* X(rows, j), [n, 1], @sum, 0);
    end

    scale = eta / batch_size;
    X = X - scale * grad_X;
    Y = Y - scale * grad_Y;
    prev_f = f_val;
end

L = X * Y';
S = M - L;

output.X = X;
output.Y = Y;
output.hist_obj = hist_obj;
output.step_history = step_history;
output.final_step = eta;
end

function val = get_opt(opts, name, default)
%GET_OPT Fetch option with default.
    if isfield(opts, name) && ~isempty(opts.(name))
        val = opts.(name);
    else
        val = default;
    end
end
