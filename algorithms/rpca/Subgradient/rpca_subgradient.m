function [L, S, output] = rpca_subgradient(M, opts)
%RPCA_SUBGRADIENT Robust PCA via adaptive subgradient method on factors.
%   [L, S, output] = RPCA_SUBGRADIENT(M, opts) decomposes the input matrix M
%   into a low-rank component L and sparse component S by minimizing
%   sum(abs(X*Y' - M)) / (m*n) with respect to factors X in R^{m x k} and
%   Y in R^{n x k}. The algorithm follows an adaptive subgradient update
%   that shrinks the step size after consecutive non-improving iterations.
%
%   Performance-only tweaks (no algorithm change):
%     - Precompute mn = m*n and inv_mn once (avoid repeated division/multiplication).
%     - Precompute sqrt(mn) for init_step default.
%     - Keep the original objective and gradients exactly (sum(abs(R),'all'), sign, G*Y, G'*X).
%     - Minor scalar/branch simplifications.

if nargin < 2 || isempty(opts)
    opts = struct();
end

[m, n] = size(M);
mn = m * n;
inv_mn = 1 / mn;

% k = get_opt(opts, 'rank', min(10, min(m, n)));
k = get_opt(opts, 'rank', min(20, min(m, n)));
max_iter = get_opt(opts, 'max_iter', 500);
init_step = get_opt(opts, 'init_step', 1 / sqrt(mn));
shrink_factor = get_opt(opts, 'shrink', 0.5);
patience = get_opt(opts, 'patience', 10);
decrease_tol = get_opt(opts, 'decrease_tol', 1e-20);
stop_tol = get_opt(opts, 'stop_tol', 1e-20);
min_step = get_opt(opts, 'min_step', 1e-15);
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

t_final = max_iter;

for t = 1:max_iter
    % Residual (same as original)
    R = X * Y' - M;

    % Objective (same as original), but use precomputed inv_mn
    f_val = sum(abs(R), 'all') * inv_mn;

    hist_obj(t) = f_val;
    step_history(t) = eta;

    % Early stop (same as original)
    if t > 1 && abs(f_val - prev_f) < stop_tol
        t_final = t;
        break;
    end

    % Adaptive step control (same as original)
    if f_val < best_f - decrease_tol
        best_f = f_val;
        no_improve_count = 0;
    else
        no_improve_count = no_improve_count + 1;
        if no_improve_count >= patience
            eta = eta * shrink_factor;
            if eta < min_step
                eta = min_step;
            end
            no_improve_count = 0;
        end
    end

    % Subgradient and gradients (same as original)
    G = sign(R);
    grad_X = G * Y;
    grad_Y = G' * X;

    % Updates (same as original)
    X = X - eta * grad_X;
    Y = Y - eta * grad_Y;

    prev_f = f_val;
end

% Trim history to actual length
hist_obj = hist_obj(1:t_final);
step_history = step_history(1:t_final);

% Final decomposition (same as original)
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
