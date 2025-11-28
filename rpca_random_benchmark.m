%% RPCA Random Benchmark
% This script generates synthetic RPCA instances and benchmarks multiple
% algorithms from LRSLibrary. It reports average CPU time and relative
% reconstruction error across several random trials.

clear; clc;

% ------------------------------------------------------------
% User-configurable parameters
% ------------------------------------------------------------
num_trials = 3;           % Number of random instances to average over
m = 100;                  % Number of rows
n = 100;                  % Number of columns
r_true = 5;               % True rank of the low-rank component
sparsity = 0.1;           % Fraction of corrupted entries in the sparse part
sparse_scale = 10;        % Magnitude of sparse corruption
algorithms = { ...
    'Subgradient', ...    % Adaptive subgradient method on factors
    'FPCP', ...           % Fast Principal Component Pursuit
    'GoDec' ...           % Go Decomposition
};

% Optional: set to a fixed value for reproducibility
base_seed = 1;

% ------------------------------------------------------------
% Setup LRSLibrary
% ------------------------------------------------------------
lrs_setup;
lrs_load_conf;

fprintf('Running RPCA benchmark with %d trials on %dx%d matrices (rank=%d, sparsity=%.2f)\n', ...
    num_trials, m, n, r_true, sparsity);

results = struct();
for a = 1:numel(algorithms)
    alg = algorithms{a};
    results.(alg).time = zeros(num_trials, 1);
    results.(alg).rel_err_L = zeros(num_trials, 1);
    results.(alg).rel_err_S = zeros(num_trials, 1);
    results.(alg).success = false(num_trials, 1);
end

for t = 1:num_trials
    rng(base_seed + t - 1);
    [M, L_true, S_true] = generate_rpca_instance(m, n, r_true, sparsity, sparse_scale);

    fprintf('\nTrial %d/%d\n', t, num_trials);
    for a = 1:numel(algorithms)
        alg = algorithms{a};
        fprintf('  - %s...', alg);
        params = struct('rank', r_true);
        try
            out = run_algorithm('RPCA', alg, M, params);
            results.(alg).time(t) = out.cputime;

            L_norm = norm(L_true, 'fro');
            if isfield(out, 'L') && ~isempty(out.L) && L_norm > 0
                results.(alg).rel_err_L(t) = norm(out.L - L_true, 'fro') / L_norm;
            else
                results.(alg).rel_err_L(t) = NaN;
            end

            S_norm = norm(S_true, 'fro');
            if isfield(out, 'S') && ~isempty(out.S) && S_norm > 0
                results.(alg).rel_err_S(t) = norm(out.S - S_true, 'fro') / S_norm;
            else
                results.(alg).rel_err_S(t) = NaN;
            end

            results.(alg).success(t) = true;
            fprintf(' done (time = %.3fs)\n', out.cputime);
        catch ME
            fprintf('\n    Warning: %s failed with message: %s\n', alg, ME.message);
            results.(alg).time(t) = NaN;
            results.(alg).rel_err_L(t) = NaN;
            results.(alg).rel_err_S(t) = NaN;
            results.(alg).success(t) = false;
        end
    end
end

% ------------------------------------------------------------
% Summary statistics
% ------------------------------------------------------------
fprintf('\n=== RPCA Benchmark Summary ===\n');
fprintf('%-15s  %-12s  %-15s  %-15s  %s\n', 'Algorithm', 'Avg Time (s)', ...
    'Avg RelErr(L)', 'Avg RelErr(S)', 'Success Rate');
fprintf('%s\n', repmat('-', 1, 70));
for a = 1:numel(algorithms)
    alg = algorithms{a};
    time_vals = results.(alg).time;
    relL_vals = results.(alg).rel_err_L;
    relS_vals = results.(alg).rel_err_S;
    success_vals = results.(alg).success;

    avg_time = mean_omit_nan(time_vals);
    avg_relL = mean_omit_nan(relL_vals);
    avg_relS = mean_omit_nan(relS_vals);
    success_rate = 100 * mean(success_vals);

    fprintf('%-15s  %12.3f  %15.3e  %15.3e  %6.1f%%%%\n', ...
        alg, avg_time, avg_relL, avg_relS, success_rate);
end
fprintf('%s\n', repmat('-', 1, 70));
fprintf('Each entry averages over %d random instances.\n', num_trials);

% ------------------------------------------------------------
% Helper functions
% ------------------------------------------------------------
function [M, L_true, S_true] = generate_rpca_instance(m, n, r, sparsity, scale)
%GENERATE_RPCA_INSTANCE Create low-rank plus sparse matrix.
%   [M, L_true, S_true] = GENERATE_RPCA_INSTANCE(m, n, r, sparsity, scale)
%   builds L_true = U*V' with rank r and sparse outliers S_true with the
%   given density and magnitude. The observed matrix is M = L_true + S_true.

    U = randn(m, r);
    V = randn(n, r);
    L_true = U * V';

    S_true = zeros(m, n);
    mask = rand(m, n) < sparsity;
    S_true(mask) = scale * (2 * rand(sum(mask(:)), 1) - 1);

    M = L_true + S_true;
end

function mval = mean_omit_nan(v)
%MEAN_OMIT_NAN Compute mean ignoring NaN entries.
    v = v(~isnan(v));
    if isempty(v)
        mval = NaN;
    else
        mval = mean(v);
    end
end
