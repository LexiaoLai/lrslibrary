% Nonconvex RPCA via alternating minimization (Zhang et al., 2021)
% Hyperparameters follow the recommended values from the paper and
% accompanying implementation notes.

if ~exist('params','var') || ~isfield(params,'rank') || isempty(params.rank)
    params.rank = min(size(M));
end

opts = struct();
opts.lambda_reg = 1e-2;
opts.tau = 0.1;
opts.step_size = 5e-3;
opts.max_iter = 1000;
opts.tol = 1e-6;
opts.verbose = false;

[L, S] = bridging_rpca(M, params.rank, opts);
