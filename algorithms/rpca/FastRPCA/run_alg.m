% Fast RPCA via gradient descent (Cherapanamjeri et al., 2016)
% Suggested parameters follow the paper: alpha = min(2*rhoS, 0.49),
% step_size = 0.8, rank = params.rank.

if ~exist('params','var') || ~isfield(params,'rank') || isempty(params.rank)
    params.rank = min(size(M));
end
if ~isfield(params,'sparsity') || isempty(params.sparsity)
    params.sparsity = 0.1;
end

opts = struct();
opts.alpha = min(2 * params.sparsity, 0.49);
opts.step_size = 0.8;
opts.lambda = 0;
opts.max_iter = 500;
opts.tol = 1e-6;
opts.verbose = false;

[L, S] = fast_rpca_factorized(M, params.rank, params.sparsity, opts);
