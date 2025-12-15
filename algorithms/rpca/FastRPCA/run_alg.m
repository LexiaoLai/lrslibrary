% Fast RPCA via nonconvex factorization (Shen and Sanghavi, 2016)
% Suggested parameters follow the paper: alpha = 2*rhoS, alpha0 = 0.05,
% eta = 1.25, rank = params.rank.

if ~exist('params','var') || ~isfield(params,'rank') || isempty(params.rank)
    params.rank = min(size(M));
end
if ~isfield(params,'sparsity') || isempty(params.sparsity)
    params.sparsity = 0.1;
end

opts = struct();
opts.alpha = min(2 * params.sparsity, 0.5);
opts.alpha0 = 0.05;
opts.eta = 1.25;
opts.max_iter = 200;
opts.tol = 1e-6;
opts.verbose = false;

[L, S] = fast_rpca_factorized(M, params.rank, params.sparsity, opts);
