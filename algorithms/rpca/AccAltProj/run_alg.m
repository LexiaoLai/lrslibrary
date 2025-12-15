% Accelerated Alternating Projections for RPCA
% Reference: Cai et al. "Accelerated Alternating Projections for Robust PCA".

if ~exist('params','var') || ~isfield(params,'rank') || isempty(params.rank)
    params.rank = min(size(M));
end

if ~isfield(params,'sparsity') || isempty(params.sparsity)
    params.sparsity = 0.1;
end

opts = struct();
opts.max_iter = 2000;
opts.tol = 1e-10;
opts.sparsity = params.sparsity;
opts.verbose = false;

[L, S] = acc_alt_proj(M, params.rank, opts);
