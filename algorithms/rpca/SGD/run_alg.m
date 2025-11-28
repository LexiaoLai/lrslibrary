% Stochastic subgradient method for RPCA factorization
% process_video('RPCA', 'SGD', 'dataset/demo.avi', 'output/demo_SGD.avi');

% Users can customize options through the "params" struct passed to
% run_algorithm, e.g. params.rank, params.batch_size or params.max_iter.
opts = struct();
if exist('params', 'var') && isstruct(params)
    opts_fields = {'rank','max_iter','batch_size','init_step', ...
                  'shrink','patience','decrease_tol','stop_tol','min_step', ...
                  'init_mode','init_scale','X0','Y0','random_seed'};
    for i = 1:numel(opts_fields)
        name = opts_fields{i};
        if isfield(params, name)
            opts.(name) = params.(name);
        end
    end
end

[L, S, output] = rpca_sgd(M, opts);

% Optional: expose convergence history in workspace if needed
hist_obj = output.hist_obj; %#ok<NASGU>
