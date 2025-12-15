clear
clc
close all

% set(groot, 'defaultFigureColor', 'w');
% set(groot, 'defaultAxesColor',   'w');
% set(groot, 'defaultAxesXColor',  'k');
% set(groot, 'defaultAxesYColor',  'k');
% set(groot, 'defaultTextColor',   'k');
% Comparison of recovery performance between Subgradient (factorized) and
% IALM (convex) RPCA solvers on synthetic low-rank plus sparse matrices.

m = 100;
n = 80;
eps = 1e-3;            % success tolerance on the relative error of L
trials = 100;           % number of Monte-Carlo runs per grid point
R = 20;                % maximum tested rank
P = 20;                % number of sparsity levels (0 : 0.5)

% ---------- NEW: output settings ----------
outdir = fullfile(pwd, 'results_rpca');
if ~exist(outdir, 'dir'); mkdir(outdir); end

timestamp = datestr(now, 'yyyymmdd_HHMMSS');
matfile = fullfile(outdir, ['success_rates_' timestamp '.mat']);
pdf_sub = fullfile(outdir, ['success_subgradient_' timestamp '.pdf']);
pdf_ialm = fullfile(outdir, ['success_ialm_' timestamp '.pdf']);
% ----------------------------------------

% Counters for successful recovery
success_subgrad = zeros(R, P + 1);
success_ialm = zeros(R, P + 1);

% Add algorithm folders to the path
addpath(fullfile('algorithms', 'rpca', 'Subgradient'));
addpath(fullfile('algorithms', 'rpca', 'IALM'));
% addpath(fullfile('algorithms', 'rpca', 'EALM'));

for r = 1:R
    for p = 0:P
        sparsity = p / P * 0.5;
        for k = 1:trials
            % Low-rank component of rank r
            U = 2 * rand(m, r) - 1;
            V = 2 * rand(n, r) - 1;
            L_true = U * V';

            % Sparse corruption with given sparsity
            supp = rand(m, n) <= sparsity;
            S_true = supp .* (2 * rand(m, n) - 1);

            M = L_true + S_true;

            % Subgradient solver on factors
            opts_sub = struct('rank', r, 'max_iter', 1000);
            [L_sub, ~, ~] = rpca_subgradient(M, opts_sub);
            if norm(L_sub - L_true, 'fro') / norm(L_true, 'fro') <= eps
                success_subgrad(r, p + 1) = success_subgrad(r, p + 1) + 1;
            end

            % IALM solver (convex RPCA)
            [L_ialm, ~, ~, ~, ~] = inexact_alm_rpca(M);
            if norm(L_ialm - L_true, 'fro') / norm(L_true, 'fro') <= eps
                success_ialm(r, p + 1) = success_ialm(r, p + 1) + 1;
            end
        end
    end
end

% Convert counts to success percentages
success_subgrad = success_subgrad / trials * 100;
success_ialm = success_ialm / trials * 100;

% ---------- NEW: save results to MAT ----------
sparsity_grid = (0:P) / P * 0.5;
rank_grid = 1:R;
save(matfile, ...
    'success_subgrad', 'success_ialm', ...
    'sparsity_grid', 'rank_grid', ...
    'm', 'n', 'eps', 'trials', 'R', 'P');
% --------------------------------------------

% Visualization: success rate heatmaps

% Subgradient
fig1 = figure;
colormap('default');
imagesc(sparsity_grid, sort(1:R, 'descend'), success_subgrad(sort(1:R, 'descend'), :));
colorbar('XTickLabel', {'0','10','20','30','40','50','60','70','80','90','100'}, 'XTick', 0:10:100);
set(gca, 'YDir', 'normal', 'FontSize', 15, 'YTick', 1:R);
axis on;
xlabel('Sparsity', 'FontSize', 18, 'fontweight', 'bold');
xticks(0:.05:.5);
xticklabels({'0','0.05','0.1','0.15','0.2','0.25','0.3','0.35','0.4','0.45','0.5'});
ylabel('Rank', 'FontSize', 18, 'fontweight', 'bold');

% ---------- NEW: save figure to PDF ----------
set(fig1, 'PaperPositionMode', 'auto');
exportgraphics(fig1, pdf_sub, 'ContentType', 'vector')%, 'BackgroundColor','white');
% --------------------------------------------

% IALM
fig2 = figure;
colormap('default');
imagesc(sparsity_grid, sort(1:R, 'descend'), success_ialm(sort(1:R, 'descend'), :));
colorbar('XTickLabel', {'0','10','20','30','40','50','60','70','80','90','100'}, 'XTick', 0:10:100);
set(gca, 'YDir', 'normal', 'FontSize', 15, 'YTick', 1:R);
axis on;
xlabel('Sparsity', 'FontSize', 18, 'fontweight', 'bold');
xticks(0:.05:.5);
xticklabels({'0','0.05','0.1','0.15','0.2','0.25','0.3','0.35','0.4','0.45','0.5'});
ylabel('Rank', 'FontSize', 18, 'fontweight', 'bold');

% ---------- NEW: save figure to PDF ----------
set(fig2, 'PaperPositionMode', 'auto');
exportgraphics(fig2, pdf_ialm, 'ContentType', 'vector')%, 'BackgroundColor','white');
% --------------------------------------------
