function analyze_difficulty(expType)
    % ANALYZE_DIFFICULTY  Sweep one experiment type and plot D_cat vs. parameter.
    %
    %   analyze_difficulty('Size')
    %   analyze_difficulty('Position')  % 'center' | 'mid' | 'peripheral'
    %   analyze_difficulty('Count')
    %   analyze_difficulty('AnatomyVar')
    %   analyze_difficulty('Sharpness')

    close all;
    clc;

    genRoot = fileparts(mfilename('fullpath'));
    addpath(fullfile(genRoot, 'eikonal'));
    initEikonalPaths();

    cfg_base.imSize = 128;
    cfg_base.num_sensors = 64;
    cfg_base.num_receivers = 64;
    cfg_base.sos_background = 1.480;
    cfg_base.sos_healthy = 1.540;
    cfg_base.sos_tumor_mean = 1.580;
    cfg_base.sos_tumor_spread = 0.02;
    cfg_base.noise_std = 0.001;
    cfg_base.verbose_tof = false;

    rng(42, 'twister');

    [cfg_tests, x_numeric, x_labels] = configTestGenerator(expType);
    num_samples = 10;

    tof_cfg = make_tof_cfg_an(cfg_base);
    d_scores = zeros(length(cfg_tests), 1);

    et = lower(strtrim(expType));

    for v = 1:length(cfg_tests)
        cfg_t = merge_cfg(cfg_base, cfg_tests(v));
        tofs = cell(num_samples, 1);
        signals = zeros(num_samples, 1);

        for i = 1:num_samples
            [Vh, Vt, ~] = runAnatomy(et, cfg_t);
            tof_eng = ToF();
            tof_eng.setProperties(tof_cfg);
            th = tof_eng.createTravelTime(Vh);
            tt = tof_eng.createTravelTime(Vt);
            signals(i) = norm(tt(:) - th(:), 2);
            tofs{i} = tt;
        end

        intra_diffs = [];
        for i = 1:num_samples
            for j = (i + 1):num_samples
                intra_diffs(end + 1) = norm(tofs{i}(:) - tofs{j}(:), 2); %#ok<AGROW>
            end
        end

        if isempty(intra_diffs) || mean(intra_diffs) == 0
            d_scores(v) = NaN;
        else
            d_scores(v) = mean(signals) / mean(intra_diffs);
        end

        fprintf('Parameter %s | D_cat = %.4f\n', x_labels{v}, d_scores(v));
    end

    figure;
    if isnumeric(x_numeric) && numel(x_numeric) == numel(d_scores) && all(isfinite(x_numeric(:)))
        plot(x_numeric(:), d_scores, '-o', 'LineWidth', 1.5);
        set(gca, 'XTick', x_numeric(:));
    else
        plot(1:length(d_scores), d_scores, '-o', 'LineWidth', 1.5);
        set(gca, 'XTick', 1:length(x_labels));
        set(gca, 'XTickLabel', x_labels);
    end
    xlabel(sprintf('Parameter: %s', expType));
    ylabel('D_{cat}');
    title('Learning Capability vs. Parameter Difficulty');
    grid on;
end

function [tests, x_num, x_lab] = configTestGenerator(expType)
    et = lower(strtrim(expType));
    switch et
        case 'size'
            vals = [2.5, 5, 7.5, 10.5];
            for k = 1:length(vals)
                tests(k).value = vals(k);
            end
            x_num = vals(:);
            x_lab = arrayfun(@(v) sprintf('%.1f', v), vals, 'UniformOutput', false);
        case 'position'
            labs = {'center', 'mid', 'peripheral'};
            for k = 1:length(labs)
                tests(k).type = labs{k};
            end
            x_num = NaN;
            x_lab = labs(:);
        case 'count'
            vals = [1, 2, 3, 5];
            for k = 1:length(vals)
                tests(k).value = vals(k);
            end
            x_num = vals(:);
            x_lab = arrayfun(@int2str, vals, 'UniformOutput', false);
        case 'anatomyvar'
            vals = [20, 25, 30];
            for k = 1:length(vals)
                tests(k).value = vals(k);
            end
            x_num = vals(:);
            x_lab = arrayfun(@(v) sprintf('%g', v), vals, 'UniformOutput', false);
        case 'sharpness'
            vals = [0, 1.5, 3.0];
            for k = 1:length(vals)
                tests(k).value = vals(k);
            end
            x_num = vals(:);
            x_lab = arrayfun(@(v) sprintf('%.1f', v), vals, 'UniformOutput', false);
        otherwise
            error('configTestGenerator: unsupported expType ''%s''', expType);
    end
end

function c = merge_cfg(base, t)
    c = base;
    fn = fieldnames(t);
    for i = 1:length(fn)
        c.(fn{i}) = t.(fn{i});
    end
end

function tf = make_tof_cfg_an(cfg)
    tf.imSize = cfg.imSize;
    tf.num_sensors = cfg.num_sensors;
    if isfield(cfg, 'num_receivers') && ~isempty(cfg.num_receivers)
        tf.num_receivers = cfg.num_receivers;
    else
        tf.num_receivers = cfg.num_sensors;
    end
    tf.verbose = false;
    if isfield(cfg, 'verbose_tof'), tf.verbose = cfg.verbose_tof; end
    if isfield(cfg, 'sos_background'), tf.sos_background = cfg.sos_background; end
    if isfield(cfg, 'sos_healthy'),   tf.sos_healthy = cfg.sos_healthy; end
    if isfield(cfg, 'sos_fat'),       tf.sos_fat = cfg.sos_fat; end
    if isfield(cfg, 'sos_tumor_mean'), tf.sos_tumor_mean = cfg.sos_tumor_mean; end
end
