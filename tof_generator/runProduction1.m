function runProduction1()
    % RUNPRODUCTION1  Generate paired healthy/tumour phantoms, ToF, previews, and .mat (v7.3).

    close all;
    clc;

    genRoot = fileparts(mfilename('fullpath'));
    addpath(fullfile(genRoot, 'eikonal'));
    hasMex = initEikonalPaths();
    if hasMex
        fprintf('msfm2d: using MEX backend (%s)\n', ['msfm2d.' mexext]);
    else
        fprintf('msfm2d: using MATLAB backend (no MEX for this platform)\n');
    end

    p = gcp('nocreate');
    if isempty(p)
        try
            parpool;
        catch
        end
    end
    p = gcp('nocreate');
    if ~isempty(p)
        try
            pathCmd = pctPathEvalString(genRoot);
            pctRunOnAll(@() eval(pathCmd));
        catch ME
            warning('TOF:parforPathSync', 'pctRunOnAll path sync failed: %s', ME.message);
        end
    end

    %% Configuration
    cfg.imSize = 128;
    cfg.num_sensors = 64;
    cfg.num_receivers = 64;
    cfg.sos_background = 1.480;
    cfg.sos_healthy = 1.540;
    cfg.sos_tumor_mean = 1.580;
    cfg.sos_tumor_spread = 0.02;
    cfg.noise_std = 0.001;
    cfg.total_samples = 500;
    cfg.output_root = 'data_v1';
    cfg.split = [0.70, 0.15, 0.15];
    cfg.save_full_tmaps = false;
    cfg.verbose_tof = false;
    cfg.save_previews = true;
    cfg.mat_minimal = false;

    subfolders = {'train', 'val', 'test'};
    for i = 1:length(subfolders)
        mkdir(fullfile(cfg.output_root, subfolders{i}, 'mat'));
        mkdir(fullfile(cfg.output_root, subfolders{i}, 'previews'));
    end

    date_label = lower(datestr(now, 'dd_mmm_yyyy'));
    tof_cfg = make_tof_cfg(cfg);
    geom = make_tof_geom(tof_cfg);

    % Healthy ToF is identical for every sample only when noise_std == 0 (same V_h).
    % If you change expType or per-sample healthy geometry, disable this path.
    use_healthy_cache = (cfg.noise_std == 0);
    t_h_ref = [];
    tmap_h_ref = [];
    if use_healthy_cache
        cfg_ref = cfg;
        cfg_ref.num_tumours = 1;
        [V_h_ref, ~, ~] = runAnatomy('default', cfg_ref);
        [t_h_ref, ~, tmap_h_ref, ~] = tofTravelTimeFromGeom(geom, V_h_ref);
    end

    start_time = tic;
    fprintf('Starting production (label date %s)\n', date_label);

    parfor idx = 1:cfg.total_samples
        [split_name, num_tumours] = get_sample_metadata(idx, cfg);

        cfg_a = cfg;
        cfg_a.num_tumours = num_tumours;
        cfg_a.current_split = split_name;

        sample_id = sprintf('%s_prod_t%d_%05d', date_label, num_tumours, idx);

        [V_h, V_t, Mask] = runAnatomy('default', cfg_a);

        if use_healthy_cache
            th = t_h_ref;
            tmap_h = tmap_h_ref;
        else
            [th, ~, tmap_h, ~] = tofTravelTimeFromGeom(geom, V_h);
        end
        [tt, ~, tmap_t, ~] = tofTravelTimeFromGeom(geom, V_t);
        tdiff = tt - th;

        data_to_save.Vh = V_h;
        data_to_save.Vt = V_t;
        data_to_save.Mask = Mask;
        data_to_save.th = th;
        data_to_save.tt = tt;
        data_to_save.tdiff = tdiff;
        data_to_save.tmap_h = tmap_h;
        data_to_save.tmap_t = tmap_t;
        data_to_save.x_s = geom.S;
        data_to_save.x_r = geom.R;

        save_comprehensive_sample(data_to_save, cfg_a, sample_id);

        if cfg.save_previews
            prev_dir = fullfile(cfg.output_root, split_name, 'previews');
            export_png_gray(V_h, fullfile(prev_dir, ['anatomy_noTumors_' sample_id '.png']));
            export_png_gray(V_t, fullfile(prev_dir, ['anatomy_withTumors_' sample_id '.png']));
            export_png_tof(tt, fullfile(prev_dir, ['tof_' sample_id '.png']));
            export_png_tof(tdiff, fullfile(prev_dir, ['tof_diff_' sample_id '.png']));
        end
    end

    fprintf('\nProduction finished in %s\n', sec2hms(toc(start_time)));
end

%% --- helpers ---

function [split_name, num_tumours] = get_sample_metadata(idx, cfg)
    val_start = round(cfg.total_samples * cfg.split(1));
    test_start = round(cfg.total_samples * (cfg.split(1) + cfg.split(2)));

    if idx <= val_start
        split_name = 'train';
    elseif idx <= test_start
        split_name = 'val';
    else
        split_name = 'test';
    end

    num_tumours = randi([1, 5]);
end

function tf = make_tof_cfg(cfg)
    tf.imSize = cfg.imSize;
    tf.num_sensors = cfg.num_sensors;
    if isfield(cfg, 'num_receivers') && ~isempty(cfg.num_receivers)
        tf.num_receivers = cfg.num_receivers;
    else
        tf.num_receivers = cfg.num_sensors;
    end
    tf.verbose = cfg.verbose_tof;
    if isfield(cfg, 'sos_background'), tf.sos_background = cfg.sos_background; end
    if isfield(cfg, 'sos_healthy'),   tf.sos_healthy = cfg.sos_healthy; end
    if isfield(cfg, 'sos_fat'),       tf.sos_fat = cfg.sos_fat; end
    if isfield(cfg, 'sos_tumor_mean'), tf.sos_tumor_mean = cfg.sos_tumor_mean; end
end

function geom = make_tof_geom(tof_cfg)
    t = ToF();
    t.setProperties(tof_cfg);
    geom.x = t.x;
    geom.y = t.y;
    geom.z = t.z;
    geom.R = t.R;
    geom.xs_sources = t.xs_sources;
    geom.ys_sources = t.ys_sources;
    geom.number_of_sources = t.number_of_sources;
    geom.number_of_receivers = t.number_of_receivers;
    geom.verbose = t.verbose;
    geom.one_source = t.one_source;
    geom.m = t.m;
    geom.n = t.n;
    geom.S = t.S;
end

function export_png_gray(V, outpath)
    f = figure('Visible', 'off');
    imagesc(V);
    colormap gray;
    axis square;
    axis off;
    exportgraphics(f, outpath, 'Resolution', 72);
    close(f);
end

function export_png_tof(T, outpath)
    f = figure('Visible', 'off');
    imagesc(T);
    colormap jet;
    colorbar;
    axis square;
    axis off;
    exportgraphics(f, outpath, 'Resolution', 72);
    close(f);
end

function hms = sec2hms(t)
    hours = floor(t / 3600);
    t = t - hours * 3600;
    mins = floor(t / 60);
    secs = t - mins * 60;
    hms = sprintf('%02d:%02d:%05.2f', hours, mins, secs);
end
