function runProduction1()
    % RUNPRODUCTION1  Generate paired healthy/tumour phantoms, ToF, previews, and .mat (v7.3).

    close all;
    clc;

    genRoot = fileparts(mfilename('fullpath'));
    addpath(genpath(fullfile(genRoot, 'eikonal')));
    addpath(fullfile(genRoot, '..', 'matlab_src'));

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

    subfolders = {'train', 'val', 'test'};
    for i = 1:length(subfolders)
        mkdir(fullfile(cfg.output_root, subfolders{i}, 'mat'));
        mkdir(fullfile(cfg.output_root, subfolders{i}, 'previews'));
    end

    date_label = lower(datestr(now, 'dd_mmm_yyyy'));
    tof_cfg = make_tof_cfg(cfg);

    start_time = tic;
    fprintf('Starting production (label date %s)\n', date_label);

    parfor idx = 1:cfg.total_samples
        [split_name, num_tumours] = get_sample_metadata(idx, cfg);

        cfg_a = cfg;
        cfg_a.num_tumours = num_tumours;
        cfg_a.current_split = split_name;

        sample_id = sprintf('%s_prod_t%d_%05d', date_label, num_tumours, idx);

        [V_h, V_t, Mask] = runAnatomy('default', cfg_a);

        tof_eng = ToF();
        tof_eng.setProperties(tof_cfg);
        [th, ~, tmap_h, ~] = tof_eng.createTravelTime(V_h);
        [tt, ~, tmap_t, ~] = tof_eng.createTravelTime(V_t);
        tdiff = tt - th;

        data_to_save.Vh = V_h;
        data_to_save.Vt = V_t;
        data_to_save.Mask = Mask;
        data_to_save.th = th;
        data_to_save.tt = tt;
        data_to_save.tdiff = tdiff;
        data_to_save.tmap_h = tmap_h;
        data_to_save.tmap_t = tmap_t;
        data_to_save.tof_obj = tof_eng;

        save_comprehensive_sample(data_to_save, cfg_a, sample_id);

        prev_dir = fullfile(cfg.output_root, split_name, 'previews');
        export_png_gray(V_h, fullfile(prev_dir, ['anatomy_noTumors_' sample_id '.png']));
        export_png_gray(V_t, fullfile(prev_dir, ['anatomy_withTumors_' sample_id '.png']));
        export_png_tof(tt, fullfile(prev_dir, ['tof_' sample_id '.png']));
        export_png_tof(tdiff, fullfile(prev_dir, ['tof_diff_' sample_id '.png']));
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
