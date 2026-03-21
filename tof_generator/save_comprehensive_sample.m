function save_comprehensive_sample(data, cfg, label)
    % SAVE_COMPREHENSIVE_SAMPLE  Write one training sample (.mat v7.3) for Python/h5py.
    %
    % data fields: Vh, Vt, Mask, th, tt, tdiff, tof_obj (ToF handle); optional tmap_h, tmap_t
    % cfg: output_root, current_split ('train'|'val'|'test'); optional save_full_tmaps

    save_path = fullfile(cfg.output_root, cfg.current_split, 'mat');
    if ~exist(save_path, 'dir')
        mkdir(save_path);
    end

    filename = fullfile(save_path, [label, '.mat']);

    D.x_s = data.tof_obj.S;
    D.x_r = data.tof_obj.R;
    D.tof_tumor_raw = data.tt;
    D.sos_map = data.Vt;
    D.sos_healthy_base = data.Vh;
    D.tof_healthy_raw = data.th;
    D.tof_diff_raw = data.tdiff;
    D.tumor_mask = data.Mask;

    if isfield(cfg, 'save_full_tmaps') && cfg.save_full_tmaps && isfield(data, 'tmap_t') && isfield(data, 'tmap_h')
        D.tof_maps_tumor = data.tmap_t;
        D.tof_maps_healthy = data.tmap_h;
        D.tof_maps_diff = data.tmap_t - data.tmap_h;
    end

    D.metadata.grid_size = size(data.Vt, 1);
    D.metadata.num_sensors = size(D.x_s, 1);
    D.metadata.timestamp = datestr(now);

    save(filename, '-struct', 'D', '-v7.3');
end
