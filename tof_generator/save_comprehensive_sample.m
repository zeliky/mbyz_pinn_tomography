function save_comprehensive_sample(data, cfg, label)
    % SAVE_COMPREHENSIVE_SAMPLE  Write one training sample (.mat v7.3) for Python/h5py.
    %
    % data: Vh, Vt, Mask, th, tt, tdiff; x_s/x_r (sensor matrices) or tof_obj (ToF handle)
    % cfg: output_root, current_split; optional save_full_tmaps, mat_minimal
    %
    % When cfg.mat_minimal is true, only tensors needed for scale-up training are saved
    % (requires a matching Python loader; default path stays backward compatible).

    save_path = fullfile(cfg.output_root, cfg.current_split, 'mat');
    if ~exist(save_path, 'dir')
        mkdir(save_path);
    end

    filename = fullfile(save_path, [label, '.mat']);

    if isfield(data, 'x_s') && isfield(data, 'x_r')
        x_s = data.x_s;
        x_r = data.x_r;
    else
        x_s = data.tof_obj.S;
        x_r = data.tof_obj.R;
    end

    if isfield(cfg, 'mat_minimal') && cfg.mat_minimal
        D.tof_diff_raw = data.tdiff;
        D.sos_map = data.Vt;
        D.tumor_mask = data.Mask;
        D.x_s = x_s;
        D.x_r = x_r;
        D.metadata.grid_size = size(data.Vt, 1);
        D.metadata.num_sensors = size(x_s, 1);
        D.metadata.timestamp = datestr(now);
        save(filename, '-struct', 'D', '-v7.3');
        return
    end

    D.x_s = x_s;
    D.x_r = x_r;
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
