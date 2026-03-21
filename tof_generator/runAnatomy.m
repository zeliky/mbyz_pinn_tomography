function [V_h, V_t, Mask] = runAnatomy(expType, cfg)
    % RUNANATOMY  Build healthy and tumour SoS maps and a binary tumour mask.
    %
    %   [V_h, V_t, Mask] = runAnatomy(expType, cfg)
    %
    % Units: 1 pixel = 1 mm; speeds in mm/us.
    %
    % expType (case-insensitive):
    %   'default'    – production-style random tumours; uses cfg.num_tumours; no
    %                  experiment-specific overrides.
    %   'Size'       – tumour scale from cfg.value; aspect ratio rho = ra_t/rb_t
    %                  sampled (Normal, clipped to cfg bounds).
    %   'Position'   – cfg.type is 'center' | 'mid' | 'peripheral'.
    %   'Count'      – cfg.value = number of tumours (e.g. 1,2,3,5).
    %   'AnatomyVar' – cfg.value = +/- jitter on prostate radii (mm).
    %   'Sharpness'  – cfg.value = Gaussian sigma on tumour map (imgaussfilt).
    %
    % cfg fields (optional defaults in parentheses):
    %   imSize, grid_size (128) – grid side length
    %   sos_background (1.480), sos_healthy (1.540)
    %   sos_tumor_mean (1.580), sos_tumor_spread (0.02)  -> mean + rand*spread
    %   noise_std (0.001)
    %   num_tumours – required for 'default'
    %   tumor_aspect_ratio_min/max/mean/std (0.8 / 1.2 / 1.0 / 0.1) for Size
    %
    % Outputs:
    %   V_h   – healthy anatomy (background + prostate + noise)
    %   V_t   – same + tumours (+ optional blur) + same noise as V_h
    %   Mask  – binary mask (1 inside tumour ellipses)

    grid_size = get_cfg_field(cfg, 'imSize', 128);
    if isfield(cfg, 'grid_size') && ~isempty(cfg.grid_size)
        grid_size = cfg.grid_size;
    end

    sos_bg = get_cfg_field(cfg, 'sos_background', 1.480);
    sos_h = get_cfg_field(cfg, 'sos_healthy', 1.540);
    sos_tm = get_cfg_field(cfg, 'sos_tumor_mean', 1.580);
    sos_tspread = get_cfg_field(cfg, 'sos_tumor_spread', 0.02);
    noise_std = get_cfg_field(cfg, 'noise_std', 0.001);

    ana = Anatomy();
    ana.setProperties();
    ana.chooseGridSize(grid_size, grid_size);

    % Baseline (fixed except where experiments override)
    ra_p = 30;
    rb_p = 35;
    num_t = 1;
    ra_t_val = 8.5;
    rb_t_val = 6.5;
    pos_type = 'center';
    blur_sigma = 0;

    et = lower(strtrim(expType));

    switch et
        case 'size'
            ra_t_val = cfg.value;
            rho = sample_tumor_aspect_ratio(cfg);
            rb_t_val = ra_t_val / rho;
        case 'position'
            pos_type = cfg.type;
        case 'count'
            num_t = cfg.value;
        case 'anatomyvar'
            ra_p = 30 + (rand - 0.5) * 2 * cfg.value;
            rb_p = 35 + (rand - 0.5) * 2 * cfg.value;
        case 'sharpness'
            blur_sigma = cfg.value;
        case 'default'
            if ~isfield(cfg, 'num_tumours') || isempty(cfg.num_tumours)
                error('runAnatomy:default requires cfg.num_tumours');
            end
            num_t = cfg.num_tumours;
            % All other baseline knobs stay; per-tumour geometry drawn in loop below.
        otherwise
            error('runAnatomy: unknown expType ''%s''', expType);
    end

    % --- Healthy base ---
    V_base = ones(grid_size, grid_size) * sos_bg;
    ana.setCoordinates(ra_p, rb_p, 0);
    ana.setCenter(grid_size / 2, grid_size / 2);
    ana.setValue(sos_h);
    V_base = ana.addTiltedEllipse(V_base);

    % --- Tumours ---
    V_t_no_noise = V_base;
    Mask = zeros(grid_size, grid_size);

    if strcmp(et, 'default')
        ra_mean = 8.5;
        rb_mean = 6.5;
        spread_factor = 4;
        for i = 1:num_t
            ra_t = random('Normal', ra_mean, ra_mean / spread_factor);
            rb_t = random('Normal', rb_mean, rb_mean / spread_factor);
            dist_from_center = (ra_p - ra_t - 2);
            x0_t = grid_size / 2 + (rand(1) - 0.5) * 2 * dist_from_center;
            y0_t = grid_size / 2 + (rand(1) - 0.5) * 2 * dist_from_center;
            ana.setCoordinates(ra_t, rb_t, randi([0, 180]));
            ana.setCenter(x0_t, y0_t);
            ana.setValue(sos_tm + rand * sos_tspread);
            V_t_no_noise = ana.addTiltedEllipse(V_t_no_noise);
            ana.setValue(1);
            Mask = ana.addTiltedEllipse(Mask);
        end
    else
        for i = 1:num_t
            if strcmp(et, 'size')
                rho_i = sample_tumor_aspect_ratio(cfg);
                ra_i = cfg.value;
                rb_i = ra_i / rho_i;
            else
                ra_i = ra_t_val;
                rb_i = rb_t_val;
            end
            [x0, y0] = ana.ellipticalTumorPosition(pos_type, ra_p, rb_p, ra_i, grid_size);
            ana.setCoordinates(ra_i, rb_i, randi([0, 180]));
            ana.setCenter(x0, y0);
            ana.setValue(sos_tm + rand * sos_tspread);
            V_t_no_noise = ana.addTiltedEllipse(V_t_no_noise);
            ana.setValue(1);
            Mask = ana.addTiltedEllipse(Mask);
        end
    end

    if blur_sigma > 0
        V_t_no_noise = imgaussfilt(V_t_no_noise, blur_sigma);
    end

    noise = randn(grid_size, grid_size) * noise_std;
    V_h = V_base + noise;
    V_t = V_t_no_noise + noise;
end

function v = get_cfg_field(s, name, defaultVal)
    if isfield(s, name) && ~isempty(s.(name))
        v = s.(name);
    else
        v = defaultVal;
    end
end

function rho = sample_tumor_aspect_ratio(cfg)
    lo = get_cfg_field(cfg, 'tumor_aspect_ratio_min', 0.8);
    hi = get_cfg_field(cfg, 'tumor_aspect_ratio_max', 1.2);
    mu = get_cfg_field(cfg, 'tumor_aspect_ratio_mean', 1.0);
    sig = get_cfg_field(cfg, 'tumor_aspect_ratio_std', 0.1);
    rho = mu + sig * randn;
    rho = min(max(rho, lo), hi);
end

function cVal = getRandomTumourSpeed()
    malignant_range = [1.560, 1.600];
    cVal = random('Uniform', malignant_range(1), malignant_range(2));
    cVal = cVal + 0.001 * randn();
end
