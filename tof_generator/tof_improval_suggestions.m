function [V_h, V_t, Mask] = runAnatomy(expType, cfg)
    % expType: 'Size', 'Position', 'Count', 'AnatomyVar', 'Sharpness'
    % cfg: מכיל את הערך הספציפי שנבדק בתוך הניסוי
    
    grid_size = 128; %
    ana = Anatomy();
    ana.setProperties();
    ana.chooseGridSize(grid_size, grid_size);

    % --- פרמטרי בסיס (Baseline) ---
    % נשמור על הכל קבוע חוץ מהפרמטר הנבדק
    ra_p = 30; rb_p = 35; % רדיוס פרוסטטה בסיסי
    num_t = 1; 
    ra_t_val = 8.5; rb_t_val = 6.5; % גודל בסיסי
    pos_type = 'center';
    blur_sigma = 0; % חדות גבולות (0 = חד)
    
    % --- הגדרת הניסוי ---
    switch expType
        case 'Size'
            ra_t_val = cfg.value; % שינוי רדיוס הגידול
            rb_t_val = cfg.value * 0.8;
        case 'Position'
            pos_type = cfg.type; % 'center', 'mid', 'peripheral'
        case 'Count'
            num_t = cfg.value; % 1, 2, 3...
        case 'AnatomyVar'
            ra_p = 30 + (rand-0.5) * 2 * cfg.value;
            rb_p = 35 + (rand-0.5) * 2 * cfg.value;
        case 'Sharpness'
            blur_sigma = cfg.value; % פקטור טשטוש הגבולות
    end

    % --- בניית אנטומיה בריאה ---
    V_base = ones(grid_size, grid_size) * 1.480; %
    ana.setCoordinates(ra_p, rb_p, 0);
    ana.setCenter(grid_size/2, grid_size/2);
    ana.setValue(1.540); %
    V_base = ana.addTiltedEllipse(V_base);

    % --- הוספת גידולים ---
    V_t_no_noise = V_base; Mask = zeros(grid_size, grid_size);
    for i = 1:num_t
        [x0, y0] = get_elliptical_pos(pos_type, ra_p, rb_p, ra_t_val, grid_size);
        
        ana.setCoordinates(ra_t_val, rb_t_val, randi([0, 180]));
        ana.setCenter(x0, y0);
        
        % יצירת הגידול (SoS)
        ana.setValue(1.580 + rand*0.02); %
        V_t_no_noise = ana.addTiltedEllipse(V_t_no_noise);
        
        % יצירת המסיכה
        ana.setValue(1);
        Mask = ana.addTiltedEllipse(Mask);
    end
    
    % טיפול בחדות גבולות (Sharpness) אם נדרש
    if blur_sigma > 0
        V_t_no_noise = imgaussfilt(V_t_no_noise, blur_sigma);
    end

    % הוספת רעש זהה
    noise = randn(grid_size, grid_size) * 0.001; 
    V_h = V_base + noise;
    V_t = V_t_no_noise + noise;
end


function [V] = addTiltedEllipse(this, V)
    % יצירת נקודות האליפסה
    [ex, ey] = ellipse(this.ra, this.rb, this.ang, this.x0, this.y0);
    
    % יצירת גריד של כל הנקודות בתמונה
    [gridX, gridY] = meshgrid(this.x, this.y);
    
    % מציאת כל הפיקסלים שנמצאים בתוך האליפסה בבת אחת
    inMask = inpolygon(gridX, gridY, ex, ey);
    
    % עדכון המטריצה ללא לולאות
    V(inMask) = this.value;
end


function [x, y] = get_elliptical_pos(type, ra_p, rb_p, ra_t, gs)
    % פונקציית עזר למיקום בתוך האליפסה
    margin = ra_t + 2;
    if strcmp(type, 'center')
        x = gs/2 + (rand-0.5)*5; y = gs/2 + (rand-0.5)*5;
    elseif strcmp(type, 'peripheral')
        ang = rand*2*pi;
        x = gs/2 + (ra_p-margin)*cos(ang); y = gs/2 + (rb_p-margin)*sin(ang);
    else % random
        ang = rand*2*pi; r = sqrt(rand)*(ra_p-margin);
        x = gs/2 + r*cos(ang); y = gs/2 + r*sin(ang);
    end
end


.............. RUN PRODUCTION .................


function runProduction1()
    % Simplified Unified pipeline for Anomaly ToF Generation
    cfg.imSize = 128; %
    cfg.num_sensors = 64; %
	
	
	cfg.num_tumours = 1;
	cfg.tumor_size_multi = 0.5; % גידול קטן פי 2
	cfg.pos_type = 'peripheral';
	cfg.noise_std = 0.001;
	num_samples = 10;



    cfg.total_samples = 500; %
    cfg.output_root = 'dataset_anomaly_v1'; %
    cfg.split = [0.70, 0.15, 0.15]; %

    % Directory Setup
    subfolders = {'train', 'val', 'test'};
    for i = 1:length(subfolders)
        mkdir(fullfile(cfg.output_root, subfolders{i}, 'mat'));
    end

    genRoot = fileparts(mfilename('fullpath'));
    addpath(fullfile(genRoot, 'eikonal'));
    initEikonalPaths();
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

    parfor idx = 1:cfg.total_samples %
        % 1. Meta-data and Paths
        [split_name, num_tumours] = get_sample_metadata(idx, cfg); %
        sample_id = sprintf('anom_%05d', idx);
        
        % 2. Generate Anatomy Pair & Mask
        [V_h, V_t, Mask] = runAnatomy(cfg.imSize, num_tumours, cfg.imSize);
        
        % 3. Run Physics twice to get Difference (t_diff)
        t_h = run_silent_ToF(cfg.num_sensors, cfg.imSize, V_h);
        t_t = run_silent_ToF(cfg.num_sensors, cfg.imSize, V_t);
        t_diff = t_t - t_h; % The Anomaly Signal
        
        % 4. Save for Python (Input: t_diff, Label: Mask)
        save_anomaly_sample(t_diff, Mask, split_name, sample_id, cfg);
    end
end

function [t_obs] = run_silent_ToF(num_sensors, imSize, V)
    % Returns raw ToF matrix without saving files
    tof_eng = ToF();
    tof_eng.setProperties(struct('imSize', imSize, 'num_sensors', num_sensors, 'num_receivers', num_sensors)); %
    [t_obs, ~, ~, ~] = tof_eng.createTravelTime(V); %
end

function save_anomaly_sample(t_diff, Mask, split, id, cfg)
    % Saves the training pair into a single .mat file
    dest = fullfile(cfg.output_root, split, 'mat', [id, '.mat']);
    save(dest, 't_diff', 'Mask');
end







......................... analyze_difficulty ...............................
%% Parameter Sweep & D_cat Diagnostic
exp_name = 'Size'; % הפרמטר שאנחנו בוחנים
values_to_test = [2, 4, 6, 8, 10]; % רדיוס הגידול במ"מ
num_samples = 10;
d_scores = [];

tof_eng = ToF(); %
tof_eng.chooseGridSize(128, 128); %
tof_eng.chooseNumberOfSourcesAndReceivers(64, 64); %

for v = 1:length(values_to_test)
    cfg_test.value = values_to_test(v);
    tofs = cell(num_samples, 1);
    signals = zeros(num_samples, 1);
    
    for i = 1:num_samples
        [Vh, Vt, ~] = runAnatomy(exp_name, cfg_test);
        th = tof_eng.createTravelTime(Vh); %
        tt = tof_eng.createTravelTime(Vt);
        
        signals(i) = norm(tt(:) - th(:), 2);
        tofs{i} = tt;
    end
    
    % חישוב שונות בתוך הקטגוריה (Intra-Category)
    intra_diffs = [];
    for i = 1:num_samples
        for j = i+1:num_samples
            intra_diffs(end+1) = norm(tofs{i}(:) - tofs{j}(:), 2);
        end
    end
    
    d_scores(v) = mean(signals) / mean(intra_diffs);
    fprintf('Value: %.1f | D_cat: %.4f\n', values_to_test(v), d_scores(v));
end

% הצגת גרף קושי
figure; plot(values_to_test, d_scores, '-o');
xlabel(['Parameter: ', exp_name]); ylabel('D_{cat} Score');
title('Learning Capability vs. Parameter Difficulty');
grid on;




%% Anatomy Variation Sweep for 5mm Tumor
exp_name = 'AnatomyVar'; 
var_levels = [0, 0.5, 1, 2, 3, 5]; % שונות ברדיוס הפרוסטטה במ"מ
num_samples = 10;
d_scores = [];

cfg_test.value = 5; % רדיוס הגידול קבוע על 5 מ"מ

for v = 1:length(var_levels)
    cfg_var.value = var_levels(v); % רמת השונות האנטומית הנוכחית
    tofs = cell(num_samples, 1);
    signals = zeros(num_samples, 1);
    
    for i = 1:num_samples
        % מייצר זוג (בריא/חולה) עם שונות אנטומית בתוך הסט
        [Vh, Vt, ~] = runAnatomy(exp_name, cfg_var); 
        th = tof_eng.createTravelTime(Vh); %
        tt = tof_eng.createTravelTime(Vt); %
        
        signals(i) = norm(tt(:) - th(:), 2);
        tofs{i} = tt;
    end
    
    % חישוב השונות הנגרמת מהאנטומיה המשתנה
    intra_diffs = [];
    for i = 1:num_samples
        for j = i+1:num_samples
            d = tofs{i} - tofs{j};
            intra_diffs(end+1) = norm(d(:), 2);
        end
    end
    
    d_scores(v) = mean(signals) / mean(intra_diffs);
    fprintf('Anatomy Var: +/-%.1f mm | D_cat: %.4f\n', var_levels(v), d_scores(v));
end



................................................

function save_comprehensive_sample(data, cfg, label)
    % data: struct containing [Vh, Vt, Mask, th, tt, tdiff, tmap_h, tmap_t]
    % cfg: global configuration
    % label: sample ID (e.g., 'anom_00001')

    % קביעת נתיב השמירה (לפי ה-split המוגדר ב-cfg)
    save_path = fullfile(cfg.output_root, cfg.current_split, 'mat');
    if ~exist(save_path, 'dir'), mkdir(save_path); end
    
    filename = fullfile(save_path, [label, '.mat']);

    %% 1. חובה לכל השלבים (Core Data)
    D.x_s = data.tof_obj.S;             % מיקומי מקורות
    D.x_r = data.tof_obj.R;             % מיקומי מקלטים
    D.tof_tumor_raw = data.tt;          % מטריצת ToF עם גידולים
    D.sos_map = data.Vt;                % מפת מהירות הקול (Ground Truth)
    D.sos_healthy_base = data.Vh;       % מפת הייחוס הבריאה (כולל אנטומיה)

    %% 2. חשוב ל־Stage 0 / מחקר (Differential Analysis)
    D.tof_healthy_raw = data.th;        % ToF ללא גידולים
    D.tof_diff_raw = data.tdiff;        % האות שהרשת לומדת (Anomaly)
    D.tumor_mask = data.Mask;           % מסיכה בינארית (Label ל-U-Net)

    %% 3. חשוב ל־debug / validation (Physics Data)
    % שימוש בדגל ב-cfg כדי לא להכביד על הדיסק בייצור המוני
    if isfield(cfg, 'save_full_tmaps') && cfg.save_full_tmaps
        D.tof_maps_tumor = data.tmap_t;   % כל מפות הזמנים (Eikonal solution)
        D.tof_maps_healthy = data.tmap_h; 
        D.tof_maps_diff = data.tmap_t - data.tmap_h;
    end

    % מטא-דאטה נוסף
    D.metadata.grid_size = size(data.Vt, 1);
    D.metadata.num_sensors = size(D.x_s, 1);
    D.metadata.timestamp = datestr(now);

    % שמירה בפורמט v7.3 המאפשר קריאה קלה ב-Python (h5py)
    save(filename, '-struct', 'D', '-v7.3');
end




% בתוך הלולאה
[Vh, Vt, Mask] = runAnatomy(expType, cfg);
th = tof_eng.createTravelTime(Vh);
tt = tof_eng.createTravelTime(Vt);

% ריכוז הנתונים
data_to_save.Vh = Vh; data_to_save.Vt = Vt; data_to_save.Mask = Mask;
data_to_save.th = th; data_to_save.tt = tt; data_to_save.tdiff = tt - th;
data_to_save.tof_obj = tof_eng; % לצורך x_s, x_r

% קריאה לפונקציית השמירה
save_comprehensive_sample(data_to_save, cfg, sample_id);


