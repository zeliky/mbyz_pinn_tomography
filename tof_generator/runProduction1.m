function runProduction1()
    % RUNPRODUCTION - Unified data generation pipeline for PINN Tomography
    % Generates synthetic prostate phantoms and their corresponding ToF data.
    
    clear; close all; clc;
    addpath(genpath('eikonal')); % Ensure solver is in path

    %% --- Configuration ---
    cfg.imSize = 128;               % 1px = 1mm
    cfg.num_sensors = 64;           % Increased for better resolution
    cfg.total_samples = 500;        % Total number of phantoms to generate
    cfg.display_bent_rays = true;  % Keep false for mass production
    cfg.output_root = 'data_v1';    % Base folder for dataset
    
    % Split ratios: 70% Train, 15% Val, 15% Test
    cfg.split = [0.70, 0.15, 0.15]; 

    %% --- Setup Environment ---
    start_time = tic;
    timestamp = datestr(now, 'yyyy_mm_dd_HH_MM');
    fprintf('Starting Production: %s\n', timestamp);
    
    % Create directory structure
    subfolders = {'train', 'val', 'test'};
    for i = 1:length(subfolders)
        mkdir(fullfile(cfg.output_root, subfolders{i}, 'mat'));
        mkdir(fullfile(cfg.output_root, subfolders{i}, 'previews'));
    end

    %% --- Parallel Production Loop ---
    % Use parfor if Parallel Computing Toolbox is available
    parfor idx = 1:cfg.total_samples
        % 1. Determine Split and Tumor Count
        [split_name, num_tumours] = get_sample_metadata(idx, cfg);
        
        % 2. Generate Thread Label
        sample_id = sprintf('sample_%05d_t%d', idx, num_tumours);
        
        % 3. Run Physics Engine
        % Generate Anatomy (SoS map V)
        V = runAnatomy(cfg.imSize, num_tumours, sample_id, cfg.imSize);
        
        % Run ToF Simulation
        % Note: We use 'invisible' figures to prevent GUI crashing
        [t_obs] = run_silent_ToF(cfg.num_sensors, cfg.imSize, sample_id, V, cfg.display_bent_rays);
        
        % 4. Save Data and Previews
        save_production_sample(V, t_obs, split_name, sample_id, cfg);
        
        if mod(idx, 10) == 0
            fprintf('Completed sample %d/%d (%s)\n', idx, cfg.total_samples, split_name);
        end
    end

    %% --- Finalize ---
    total_duration = toc(start_time);
    fprintf('\nProduction Finished!\nTotal Time: %s\n', sec2hms(total_duration));
end

%% --- Helper Logic ---

function [split_name, num_tumours] = get_sample_metadata(idx, cfg)
    % Assign split based on index
    val_start = cfg.total_samples * cfg.split(1);
    test_start = cfg.total_samples * (cfg.split(1) + cfg.split(2));
    
    if idx <= val_start
        split_name = 'train';
    elseif idx <= test_start
        split_name = 'val';
    else
        split_name = 'test';
    end
    
    % Randomize tumor count (1 to 5) to ensure diversity
    num_tumours = randi([1, 5]);
end

function [t_obs] = run_silent_ToF(num_sensors, imSize, label, V, show_rays)
    % Wrapper to run ToF class without popping up windows
    tof_eng = ToF();
    tof_eng.setProperties();
    tof_eng.chooseGridSize(imSize, imSize);
    tof_eng.chooseNumberOfSourcesAndReceivers(num_sensors, num_sensors);
    
    % Computation
    [t_obs, ~, tmap_all, ~] = tof_eng.createTravelTime(V);
    
    % Save mat file via the class method (modified to be silent)
    tof_eng.saveData(V, t_obs, tmap_all, label, show_rays);
    
    % Close any figures opened by the class
    close all; 
end

function save_production_sample(V, t_obs, split, id, cfg)
    % Saves a visual preview for quick inspection without loading .mat
    dest_img = fullfile(cfg.output_root, split, 'previews', [id, '.png']);
    
    % Create a composite image: Anatomy | ToF
    f = figure('Visible', 'off'); 
    subplot(1,2,1); imagesc(V); colormap gray; title('Anatomy'); axis square; axis off;
    subplot(1,2,2); imagesc(t_obs); colormap jet; title('ToF'); axis square; axis off;
    
    % Save and cleanup
    exportgraphics(f, dest_img, 'Resolution', 72);
    close(f);
    
    % Move the .mat file created by ToF.saveData to the correct split folder
    % Assuming ToF.saveData saves to 'TimeOfFlightData/'
    source_mat = fullfile('TimeOfFlightData', ['ToF_', id, '.mat']);
    dest_mat = fullfile(cfg.output_root, split, 'mat', [id, '.mat']);
    
    if exist(source_mat, 'file')
        movefile(source_mat, dest_mat);
    end
end

function hms = sec2hms(t)
    hours = floor(t/3600);
    t = t - hours*3600;
    mins = floor(t/60);
    secs = t - mins*60;
    hms = sprintf('%02d:%02d:%05.2f', hours, mins, secs);
end