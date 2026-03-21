%% PINN Tomography: Single Sample Test & Visualization
clear;
close all;
clc;

scr = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(scr, 'eikonal')));
addpath(fullfile(scr, '..', 'matlab_src'));

% 1. Setup Parameters (1px = 1mm)
cfg.imSize = 128;
cfg.num_sensors = 64;
cfg.num_receivers = 64;
cfg.num_tumours = 3;
cfg.sos_background = 1.480;
cfg.sos_healthy = 1.540;
cfg.sos_tumor_mean = 1.580;
cfg.sos_tumor_spread = 0.02;
cfg.noise_std = 0.001;
cfg.display_bent_rays = true;
cfg.verbose_tof = true;

% 2. Generate Anatomy
fprintf('Generating Anatomy...\n');
[V_healthy, V_tumours, ~] = runAnatomy('default', cfg);

% 3. Initialize ToF Engine
tof_eng = ToF();
tof_eng.setProperties(cfg);
tof_eng.chooseGridSize(cfg.imSize, cfg.imSize);
tof_eng.chooseNumberOfSourcesAndReceivers(cfg.num_sensors, cfg.num_receivers);
tof_eng.chooseOneSource(32);

% 4. Step-by-Step Visualization
fprintf('Visualizing Setup...\n');
tof_eng.displayAnatomyAndUltrasoundSetup(V_healthy);
tof_eng.displayAnatomyAndUltrasoundSetup(V_tumours);

% 5. Run Physics Simulation
fprintf('Running Eikonal Solver for empty anatomy...\n');
[t_healthy, tmap_plot, tmap_all, v_plot] = tof_eng.createTravelTime(V_healthy);

fprintf('Running Eikonal Solver (this may take a few seconds)...\n');
[t_tumours, tmap_plot, tmap_all, v_plot] = tof_eng.createTravelTime(V_tumours);

% 6. Visualize Results
fprintf('Displaying Results...\n');

tof_eng.displayToF(t_tumours);
tof_eng.displayToF(t_healthy);
tof_eng.displayToF(t_tumours - t_healthy);

if cfg.display_bent_rays
    tof_eng.displayRaysAndContourOnApparatus(V_tumours, tmap_plot, v_plot);
end

fprintf('Done. Check the open figures.\n');
