addpath('eikonal\');
addpath('eikonal\fast_marching_kroon\');
addpath('eikonal\fast_marching_kroon\functions\');


%% PINN Tomography: Single Sample Test & Visualization
clear; close all; clc;

% 1. Setup Parameters (1px = 1mm)
cfg.imSize = 128;
cfg.num_sensors = 64;
cfg.num_receivers = 64;
cfg.num_tumours = 3;
cfg.display_bent_rays = true; % We want to see the rays now

% 2. Generate Anatomy
fprintf('Generating Anatomy...\n');
% Note: Using a fixed label for testing
[V_healthy, V_tumours] = runAnatomy(cfg.imSize, cfg.num_tumours, 'test_unit', cfg.imSize);


% 3. Initialize ToF Engine
tof_eng = ToF();
tof_eng.setProperties(cfg);
tof_eng.chooseGridSize(cfg.imSize, cfg.imSize);
tof_eng.chooseNumberOfSourcesAndReceivers(cfg.num_sensors, cfg.num_receivers);
tof_eng.chooseOneSource(32); % Let's look at the rays from the middle source




% 4. Step-by-Step Visualization
fprintf('Visualizing Setup...\n');
% This shows the sensors (Green/Red) on top of the anatomy
tof_eng.displayAnatomyAndUltrasoundSetup(V_healthy);
tof_eng.displayAnatomyAndUltrasoundSetup(V_tumors);

% 5. Run Physics Simulation
fprintf('Running Eikonal Solver for empty anathomy...\n');
[t_healthy, tmap_plot, tmap_all, v_plot] = tof_eng.createTravelTime(V_healthy);

fprintf('Running Eikonal Solver (this may take a few seconds)...\n');
[t_tumors, tmap_plot, tmap_all, v_plot] = tof_eng.createTravelTime(V_tumors);

% 6. Visualize Results
fprintf('Displaying Results...\n');

% Show the ToF Matrix
tof_eng.displayToF(t_tumors);
tof_eng.displayToF(t_healthy);
tof_eng.displayToF(t_tumors - t_healthy);

% Show Wavefronts and Bent Rays
% This calls CalculateL_Forward internally to trace rays
if cfg.display_bent_rays
    tof_eng.displayRaysAndContourOnApparatus(V_tumors, tmap_plot, v_plot);
end

fprintf('Done. Check the open figures.\n');