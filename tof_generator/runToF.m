function [t_obs] = runToF(...
    number_of_sensors, grid_size, thread_label, thread_num, V, display_bent_rays) 
    tic
    tof = ToF();
    %% Set Properties %%
    tof.setProperties;
    tof.setFontSize(18);
    %% Define Transducers (by range of angles) %%
    tof.chooseGridSize(grid_size, grid_size);
    tof.chooseNumberOfSourcesAndReceivers(number_of_sensors, number_of_sensors);
    tof.chooseAngles (0, 2*pi); %(-pi/3, pi/3);
    tof.chooseOneSource(30); % from which contour 
                             % and rays will be displayed
    %% Display Anatomy and Ultrasound Setup%%
    tof.displayAnatomyAndUltrasoundSetup(V);
    %% Create Travel Times and Display ToF %%
    %[t_obs, tmap_for_plot, v_for_plot] = tof.createTravelTime(V);
    
    [t_obs, tmap_for_plot,tmap_all, v_for_plot] = tof.createTravelTime(V); 
    tof.displayToF(t_obs);
    %% Display Bent Rays %%
    if (display_bent_rays)
        tof.displayRaysAndContourOnApparatus(V, tmap_for_plot, v_for_plot);
    end
    %% Save Data for Reconstruction %%
    %tof.saveData(V,t_obs, thread_label, display_bent_rays);
    tof.saveData(V,t_obs,tmap_all, thread_label, display_bent_rays);
    tEnd = toc;
    disp(' ')
    disp(['Elapsed time for ToF simulation = ' num2str(tEnd) 's'])
end

