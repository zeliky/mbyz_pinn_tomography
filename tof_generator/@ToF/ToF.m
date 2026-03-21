classdef ToF < handle
    % TOF - Tomography Simulation Class
    % Organized version for PINN Rebuild
    
    properties (SetAccess = private)
        %% Grid Configuration
        m, n;           % Grid dimensions (pixels)
        x, y, z;        % Coordinate vectors
        margin;         % Boundary margin
        
        %% Sensor Configuration (Transducers)
        number_of_sources;
        number_of_receivers;
        xs_sources, ys_sources; % Source positions
        xs_receivers, ys_receivers; % Receiver positions
        S, R;           % Combined coordinate matrices
        min_angle, max_angle;
        one_source;     % Index for visualization
        
        %% Physics & Medium Properties (Units: mm/us)
        c0;             % Background (Water)
        c;              % Healthy tissue
        c1;             % Fat
        c2;             % Tumor
        delta_f_refractivity;
        
        %% Visualization & UI
        fig1, fig2, fig3;
        fig1_name, fig2_name, fig3_name;
        font_size;
        verbose = true;  % set false in cfg for silent sweeps
    end
    
    methods
        %% --- Initialization & Setup ---
        function setProperties(this, cfg)
            this.font_size = 14;
            set(0, 'DefaultAxesFontSize', this.font_size);
            set(0, 'DefaultTextFontSize', this.font_size);
            
            if isfield(cfg, 'verbose')
                this.verbose = logical(cfg.verbose);
            else
                this.verbose = true;
            end
            
            imSz = 128;
            if isfield(cfg, 'imSize') && ~isempty(cfg.imSize)
                imSz = cfg.imSize;
            end
            % Grid Defaults
            this.m = imSz;
            this.n = imSz;
            this.x = linspace(1, this.m, this.m);
            this.y = linspace(1, this.n, this.n);
            this.z = 1;
            this.margin = 10;
            
            % Sensor Defaults
            ns = 64;
            nr = 64;
            if isfield(cfg, 'num_sensors') && ~isempty(cfg.num_sensors)
                ns = cfg.num_sensors;
            end
            if isfield(cfg, 'num_receivers') && ~isempty(cfg.num_receivers)
                nr = cfg.num_receivers;
            else
                nr = ns;
            end
            this.number_of_sources = ns;
            this.number_of_receivers = nr;
            this.max_angle = 2*pi;
            this.min_angle = 0;
            
            % Physics Constants (mm/us) — defaults; override via cfg from runAnatomy
            this.c0 = 1.480;
            this.c  = 1.540;
            this.c1 = 1.450;
            this.c2 = 1.580;
            if isfield(cfg, 'sos_background'), this.c0 = cfg.sos_background; end
            if isfield(cfg, 'sos_healthy'),   this.c  = cfg.sos_healthy; end
            if isfield(cfg, 'sos_fat'),       this.c1 = cfg.sos_fat; end
            if isfield(cfg, 'sos_tumor_mean'), this.c2 = cfg.sos_tumor_mean; end
            
            this.setReceiversAndSources();
            
            % Calculated Physics
            refractivity1 = (this.c0^2) / (this.c^2);
            this.delta_f_refractivity = refractivity1 - 1;
        end
        
        function setReceiversAndSources(this)
            % Radius: Leave some space from the grid edge
            radius = (min(this.m, this.n) - 10) / 2;
            theta_s = linspace(this.min_angle, this.max_angle, this.number_of_sources);
            theta_r = linspace(this.min_angle, this.max_angle, this.number_of_receivers);
            center = [this.m/2, this.n/2] + 0.5;
            
            % Sources (Green in plots)
            this.xs_sources = center(1) + radius * cos(theta_s);
            this.ys_sources = center(2) + radius * sin(theta_s);
            this.S = [this.xs_sources', this.ys_sources', ones(this.number_of_sources, 1)];
            
            % Receivers (Red in plots)
            this.xs_receivers = center(1) + radius * cos(theta_r);
            this.ys_receivers = center(2) + radius * sin(theta_r);
            this.R = [this.xs_receivers', this.ys_receivers', ones(this.number_of_receivers, 1)];
            
            this.one_source = round(this.number_of_sources / 2);
        end
        
        function chooseGridSize(this, m, n)
            this.m = m; this.n = n;
            this.x = linspace(1, m, m);
            this.y = linspace(1, n, n);
            this.setReceiversAndSources();
        end
        
        function chooseNumberOfSourcesAndReceivers(this, ns, nr)
            this.number_of_sources = ns;
            this.number_of_receivers = nr;
            this.setReceiversAndSources();
        end

        function chooseAngles(this, from, to)
            this.min_angle = from; this.max_angle = to;
            this.setReceiversAndSources();
        end

        function chooseOneSource(this, idx), this.one_source = idx; end
        function setFontSize(this, fs), this.font_size = fs; end

        %% --- Core Simulation Logic ---
        function [t_obs, tmap_for_plot, tmap_all, v_for_plot] = createTravelTime(this, V)
            t_obs = zeros(this.number_of_sources, this.number_of_receivers);  
            tmap_all = zeros(this.number_of_sources, this.n, this.m);
            v_for_plot = [];
            
            if this.verbose
                fprintf('Running Eikonal for %d sources...\n', this.number_of_sources);
            end
            for is = 1:this.number_of_sources
                % Define source position for this iteration
                S_current = [ones(this.number_of_receivers, 1) * this.xs_sources(is), ...
                            ones(this.number_of_receivers, 1) * this.ys_sources(is), ...
                            ones(this.number_of_receivers, 1)]; 
                
                [tmap, t] = eikonal_traveltime(this.x, this.y, this.z, V, S_current, this.R);
                
                % Store results
                t_obs(is, :) = t(:);
                tmap_all(is, :, :) = squeeze(tmap);
                
                % Setup plot data for the designated "one_source"
                if is == this.one_source
                    tmap_for_plot = tmap;
                    v_for_plot = min(tmap(:)):(max(tmap(:))-min(tmap(:)))/100:max(tmap(:));
                end
            end
        end

        %% --- Visualization Methods ---
        function displayAnatomyAndUltrasoundSetup(this, V)
            this.fig1 = figure('Name', 'Anatomy Setup');
            imagesc(this.x, this.y, V); hold on; colorbar; colormap gray;
            
            % Plot Receivers (Red)
            plot(this.xs_receivers, this.ys_receivers, 'w--', 'LineWidth', 0.5);
            scatter(this.xs_receivers, this.ys_receivers, 20, 'r', 'filled', 'MarkerEdgeColor', 'k');
            
            % Plot Sources (Green)
            plot(this.xs_sources, this.ys_sources, 'w--', 'LineWidth', 0.5);
            scatter(this.xs_sources, this.ys_sources, 20, 'g', 'filled', 'MarkerEdgeColor', 'k');
            
            title('Input Anatomy & Sensor Ring');
            xlabel('x (mm)'); ylabel('y (mm)'); axis square; axis tight;
            set(gca, 'FontWeight', 'bold');
        end

        function displayToF(this, t_obs)
            this.fig2 = figure('Name', 'Measured ToF');
            imagesc(t_obs');
            title('Time Of Flight (TOF) Matrix');
            xlabel('Source Index'); ylabel('Receiver Index');
            colorbar; colormap jet; axis square;
            set(gca, 'FontWeight', 'bold');
        end

        function displayRaysAndContourOnApparatus(this, V, tmap_for_plot, v_for_plot)
            this.fig3 = figure('Name', 'Ray Tracing');
            contour(this.x, this.y, tmap_for_plot, v_for_plot);
            set(gca, 'YDir', 'reverse'); hold on; colormap gray;
            
            % Plot Ring
            plot(this.xs_sources, this.ys_sources, 'k:');
            scatter(this.xs_sources, this.ys_sources, 15, 'g', 'filled');
            scatter(this.xs_receivers, this.ys_receivers, 15, 'r', 'filled');
            
            % Highlight Active Source
            scatter(this.xs_sources(this.one_source), this.ys_sources(this.one_source), ...
                    50, 'y', 'filled', 'MarkerEdgeColor', 'k');

            % Calculate and Plot Rays
            input_to_L = this.get_input_to_L();
            CalculateL_Forward(V, input_to_L); 
            
            title(['Wavefronts & Rays (Source #' num2str(this.one_source) ')']);
            xlabel('x (mm)'); ylabel('y (mm)'); axis square;
        end

        %% --- Data Management ---
        function saveData(this, V, t_obs, tmap_all, thread_label, display_bent_rays)
            % D Structure for Python/PINN import
            D = struct();
            D.V = V;
            D.t_obs = t_obs;
            D.receivers = [this.xs_receivers; this.ys_receivers];
            D.sources = [this.xs_sources; this.ys_sources];
            D.exp_t_obs = expand_T_obs(this, t_obs);
            D.tmap = tmap_all;
            D.grid_metadata = [this.m, this.n, this.font_size];

            if ~exist('TimeOfFlightData', 'dir'), mkdir('TimeOfFlightData'); end
            if ~exist('Figures', 'dir'), mkdir('Figures'); end

            filename = fullfile('TimeOfFlightData', ['ToF_', thread_label, '.mat']);
            save(filename, '-struct', 'D');
            
            % Save Figures
            if ~isempty(this.fig1) && ishandle(this.fig1)
                saveas(this.fig1, fullfile('Figures', ['Anatomy_', thread_label, '.fig']));
            end
            
            if ~isempty(this.fig2) && ishandle(this.fig2)
                saveas(this.fig2, fullfile('Figures', ['ToF_', thread_label, '.fig']));
            end
            
            if display_bent_rays && ~isempty(this.fig3) && ishandle(this.fig3)
                saveas(this.fig3, fullfile('Figures', ['Rays_', thread_label, '.fig']));
            end
        end
    end
    
    methods (Access = private)
        function s = get_input_to_L(this)
            s.x = this.x; s.y = this.y; s.z = this.z;
            s.xs_receivers = this.xs_receivers; s.ys_receivers = this.ys_receivers;
            s.xs_sources = this.xs_sources; s.ys_sources = this.ys_sources;
            s.margin = this.margin;
            s.doPlotRaypathsInCalculateL = 1;
            s.one_source = this.one_source;
        end
    end
end

%% --- Helper Functions (External to Class) ---

function [result] = expand_T_obs(obj, t_obs)
    [ns, nr] = size(t_obs);
    result = zeros(ns * nr, 5);
    idx = 1;
    for i = 1:ns
        for j = 1:nr
            result(idx, :) = [obj.xs_sources(i), obj.ys_sources(i), ...
                             obj.xs_receivers(j), obj.ys_receivers(j), ...
                             t_obs(i, j)];
            idx = idx + 1;
        end
    end
end



function [RL, RL_accurate, L] = CalculateL_Forward(V, input_to_L)
    x = input_to_L.x; y = input_to_L.y; z = input_to_L.z;
    xs_rec = input_to_L.xs_receivers; ys_rec = input_to_L.ys_receivers;
    xs_src = input_to_L.xs_sources;   ys_src = input_to_L.ys_sources;
    doPlot = input_to_L.doPlotRaypathsInCalculateL;
    one_src_idx = input_to_L.one_source;
    
    num_rec = length(xs_rec);
    num_src = length(xs_src);
    m = length(x); n = length(y);
    
    RL = zeros(num_src, num_rec);
    RL_accurate = zeros(num_src, num_rec);
    
    L = []; 

    R_matrix = [xs_rec', ys_rec', ones(num_rec, 1)];

    for is = 1:num_src
       
        S_current = [ones(num_rec, 1)*xs_src(is), ones(num_rec, 1)*ys_src(is), ones(num_rec, 1)];
        [tmap, ~] = eikonal_traveltime(x, y, z, V, S_current, R_matrix);
        
        from_point = [xs_src(is), ys_src(is)];

        for ir = 1:num_rec
            if (xs_rec(ir) == xs_src(is) && ys_rec(ir) == ys_src(is)), continue; end
            
            to_point = [xs_rec(ir), ys_rec(ir)];
            
            dx = x(2)-x(1);
            [len, path] = EikonalRayPath(x, y, V, from_point, to_point, tmap, [dx, length(x)]);
            RL(is, ir) = len;

            if doPlot && (is == one_src_idx)
                hold on; plot(path(:,1), path(:,2), 'r', 'LineWidth', 0.5);
            end
            
            [len_acc, ~] = EikonalRayPath(x, y, V, from_point, to_point, tmap, [0.1, 20000]);
            RL_accurate(is, ir) = len_acc;
        end
    end
end

function [raylength, raypath] = EikonalRayPath(x, y, v, S, R, tmap, str_options)
    [xx, yy] = meshgrid(x, y);
    [U, V_grad] = gradient(tmap);
    
    path_cell = stream2(xx, yy, -U, -V_grad, R(1), R(2), str_options);
    
    if isempty(path_cell{1})
        raypath = []; raylength = NaN; return;
    end
    
    raypath = path_cell{1};
    
   
    dist_to_src = sqrt((raypath(:,1)-S(1)).^2 + (raypath(:,2)-S(2)).^2);
    cutoff = find(dist_to_src < (x(2)-x(1)), 1, 'first');
    
    if ~isempty(cutoff)
        raypath = [raypath(1:cutoff, :); S(1:2)];
    else
        raypath = [raypath; S(1:2)];
    end
    
   
    diffs = diff(raypath);
    raylength = sum(sqrt(sum(diffs.^2, 2)));
end