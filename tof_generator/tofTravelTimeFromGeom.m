function [t_obs, tmap_for_plot, tmap_all, v_for_plot] = tofTravelTimeFromGeom(geom, V)
%TOFTRAVELTIMEFROMGEOM  Same physics as ToF.createTravelTime without constructing ToF.
%
%   geom fields (from make_tof_geom): x, y, z, R, xs_sources, ys_sources,
%   number_of_sources, number_of_receivers, verbose, one_source, m, n.

    t_obs = zeros(geom.number_of_sources, geom.number_of_receivers);
    tmap_all = zeros(geom.number_of_sources, geom.n, geom.m);
    v_for_plot = [];

    if geom.verbose
        fprintf('Running Eikonal for %d sources...\n', geom.number_of_sources);
    end

    for is = 1:geom.number_of_sources
        nr = geom.number_of_receivers;
        S_current = [ones(nr, 1) * geom.xs_sources(is), ...
                     ones(nr, 1) * geom.ys_sources(is), ...
                     ones(nr, 1)];
        [tmap, t] = eikonal_traveltime(geom.x, geom.y, geom.z, V, S_current, geom.R);

        t_obs(is, :) = t(:);
        tmap_all(is, :, :) = squeeze(tmap);

        if is == geom.one_source
            tmap_for_plot = tmap;
            tflat = tmap(:);
            v_for_plot = min(tflat):(max(tflat) - min(tflat)) / 100:max(tflat);
        end
    end
end
