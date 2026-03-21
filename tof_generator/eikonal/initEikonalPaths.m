function hasMex = initEikonalPaths()
% INITEIKONALPATHS  Add tof_generator eikonal tree (and deps) to the MATLAB path.
%   Call from the client before parfor; use pctPathEvalString + pctRunOnAll(@() eval(cmd))
%   in run drivers so workers get paths even if the pool had a stale path.
%
%   Bootstrap (once on the client): addpath(fullfile(tof_generator_root,'eikonal'))
%   so this file is visible, then hasMex = initEikonalPaths();
%
%   Set env TOF_FORCE_MSFM_MATLAB to 1, true, or yes to skip MEX inside msfm.m.

    here = fileparts(mfilename('fullpath'));
    genRoot = fileparts(here);
    addpath(genRoot);
    addpath(genpath(here));
    matSrc = fullfile(genRoot, '..', 'matlab_src');
    if exist(matSrc, 'dir')
        addpath(matSrc);
    end
    hasMex = (exist('msfm2d', 'file') == 3);
end
