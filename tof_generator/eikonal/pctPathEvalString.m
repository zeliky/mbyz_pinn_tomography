function cmd = pctPathEvalString(genRoot)
% PCTPATHEVALSTRING  MATLAB expression to add tof_generator paths (for pctRunOnAll(@() eval(cmd))).
    genRoot = char(genRoot);
    esc = @(t) strrep(char(t), '''', '''''');
    eik = char(fullfile(genRoot, 'eikonal'));
    cmd = sprintf('addpath(''%s''); addpath(genpath(''%s''));', esc(genRoot), esc(eik));
    ms = char(fullfile(genRoot, '..', 'matlab_src'));
    if exist(ms, 'dir')
        cmd = [cmd sprintf('addpath(''%s'');', esc(ms))];
    end
end
