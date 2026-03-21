function write_dataset_normalization_metadata(output_root)
    % WRITE_DATASET_NORMALIZATION_METADATA  Scan all comprehensive .mat files and write normalization.json.
    %
    % Reads sos_map and tof_tumor_raw (same fields as save_comprehensive_sample / Python TofDataset).
    % Global min/max over train, validate, and test splits. Python loads this file from data_root.
    %
    % Scope 'all_splits': mild information leakage into val/test for the scaler (documented in docs).

    splits = {'train', 'validate', 'test'};
    minSos = inf;
    maxSos = -inf;
    minTof = inf;
    maxTof = -inf;
    nFiles = 0;

    for si = 1:numel(splits)
        matDir = fullfile(output_root, splits{si}, 'mat');
        if ~isfolder(matDir)
            continue;
        end
        d = dir(fullfile(matDir, '*.mat'));
        for k = 1:numel(d)
            if d(k).isdir
                continue;
            end
            p = fullfile(matDir, d(k).name);
            try
                S = load(p, 'sos_map', 'tof_tumor_raw');
            catch ME
                warning('TOF:NormMetaLoad', 'Skipping %s: %s', p, ME.message);
                continue;
            end
            if isfield(S, 'sos_map')
                v = S.sos_map(:);
                v = v(isfinite(v));
                if ~isempty(v)
                    minSos = min(minSos, min(v));
                    maxSos = max(maxSos, max(v));
                end
            end
            if isfield(S, 'tof_tumor_raw')
                v = S.tof_tumor_raw(:);
                v = v(isfinite(v));
                if ~isempty(v)
                    minTof = min(minTof, min(v));
                    maxTof = max(maxTof, max(v));
                end
            end
            nFiles = nFiles + 1;
        end
    end

    if nFiles == 0
        error('TOF:NoMatFiles', ...
            'No readable .mat files found under %s; cannot write normalization.json.', output_root);
    end

    epsTol = 1e-9;
    if maxSos <= minSos
        minSos = double(minSos) - epsTol;
        maxSos = double(maxSos) + epsTol;
    end
    if maxTof <= minTof
        minTof = double(minTof) - epsTol;
        maxTof = double(maxTof) + epsTol;
    end

    meta = struct( ...
        'version', 1, ...
        'min_sos', minSos, ...
        'max_sos', maxSos, ...
        'min_tof', minTof, ...
        'max_tof', maxTof, ...
        'fields', struct('sos', 'sos_map', 'tof', 'tof_tumor_raw'), ...
        'scope', 'all_splits', ...
        'num_mat_files', nFiles);

    outPath = fullfile(output_root, 'normalization.json');
    fid = fopen(outPath, 'w');
    if fid < 0
        error('TOF:NormMetaWrite', 'Cannot open for write: %s', outPath);
    end
    fprintf(fid, '%s', jsonencode(meta));
    fclose(fid);
    fprintf('Wrote normalization metadata (%d .mat files): %s\n', nFiles, outPath);
end
