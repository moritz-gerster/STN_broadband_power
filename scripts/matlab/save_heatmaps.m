%% Load table
path = fullfile('..', '..', 'results','dataframes','localization_powers.xlsx');
heat_path = fullfile('..', '..', 'derivatives', 'heatmaps');
mkdir(heat_path);
T = readtable(path);

% Pool all datasets or save single project separately?
% save_single_projects = 0;

close all;
%
%% Paths
tool_path = '/Users/moritzgerster/Library/CloudStorage/Dropbox/Code/MATLAB/';
lead_path = [tool_path 'leaddbs/'];
atlas_path = [lead_path 'templates/space/MNI152NLin2009bAsym/'];
spm_path = [tool_path 'spm12/'];
wjn_path = [tool_path 'wjn_toolbox-master/'];
addpath(spm_path);
addpath(genpath(wjn_path));
addpath(genpath(lead_path));

%% MNI: SCATTER AND HEATMAP

% flip all lefthemispheric peaks onto the right side (nonlinearly!)
if ~ismember('mni_xr', T.Properties.VariableNames)
    % creating T.mni_xr, T.mni_yr, T.mni_zr
    leftidx = find(T.mni_x(:) < 0);
    allcoords = [T.mni_x T.mni_y T.mni_z];
    T.mni_xr = allcoords(:, 1);
    T.mni_yr = allcoords(:, 2);
    T.mni_zr = allcoords(:, 3);
    newcoords = ea_flip_lr_nonlinear(allcoords(leftidx, :));
    T.mni_xr(leftidx, :) = newcoords(:, 1);
    T.mni_yr(leftidx, :) = newcoords(:, 2);
    T.mni_zr(leftidx, :) = newcoords(:, 3);
    writetable(T, path)
end


%% collect coords and peaks in different categories / frequency bands (bilateral, flipped onto right hemisphere)

info_columns = {'subject', 'project', 'ch_nme', 'ch_hemisphere', ...
                'mni_x', 'mni_y', 'mni_z', 'mni_xr', 'mni_yr', 'mni_zr'};
band_powers = T.Properties.VariableNames;
band_powers = setdiff(band_powers, info_columns);

projects = unique(T.project);

for band_idx = 1:numel(band_powers)
    band_power = band_powers{band_idx};

    % all projects combined
    coord_band = [T.mni_xr T.mni_yr T.mni_zr];
    power_band = T.(band_power);
    save_path = fullfile(heat_path, [band_power '_all' '_heatmap.nii']);

    % Call wjn_heatmap function for the current band
    % Important: lines 59 to 66 in wjn_heatmap need to be commented out
    wjn_heatmap(save_path, coord_band, power_band, [atlas_path 'bb.nii'], [0.7 0.7 0.7]);
    delete(save_path);  % only keep smoothed version to save space
    % if save_single_projects == 1
    %     for proj_idx = 1:numel(projects)
    %         project = projects{proj_idx};
    %
    %         project_rows = strcmp(T.project, project);
    %         T_sub = T(project_rows, :);
    %
    %         coord_band = [T_sub.mni_xr T_sub.mni_yr T_sub.mni_zr];
    %         power_band = T_sub.(band_power);
    %         save_path = fullfile(heat_path, [band_power '_' project '_heatmap.nii']);
    %
    %         % Call wjn_heatmap function for the current band
    %         wjn_heatmap(save_path, coord_band, power_band, [atlas_path 'bb.nii'], [0.7 0.7 0.7]);
    %         delete(save_path);  % only keep smoothed version to save space
    %     end
    % end
end
clear