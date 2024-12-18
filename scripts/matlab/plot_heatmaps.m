% Paths
tool_path = '/Users/moritzgerster/Library/CloudStorage/Dropbox/Code/MATLAB/';
lead_path = [tool_path 'leaddbs/'];
% atlas_path = [lead_path 'templates/space/MNI152NLin2009bAsym/'];
spm_path = [tool_path 'spm12/'];
wjn_path = [tool_path 'wjn_toolbox-master/'];
addpath(spm_path);
addpath(genpath(wjn_path));
addpath(genpath(lead_path));

load_path_heatmaps = fullfile('..', '..', 'derivatives', 'heatmaps');
save_dir = fullfile('..', '..', 'results', 'plots', 'heatmaps');
mkdir(save_dir);

plot_only_posterolateral = 1;

%% Load table for electrode plotting
table_path = fullfile('..', '..', 'results','dataframes','localization_powers.xlsx');
T = readtable(table_path);
if ~ismember('mni_xr', T.Properties.VariableNames)
    error('Run save_heatmaps.m first to flip coordinates')
end
mni_coords = [T.mni_xr T.mni_yr T.mni_zr];
v = ones(size(mni_coords, 1), 1);  % plot electrodes
radius = .15;
%% heatmap 3D

% general MNI figure showing just STNs
fig = ea_mnifigure('DISTAL Minimal (Ewert 2017)');

% ea_cortexselect()

figure_width = 10; % Specify the width of the figure
figure_height = 10; % Specify the height of the figure
set(gcf, 'Units', 'inches', 'Position', [10, 10, figure_width, figure_height]);

% remove MRI since not helpful
% togglestates.xyztransparencies = [0 0 0];
% togglestates.refreshcuts = 0;
% togglestates.refreshview = 0;
% togglestates.xyztoggles = [0 0 0];
% togglestates.cutview = 'xcut';
% ea_anatomyslices(fig, togglestates)

hold on;
ea_keepatlaslabels('off');
% atlassurfs = ea_keepatlaslabels('STN_right');
atlassurfs = ea_keepatlaslabels('STN_motor_right', 'STN_associative_right', 'STN_limbic_right');
ea_keepatlaslabels('off');
stn_motor = atlassurfs{1};
stn_asso = atlassurfs{2};
stn_limbic = atlassurfs{3};
% color_motor = [1 0 0];
color_motor = [77,175,74]/255;
color_asso = [1 0.5 0];
% color_asso = [55,126,184]/255;
color_limbic = [0.9763 0.9831 0.0538];
% color_limbic = [152,78,163]/255;
% alpha = 0.3;
alpha = 0.4;
STN_MOTOR = patch('faces', stn_motor.fv.faces, ...
                  'vertices', stn_motor.fv.vertices, ...
                  'FaceLighting','gouraud',...
                  'EdgeLighting','gouraud', ...
                  'SpecularExponent',3,...
                  'SpecularStrength',0.3,...
                  'DiffuseStrength',0.4,...
                  'facecolor', color_motor, ...
                  'edgecolor', 'none', ...
                  'facealpha', alpha);

STN_ASSO = patch('faces', stn_asso.fv.faces, ...
    'vertices', stn_asso.fv.vertices, ...
            'FaceLighting','gouraud',...
            'SpecularExponent',3,...
    'SpecularStrength',0.3,...
    'DiffuseStrength',0.4,...
                 'facecolor', color_asso, ...
                 'edgecolor', 'none', ...
                 'facealpha', alpha);
STN_LIMBIC = patch('faces', stn_limbic.fv.faces, ...
    'vertices', stn_limbic.fv.vertices, ...
            'FaceLighting','gouraud',...
            'SpecularExponent',3,...
    'SpecularStrength',0.3,...
    'DiffuseStrength',0.4,...
                   'facecolor', color_limbic, ...
                   'edgecolor', 'none', ...
                   'facealpha', alpha);

hold on;
heatmaps = dir(fullfile(load_path_heatmaps, '*.nii'));
heatmaps = {heatmaps.name};


% Sweetspot
sweetspot_coord_right = [12.50 -12.72 -5.38]; % sweet spot right, Dembek 2019
% ea_spherical_roi([lead_path '1mm_sphere.nii'], sweetspot_coord_right,1,1,[lead_path 'templates/space/MNI152NLin2009bAsym/' 'bb.nii']);
color_sweetspot = [0.3020    0.7490    0.9294];
radius_sweetspot = 1.5;
wjn_plot_colored_spheres(sweetspot_coord_right, 1, radius_sweetspot, color_sweetspot); hold on; %

% % Plot electrode positions
% [s, v, ov] = wjn_plot_colored_spheres(mni_coords, v, radius, 'k');
% hold on;

% zoom(7)
% campos_orig = campos;
% hA = gca;

for heat_idx = 1:numel(heatmaps)
    heat_name = heatmaps{heat_idx};

    [p, f, e] = fileparts(heat_name);
    heat_name = fullfile(p, f);

%     if ~contains(heat_name, 'beta_low')
%         continue
%     end
%     if ~contains(heat_name, 'normalized')
%         continue
%     end

    if contains(heat_name, 'fm_powers')
        kind = 'periodic';
%         continue
    elseif contains(heat_name, 'normalized')
        kind = 'normalized';
%         continue
    else
        kind = 'absolute';
%         continue
    end

    % load map
    path_heatmap = fullfile(load_path_heatmaps, heat_name);
    nii = ea_load_nii(path_heatmap);
    fv = ea_nii2fv(nii);

%     if contains(heat_name, 'on')
%         color = [.5 .5 .5];
% %         alpha = 0.5;
%         edgecolor = color;
    edgecolor = [0.5    0.2    0.9294];
    if contains(heat_name, 'beta_low')
        color = [0.9882    0.5529    0.3843];
    elseif contains(heat_name, 'beta_high')
        color = [0.9059    0.5412    0.7647];
    elseif contains(heat_name, 'theta_alpha')
        color = [0.5529, 0.6275, 0.7961];
    elseif contains(heat_name, 'alpha')
        color = [0.6510, 0.8471, 0.3294];
    elseif contains(heat_name, 'beta_')
        color = [0.94705, 0.54705, 0.5745];
    elseif contains(heat_name, 'gamma_low')
        color = [1, 0.85098, 0.18431];
    elseif contains(heat_name, 'gamma_mid')
        color = [0.8980, 0.7686, 0.18431];
    elseif contains(heat_name, 'gamma_')
        color = [0.8980, 0.7686, 0.18431];
    else
        color = [0.3020    0.7490    0.9294];
    end

%     if contains(heat_name, 'off')
%         alpha = 0.5;
%     elseif contains(heat_name, 'on')
%         alpha = 0.4;
%     end
    alpha = 0.5;

%     color = [0.3020    0.7490    0.9294];
    patch_handle = patch('faces', fv.faces, 'vertices', fv.vertices, ...
                         'facecolor', color, ...
                         'edgecolor', edgecolor, ...
                         'facealpha', alpha, 'edgealpha', alpha);

    % ea_spherical_roi([path '1mm_sphere.nii'], [12.50 -12.72 -5.38],1,1,[lead_path 'bb.nii']); % sweet spot right, Dembek 2019

%     % View settings
% %     camva(10.3717877881908);
% %     camup([0 0 1]);
%     campos([1.6 432 170]);
%     camtarget([11 -7 -5]);
%     camzoom(12)
%
%
%     view(-180, 0);
%     camtarget([11 -7 -5]);
%     campos(campos_orig);
%     camzoom(5)

    save_dir_kind = fullfile(save_dir, kind);
    mkdir(save_dir_kind);

    if plot_only_posterolateral == 0
        camva(9);
        camup([0 0 1]);
        campos([2  430  170]);
        camtarget([11 -7 -5]);
        camzoom(5)
        save_path = fullfile(save_dir_kind, [heat_name, '_cor']);
        print(gcf, save_path , '-dtiff', '-r600', '-image');
    end

%     camzoom(1/5)
    if plot_only_posterolateral == 0
        camva(33);
        camup([0 0 1]);
        campos([-8.5336  -13.1026   -5.8217]);
        camtarget([18 -13 -6]);
        save_path = fullfile(save_dir_kind, [heat_name, '_sag']);
        print(gcf, save_path, '-dtiff', '-r600', '-image');
    end
%
%     view(90, 90)
%     camva(90);
%     camup([-1 0 0]);
%     campos([0 0 1.9209]);
%     camtarget([12 -12 0]);
%     camzoom(20)
%     save_path = fullfile(save_dir, [heat_name, '_ax']);
%     print(gcf, save_path, '-dtiff', '-r600', '-image');

    camva(38.6);
    camup([0 0 1]);
    campos([0.9 -21.7 -.4]);
    camtarget([13.9 -12.6 -6.3]);

    % plot coordinate system for orientation
    if contains(heat_name, 'beta_low_normalized_abs_max_log_off')
        x = 0;
        y = -14.9;
        z = 4;
        length = 1.25;
        c1 = plot3([x - length, x], [y, y], [z, z], 'w', 'LineWidth',1.5);  % L-R
        c2 = plot3([x, x], [y - length, y], [z, z], 'w', 'LineWidth',1.5);  % Frontal-Posterior
        c3 = plot3([x, x], [y, y], [z, z + length], 'w', 'LineWidth',1.5);  % Superior-Inferior

        % Plot electrode positions
        elecs = wjn_plot_colored_spheres(mni_coords, v, radius, 'k');
        hold on;
    else
        if exist('c1', 'var') && isgraphics(c1)
            delete(c1);
            delete(c2);
            delete(c3);
            delete(elecs);
        end
%         delete(c1)
%         delete(c2)
%         delete(c3)
    end

    save_path = fullfile(save_dir_kind, [heat_name, '_posterolateral']);
    print(gcf, save_path, '-dpng', '-r150', '-image');
    delete(patch_handle);

end
close all
