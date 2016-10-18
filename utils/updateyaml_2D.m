% This script is intended to produce configuration files for the
% VectorMomentum LDDMM code which will be used to warp images based on the
% computed momenta.


%--------------------------------------------------------------------------
% CONFIG::START (change if required)
%--------------------------------------------------------------------------

% set the output directory for the .yaml files
output_conf_dir = '/tmp/';              

% set the path prefix for the folders which will hold the results of
% running the VectorMomentum LDDMM code
output_case_dir = '/tmp/case_';

% set the prefix of the .yaml files
yaml_file_prefix = 'deep_network_';

% set the path to the atlas image
path_to_atlas_image = ...
    '/home/pma/rkwitt/deformation-prediction-code/data/images/atlas.mhd';

% set the prefix for the case-specific .mhd momenta files
path_to_mhd_files = '/tmp/m_%d.mhd';

%--------------------------------------------------------------------------
% CONFIG::END
%--------------------------------------------------------------------------

fid = fopen('parsedconfig_2D.yaml', 'r');
for i = 1:50
    
    % construct the filename of the .yaml files
    outputName = fullfile(output_conf_dir, ...
        ['deep_network_', num2str(i), '.yaml']);
    
    % open the new .yaml file
    fid2 = fopen(outputName, 'w');
    
    % create the case-specific directory for the output of the
    % VectorMomentum LDDMM code
    out_dir_prefix = sprintf('%s%d/', output_case_dir, i);
    
    tline = fgetl(fid);
    while ischar(tline)
        
        disp(tline)
        tline = strrep(tline, ...
            '/directory_to_repo/data/images/atlas.mhd', ...
            path_to_atlas_image);
 
        tline = strrep(tline, ...
            '/directory_to_generated_momentums/m.mhd', ...
            sprintf(path_to_mhd_files, i));
        
        tline = strrep(tline, ...
            'output_prefix', ...
            out_dir_prefix);
        
        fprintf(fid2, '%s\n', tline);
        tline = fgetl(fid);

    end
    
    sta2 = fclose(fid2);
    type(outputName);
    frewind(fid)

end
sta = fclose(fid);
