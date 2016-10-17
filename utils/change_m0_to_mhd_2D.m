% This script reads (predicted) momenta from the generated .mat file and
% creates for each prediction a .mhd + .raw image file that will be later
% used for running the VectorMomentum LDDDM code.
%
%
% The script loads the momenta file and converts all N initial momenta to 
% .mhd +.raw 128x128 image files (spacing = [1,1,1], origin = [0,0,0] and 
% orientation = eye(3)). The N files are written to
%
%   OUTPUT_DIR/OUTPUT_PREFIX_1.mhd (+.raw file)
%   OUTPUT_DIR/OUTPUT_PREFIX_1.mhd (+.raw file)
%   ...
%   OUTPUT_DIR/OUTPUT_PREFIX_N.mhd (+.raw file)


%--------------------------------------------------------------------------
% CONFIG::START (change if required)
%--------------------------------------------------------------------------

% set the path to the momenta .mat file - the momenta file is expected to
% have a field named .test_m0_recon_all
momenta_mat_file = '~/2D_output.mat';

% set output directory for the .mhd + .raw files
output_dir = '/tmp/';

% set the output prefix
output_prefix = 'm';

%--------------------------------------------------------------------------
% CONFIG::END
%--------------------------------------------------------------------------

tmp = load(momenta_mat_file);
test_m0_recon_all = tmp.test_m0_recon_all;

idx = isnan(test_m0_recon_all);
test_m0_recon_all(idx) = 0;

for i = 1:size(test_m0_recon_all,1)
    
    imsize = [128,128,1];
    imspcq = [1,1,1];
    imorig = [0,0,0];
    imgorient = eye(3);
    scalarImg_output = VectorImageType(imsize,imorig,imspcq,imgorient);
    scalarImg_output.datax = squeeze(test_m0_recon_all(i, 1, :, :))';
    scalarImg_output.datay = squeeze(test_m0_recon_all(i, 2, :, :))';
    
    output_image = fullfile(output_dir, sprintf('%s_%d.mhd', ...
        output_prefix, i));
    
    fprintf('Write image file %s\n', output_image); 
    write_mhd(output_image, scalarImg_output);
end
   