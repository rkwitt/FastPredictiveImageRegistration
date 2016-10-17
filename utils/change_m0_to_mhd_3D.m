load '3D_output.mat'
idx = find(isnan(test_m0_recon_all));
test_m0_recon_all(idx) = 0;
for i = 1:50
    for j = 1:1
        imsize = [128,128,128];
        imspcq = [1,1,1];
        imorig = [0,0,0];
        imgorient = eye(3);
        scalarImg_output = VectorImageType(imsize,imorig,imspcq,imgorient);
        scalarImg_output.datax = permute(squeeze(test_m0_recon_all(i, 1, :, :, :)), [3 2 1]);
        scalarImg_output.datay = permute(squeeze(test_m0_recon_all(i, 2, :, :, :)), [3 2 1]);
        scalarImg_output.dataz = permute(squeeze(test_m0_recon_all(i, 3, :, :, :)), [3 2 1]);
    end
    write_mhd([ 'temp/OAS3D_' num2str(i) '_deep_new_mscalar.mhd'], scalarImg_output);
end