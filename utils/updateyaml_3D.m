fid = fopen('parsedconfig_3D.yaml', 'r');

for i = 1:50
    outputName = ['deep_network_', num2str(i), '.yaml'];
    fid2 = fopen(outputName, 'w');
    tline = fgetl(fid)
    while ischar(tline)
        disp(tline)
        tline = strrep(tline, 'outputPrefixes', num2str(i));
        tline = strrep(tline, 'm0.mhd', strcat('OAS3D_', num2str(i), '_deep_new_mscalar.mhd'));
        fprintf(fid2, '%s\n', tline);
        tline = fgetl(fid);
        % %% read tmp.yaml file
    end
    sta2 = fclose(fid2);
    type(outputName);
    frewind(fid)
end
sta = fclose(fid);
