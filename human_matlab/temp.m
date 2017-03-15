fid=fopen('deploy_1.prototxt');
flag = false; bottom_name = containers.Map();

fid2 = fopen('deploy.prototxt', 'w');
while 1
    tline = fgetl(fid);
    if ~ischar(tline), break, end
    if strcmp('  type: "BN"', tline)
        flag = true;
    end;
    if length(tline) > 11 && strcmp(tline(1:11), '  bottom: "')
        if bottom_name.isKey(tline(12:end-1));
            tline = ['  bottom: "', tline(12:end-1), '/bn"'];
        end;
    end;
    if length(tline) > 8 && strcmp(tline(1:8), '  top: "')
        if flag
            bottom_name(tline(9:end-1)) = 0;
            flag = false;
        end;
        if bottom_name.isKey(tline(9:end-1))
            tline = ['  top: "', tline(9:end-1), '/bn"'];
        end;
    end;
    
    fprintf(fid2, '%s\n', tline);
end
fclose(fid);
fclose(fid2);