files = dir('*.pgm');
% Loop through each file 
for id = 1:length(files)
    % Get the file name 
    [~, f,ext] = fileparts(files(id).name);
    rename = strcat('pos-',string(id),ext) ; 
    movefile(files(id).name, rename); 
end