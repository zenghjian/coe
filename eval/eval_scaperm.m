%% COE eval code
addpath('geoErrors');

% load corres and geod data
folder_data_in = "";
% load embeddings
folder_result_in = "";
task_name = 'COE';

load(fullfile(folder_result_in, task_name));

set_basis = basis;

num_of_data = 20;

vts_5k = {}; % all ground truth corr
M = {}; % all geodesic distance matrices
for i = 1:num_of_data
    vts = load(fullfile(folder_data_in, ['corres/mesh0',num2str(i-1+52,'%02d'),'.vts']));
    vts_5k{1,i} = vts;

    geod_utri = load(fullfile(folder_data_in, ['geod/mesh0',num2str(i-1+52,'%02d'), '.mat']));
    geod = geod_utri.geod + geod_utri.geod';
    M{1,i} = geod; 
end

all_geoErrors = [];

disp(size(set_basis));

for src = 1:num_of_data
    for tar = 1:num_of_data
        
        disp(['src: ', num2str(src+51), ', tar: ', num2str(tar+51)]);
        phiS= [squeeze(set_basis{src})];
        phiT =[squeeze(set_basis{tar})];      
    
        [idx,distance] = knnsearch(phiT,phiS);  

        matches = idx;
        gtsrc = vts_5k{src};
        gttar = vts_5k{tar};
        geodtar = M{tar};

        geoErrors = calc_geoErrors(matches, gtsrc, gttar, geodtar);

        disp('geodesic error:');
        disp(mean(geoErrors,1));

        all_geoErrors = [all_geoErrors; geoErrors];
    end
end

mean_geoError = mean(all_geoErrors, 1);

disp(task_name);
disp('Mean geodesic error dcb:');
disp(mean_geoError);

