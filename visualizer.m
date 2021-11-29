function visualizer(filename)
clc;
close all;

% Input 'all' plots all files that match trajectory*
% Otherwise, plot a single trajectory with the input filename
filenames = {};
if strcmp(filename, 'all')
    files = dir;
    for i = 1:length(files)
        if startsWith(files(i).name, 'trajectory')
            filenames = [filenames{:} {files(i).name}];
        end
    end
else
    filenames = {filename};
end

% Assemble trajector(ies) for plotting
X = {};
Y = {};
Z = {};
for i = 1:length(filenames)
    pos = readmatrix(filenames{i});
    X{i} = pos(:,1);
    Y{i} = pos(:,2);
    Z{i} = pos(:,3);
end

% Plotting
figure;
earth_sphere('m');
hold on;
for i = 1:length(filenames)
    plot3(X{i}, Y{i}, Z{i}, 'r', 'LineWidth', 1);
end
view(103.8171, 8.7652)
end