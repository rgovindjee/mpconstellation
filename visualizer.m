clear;
clc;
close all;

% Using matlab to 3D plot because it is way easier
pos = readmatrix("trajectory.csv");
X = pos(:,1);
Y = pos(:,2);
Z = pos(:,3);

% Plotting
figure;
earth_sphere('m');
hold on;
plot3(X, Y, Z, 'r', 'LineWidth', 1);
view(103.8171, 8.7652)