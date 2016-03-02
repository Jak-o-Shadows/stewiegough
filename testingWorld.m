close all
clear all
clc
%%



myworld = vrworld('world.wrl', 'new');                      % Get handle for the VR model
open(myworld);                                              % Open the VR model

vh = view(myworld);