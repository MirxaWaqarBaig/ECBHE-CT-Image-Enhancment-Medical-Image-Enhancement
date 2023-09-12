clear all
clc
%%
image = dicomread('Subject_1.dcm');
enh_image = sn_ecbhe_lcgi_dwt(image);