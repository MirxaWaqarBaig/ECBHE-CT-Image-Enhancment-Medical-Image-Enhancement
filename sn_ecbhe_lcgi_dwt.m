function [output_image] = sn_ecbhe_lcgi_dwt(input_img)
% Implementation of "An Improved and Efficient Approach for Enhancing the 
% Precision of Diagnostic CT Images. 
% SN Computer Science." - 
% Date of Publication: 21 December 2022 -------------
% Implementation by Waqar Mirza
% Input: input_img % I used a high dynamic medical image in dicom format 
% Output: output_image % The enhanced output image of the whole technique 
% input_img = double(dicomread('Subject_1.dcm'));
input_img = double(input_img);
img=linspace(min(input_img(:)),max(input_img(:)),256);
img=uint8(arrayfun(@(x) find(abs(img(:)-x)==min(abs(img(:)-x))),input_img));
%% equation 1 and equation 2
L=256;
x=[0:1:L-1];
[w,l]=size(img); 
len=w*l;
y=reshape(img,len,1);   
xpdf=hist(y,[0:L-1]);
exposure=sum(xpdf.*x)/sum(xpdf)/(L);
aNorm=(1-exposure);
Xm=round(L*aNorm);
%% Clipping Process
Tc=mean(xpdf);  % mean pixels for gray levels
Tc=round(Tc);
Ihist=zeros(1,256);   % intermediate histogram for clipping
 for i=1:256
     if xpdf(i)>Tc
     Ihist(i)=Tc;
     elseif xpdf(i)==0
          Ihist(i)=xpdf(i);
     else
         Ihist(i)=xpdf(i);
     end     
 end
%% 
ecbh_1 =zeros(size(img));          
C_L=zeros(1,Xm+1);
C_U=zeros(1,(256-(Xm+1)));
n_L=sum(Ihist(1:Xm+1));
n_U=sum(Ihist(Xm+2:256));
P_L=Ihist(1:Xm+1)/n_L;
P_U=Ihist(Xm+2:256)/n_U;
C_L(1)=P_L(1);
for r=2:length(P_L)
    C_L(r)=P_L(r)+C_L(r-1);
end
C_U(1)=P_U(1);
for r=2:(length(P_U))
    C_U(r)=P_U(r)+C_U(r-1);
end
for r=1:w                       
    for s=1:l
        if img(r,s)<(Xm+1)
            f=Xm*C_L(img(r,s)+1);
            ecbh_1(r,s)=round(f);
        else
            f=(Xm+1)+(255-Xm)*C_U((img(r,s)-(Xm+1))+1);
            ecbh_1(r,s)=round(f);
        end
    end
end

%% Divide the image into non-overlapping blocks
% Set the block size to 50
block_size = 50;
[h, wi] = size(ecbh_1);
num_blocks_h = floor(h / block_size);
num_blocks_w = floor(wi / block_size);

block_hist = zeros(256, num_blocks_h * num_blocks_w);
% r_f = fraction of the local pixel to total pixel of ecbh_1
r_f = zeros(1, num_blocks_h * num_blocks_w);
block_entropy = zeros(1, num_blocks_h * num_blocks_w);

for i = 1:num_blocks_h
for j = 1:num_blocks_w
block = ecbh_1((i-1)*block_size+1:i*block_size, (j-1)*block_size+1:j*block_size);
block_hist(:, (i-1)*num_blocks_w+j) = hist(block(:), 0:255);
r_f((i-1)*num_blocks_w+j) = sum(sum(block)) / (h * wi);
block_entropy((i-1)*num_blocks_w+j) = entropy(block_hist(:, (i-1)*num_blocks_w+j));
end
end
ecbh_1_entropy = entropy(hist(ecbh_1(:), 0:255));

%%
% Compute average block entropy
avg_block_entropy = mean(block_entropy);
% Compute block weights
block_weights = block_entropy / avg_block_entropy;
% Compute global enhanced image
global_enhanced_image = zeros(h, wi);
for i = 1:num_blocks_h
    for j = 1:num_blocks_w
        block = ecbh_1((i-1)*block_size+1:i*block_size, (j-1)*block_size+1:j*block_size);
        global_enhanced_image((i-1)*block_size+1:i*block_size, (j-1)*block_size+1:j*block_size) = block .* block_weights((i-1)*num_blocks_w+j);
    end
end
% Apply Gaussian filter
gaussian_filter = fspecial('gaussian', [3 3], 0.5);
global_enhanced_image = imfilter(global_enhanced_image, gaussian_filter);
% Normalize the image
global_enhanced_image = (global_enhanced_image - min(global_enhanced_image(:))) / (max(global_enhanced_image(:)) - min(global_enhanced_image(:))) * 255;
% Convert the image to uint8 format
global_enhanced_image = uint8(global_enhanced_image);
%% DWT 
% Apply DWT to ecbh_1
[LL_ecbh1, LH_ecbh1, HL_ecbh1, HH_ecbh1] = dwt2(ecbh_1, 'haar');

% Apply DWT to global_enhanced_image
[LL_global, LH_global, HL_global, HH_global] = dwt2(global_enhanced_image, 'haar');
%% Apply the bilateral filter on global enhanced image and ecbh_image
% Set the fast bilateral filter parameters
sigma_s = 2; % spatial standard deviation
sigma_r = 0.1; % range standard deviation
n= 2;
% Apply fast bilateral filter to high frequency components of global
% enhanced image
LH_global_filt = bilateral_fast(LH_global,sigma_s, sigma_r,n);
HL_global_filt = bilateral_fast(HL_global,sigma_s, sigma_r,n);
HH_global_filt = bilateral_fast(HH_global,sigma_s, sigma_r,n);
% Apply fast bilateral filter to high frequency components of ecbh
% enhanced image
LH_ecbh1_filt = bilateral_fast(LH_ecbh1,sigma_s, sigma_r,n);
HL_ecbh1_filt = bilateral_fast(HL_ecbh1,sigma_s, sigma_r,n);
HH_ecbh1_filt = bilateral_fast(HH_ecbh1,sigma_s, sigma_r,n);
%% Fusion of Low frequency components
wei = 0.2;
LL_o = (1-wei)*LL_global+wei*LL_ecbh1;
%%
% Extract the high-frequency subbands (HH) from the DWT coefficients
LH_ecbh1_max = max(abs(LH_ecbh1(:)));
LH_global_max = max(abs(LH_global(:)));
% Take the element-wise maximum of the high-frequency subbands
LH_merged = max(abs(LH_ecbh1), abs(LH_global));
%%
% Extract the high-frequency subbands (HH) from the DWT coefficients
HL_ecbh1_max = max(abs(HL_ecbh1(:)));
HL_global_max = max(abs(HL_global(:)));
% Take the element-wise maximum of the high-frequency subbands
HL_merged = max(abs(HL_ecbh1), abs(HL_global));
%%
% Extract the high-frequency subbands (HH) from the DWT coefficients
HH_ecbh1_max = max(abs(HH_ecbh1(:)));
HH_global_max = max(abs(HH_global(:)));
% Take the element-wise maximum of the high-frequency subbands
HH_merged = max(abs(HH_ecbh1), abs(HH_global));
%% Inverse DWT
output_image = idwt2(LL_o ,LH_merged,HL_merged,HH_merged,'haar');
output_image = (output_image - min(output_image(:))) ./ (max(output_image(:)) - min(output_image(:)));
% Scale the image to the range [0, 255]
output_image = output_image * 255;
% Convert the image to uint8 format
output_image = uint8(output_image);
imshow(output_image,[]);
end
%%%%%
%%%%%
%%%%%
%%%%%
%% Functions
%% Bilateral Filter 
function Y = bilateral_fast( X, sigma_s, sigma_r, n )
% Fast bilateral filter
%
% Y = bilateral_fast( X, sigma_s, sigma_r )
%
% X - input image
% Y - blurred image
% sigma_s - standard deviation of the spatial Gaussian kernel
% sigma_r - standard deviation of the range Gaussian kernel
% n - number of layers to use (the higher the number, the more accurate is
% the filter but it takes more time).
%
% Based on the algoritm from:
%
% Durand, F., & Dorsey, J. (2002). Fast bilateral filtering for the
% display of high-dynamic-range images. ACM Transactions on Graphics, 21(3). doi:10.1145/566654.566574
%
% (c) 2012 Rafal Mantiuk

if( size( X, 3 ) ~= 1 )
    error( 'bilateral_fast can process only grayscale images' );
end

if( ~exist( 'n', 'var' ) )
    n=6; % number of layers
end

min_x = min(X(:)); %prctile( X(:), 1 );
max_x = max(X(:)); %prctile( X(:), 99 );

r = linspace( min_x, max_x, n );

L = zeros( n, numel( X ) );

for i=1:n
    D = exp(-(X - r(i)).^2/(2*sigma_r^2));
    K = blur_gaussian( D, sigma_s );
    Ls = blur_gaussian( X.*D, sigma_s );    
    L(i,:) = Ls(:)./K(:);
end

% interpolate
ind_r = clamp((X(:)-min_x)/(max_x-min_x)*(n-1)+1, 1, n);
ind_down = floor(ind_r);
ind_up = ceil(ind_r);
ind_fix = (0:n:((numel(X)-1)*n))';
ind_up = ind_up + ind_fix;
ind_down = ind_down + ind_fix;
ratio = mod( ind_r, 1 );

Y = zeros( size(X) );
Y(:) = L(ind_up).*ratio + L(ind_down).*(1-ratio);

end


function Y = blur_gaussian( X, sigma )
% Low-pass filter image using the Gaussian filter
% 
% Y = blur_gaussian( X, sigma )
%  

ksize = ceil(sigma*6);
h = fspecial( 'gaussian', ksize, sigma );
Y = imfilter( X, h, 'replicate' );

end
function Y = clamp( X, min, max )
% Y = clamp( X, min, max )
% 
% Restrict values of 'X' to be within the range from 'min' to 'max'.
%
% (c) 2012 Rafal Mantiuk

  Y = X;
  Y(X<min) = min;
  Y(X>max) = max;
end