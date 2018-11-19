% Add path
addpath(genpath('GCmex1.5'));
im = im2double( imread('cat.jpg') );

org_im = im;
imr= im(:,:,1);
imb=im(:,:,2);
img=im(:,:,3);
H = size(im, 1); W = size(im, 2); K = 3;

% Load the mask
load cat_poly
inbox = poly2mask(poly(:,1), poly(:,2), size(im, 1), size(im,2));
N = H*W;
% 1) Fit Gaussian mixture model for foreground regions
FG(:,1)= imr(inbox);
FG(:,2)= imb(inbox);
FG(:,3)= img(inbox);
% 2) Fit Gaussian mixture model for background regions
BG(:,1)= imr(~inbox);
BG(:,2)= imb(~inbox);
BG(:,3)= img(~inbox);
FG_GMM = fitgmdist(FG, 17);
BG_GMM = fitgmdist(BG, 17);
% 3) Prepare the data cost
% - data [Height x Width x 2] 
% - data(:,:,1) the cost of assigning pixels to label 1
% - data(:,:,2) the cost of assigning pixels to label 2
Pfg = -log(pdf(FG_GMM, reshape(im,[N,3])));
Pbg = -log(pdf(BG_GMM, reshape(im,[N,3])));

Pfg1 = reshape(Pfg, [H,W]);
Pbg1 = reshape(Pbg, [H,W]);
data(:,:,1) = Pfg1;
data(:,:,2) = Pbg1;
imshow(data(:,:,1));
figure();
imagesc(data(:,:,1));
figure();
imshow(data(:,:,2));
figure();
imagesc(data(:,:,2));
figure();
% 4) Prepare smoothness cost
% - smoothcost [2 x 2]
% - smoothcost(1, 2) = smoothcost(2,1) => the cost if neighboring pixels do not have the same label
smoothcost = [0 1; 1 0];
% 5) Prepare contrast sensitive cost
% - vC: [Height x Width]: vC = 2-exp(-gy/(2*sigma)); 
% - hC: [Height x Width]: hC = 2-exp(-gx/(2*sigma));
im1 = im;
[gxr, gyr] = gradient(im1(:,:,1));
[gxg, gyg] = gradient(im1(:,:,2));
[gxb, gyb] = gradient(im1(:,:,3));
gx = sqrt((gxr.^2) + (gxg.^2) + (gxb.^2));
gy = sqrt((gyr.^2) + (gyg.^2) + (gyb.^2));
m1 = mean(mean(im1));
sigma = 0.005;
vC = 2-exp(-gy/(2*sigma));
hC = 2-exp(-gx/(2*sigma));
% 6) Solve the labeling using graph cut
% - Check the function GraphCut
[gch] = GraphCut('open', data, smoothcost, vC, hC); 
[gch, labels] = GraphCut('expand', gch);
%[gch,labels] = GraphCut('swap', gch);
for i = 1:size(labels,1)
    for j = 1:size(labels,2)
        if labels(i,j)==1
            im(i,j,1)=0;
            im(i,j,2) = 0;
            im(i,j,3) = 255;
        end 
    end
end
imshow(im);
figure();
img1 = bsxfun(@times, im, cast(~labels,'like', im));
imshow(img1);
figure();
imagesc(labels);
% 7) Visualize the results