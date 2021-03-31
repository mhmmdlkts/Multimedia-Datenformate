clear; close;
ref = aviFile('test.avif');
imshow(ref)

ref2 = imread('test.png');
imshow(ref2)

err = immse(ref2, ref);
[peaksnr, snr] = psnr(ref2, ref);
[ssimval,ssimmap] = ssim(ref2,ref);
% imshow(ssimmap,[])
ref = ref(:,:);
ref2 = ref2(:,:);
imshow(ref);
RGBNoisy = imnoise(ref,"salt & pepper");
score = multissim(RGBNoisy,ref);
score = squeeze(score);
  
fprintf('\n The Peak-SNR value is %0.4f', peaksnr);
fprintf('\n The SNR value is %0.4f', snr);
fprintf('\n The mean-squared error is %0.4f', err);
fprintf('\n The Local SSIM Map with Global SSIM Value is %0.4f', num2str(ssimval));
fprintf('\n The Multiscale structural similarity Value is %0.4f', score);

fprintf('\n');