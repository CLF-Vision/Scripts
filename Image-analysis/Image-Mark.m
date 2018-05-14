folder = 'path/to/load/image';
out_folder = 'output/';

filepaths = [];
filepaths = [filepaths; dir(fullfile(folder, '*.jpg'))];
filepaths = [filepaths; dir(fullfile(folder, '*.bmp'))];
filepaths = [filepaths; dir(fullfile(folder, '*.png'))];

im = imread(fullfile(folder,filepaths(1).name));
imshow(im);
hBox = imrect;
roiPosition = wait(hBox);

RGB = insertShape(im,'Rectangle',roiPosition,'LineWidth',5);
imwrite(RGB, fullfile(out_folder,'mask.png'));

for ii = 1 : length(filepaths)
    image = imread(fullfile(folder, filepaths(ii).name));
    crop = imcrop(image, roiPosition);
    imwrite(crop, fullfile(out_folder,filepaths(ii).name));
end
