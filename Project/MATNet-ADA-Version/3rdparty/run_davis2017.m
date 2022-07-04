clc

save_dir = '/home2/rodosingh/AssignmentsProjects/CVProject/MATNet/data/DAVIS2017/Annotations_edge/480p';

src_dir = '/home2/rodosingh/AssignmentsProjects/CVProject/MATNet/data/DAVIS2017/Annotations_unsupervised/480p';

seqs = dir(src_dir);
seqs = {seqs.name};
seqs = seqs(3:end);
disp("Going into For LOOP!!!")
for i = 1 : length(seqs)
    seq_path = [src_dir, '/', seqs{i}];
    images = dir([seq_path, '/*.png']);
    images = {images.name};
    %disp("Going into 2nd For LOOP!!!")
    
    for j = 1 : length(images)
        imagefile = fullfile(seq_path, images{j});
        im = imread(imagefile);
        
        res = seg2edge(im, 2, [], 'regular');
        
        save_path = fullfile(save_dir, seqs{i});
        if ~exist(save_path, 'dir')
            mkdir(save_path)
        end
        imwrite(res, fullfile(save_path, images{j}))
        %disp("In 2nd For LOOP!!!")
    end
end