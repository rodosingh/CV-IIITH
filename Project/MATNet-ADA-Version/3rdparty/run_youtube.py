import os
import glob
import torch
import numpy
import PIL
from HED import estimate

def main():
    davis_folder = '/home2/rodosingh/AssignmentsProjects/CVProject/MATNet/data/YouTubeVOS_2018/train/JPEGImages'
    save_dir = '/home2/rodosingh/AssignmentsProjects/CVProject/MATNet/data/YouTubeVOS_2018-hed'

    videos = os.listdir(davis_folder)
    print(videos)

    for idx, video in enumerate(videos):
        print('process {}[{}/{}]'.format(video, idx, len(videos)))
        save_dir_video = os.path.join(save_dir, video)
        if not os.path.exists(save_dir_video):
            os.makedirs(save_dir_video)

        imagefiles = sorted(glob.glob(os.path.join(davis_folder, video, '*.jpg')))

        for imagefile in imagefiles:
            tensorInput = torch.FloatTensor(numpy.array(PIL.Image.open(imagefile))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0))

            tensorOutput = estimate(tensorInput)

            save_name = os.path.basename(imagefile)
            save_file = os.path.join(save_dir_video, save_name)
            PIL.Image.fromarray(
                (tensorOutput.clamp(0.0, 1.0).numpy().transpose(1, 2, 0)[:, :, 0] * 255.0).astype(numpy.uint8)).save(
                save_file)

main()
