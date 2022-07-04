import os
import glob
import torch
import numpy
import PIL
from HED import estimate
import argparse

def main(davis_folder, save_dir):
    # davis_folder = '/home2/rodosingh/AssignmentsProjects/CVProject/MATNet/data/DAVIS2017/JPEGImages/480p'
    # save_dir = '/home2/rodosingh/AssignmentsProjects/CVProject/MATNet/data/DAVIS2017/DAVIS2017-HED'

    videos = os.listdir(davis_folder)
    print(sorted(videos))

    for idx, video in enumerate(videos):
        if video == "vid_1":#<-------------------------------------------------------
            print('process {}[{}/{}]'.format(video, idx, len(videos)))
            save_dir_video = os.path.join(save_dir, video)
            if not os.path.exists(save_dir_video):
                os.makedirs(save_dir_video)

            imagefiles = sorted(glob.glob(os.path.join(davis_folder, video, '*.png')))# .jpg replace it

            for imagefile in imagefiles:
                tensorInput = torch.FloatTensor(numpy.array(PIL.Image.open(imagefile))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0))

                tensorOutput = estimate(tensorInput)

                save_name = os.path.basename(imagefile)
                save_file = os.path.join(save_dir_video, save_name)
                PIL.Image.fromarray(
                    (tensorOutput.clamp(0.0, 1.0).numpy().transpose(1, 2, 0)[:, :, 0] * 255.0).astype(numpy.uint8)).save(
                    save_file)

# ===========================================================================================================================
if __name__ == '__main__':

    # Create the parser
    parser = argparse.ArgumentParser()
    # Add an argument
    parser.add_argument('--davis_folder', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    # Parse the argument
    args = parser.parse_args()
    main(args.davis_folder, args.save_dir)
