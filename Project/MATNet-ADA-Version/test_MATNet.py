import torch
from torchvision import transforms

import os
import glob
from tqdm import tqdm
from PIL import Image
from scipy.misc import imresize

from modules.MATNet import Encoder, Decoder
from utils.utils import check_parallel
from utils.utils import load_checkpoint_epoch
import argparse

def flip(x, dim):
    if x.is_cuda:
        return torch.index_select(x, dim, torch.arange(x.size(dim) - 1, -1, -1).long().cuda(0))
    else:
        return torch.index_select(x, dim, torch.arange(x.size(dim) - 1, -1, -1).long())

def model_loader(model_name, epoch):
    """Load trained encoder and decoder..."""
    encoder_dict, decoder_dict, enc_opt_dict, dec_opt_dict, load_args = load_checkpoint_epoch(model_name, epoch, True, False)
    encoder = Encoder()
    decoder = Decoder()
    encoder_dict, decoder_dict = check_parallel(encoder_dict, decoder_dict)
    encoder.load_state_dict(encoder_dict)
    decoder.load_state_dict(decoder_dict)

    encoder.cuda()
    decoder.cuda()

    encoder.train(False)
    decoder.train(False)
    return encoder, decoder

def segment(image_dir, flow_dir, davis_result_dir):
    """Function that segments fg moving objects from BackGround"""
    model_name = 'MATNet'  # specify the model name
    epoch = 0  # specify the epoch number

    inputRes = (473, 473)
    use_flip = True

    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    image_transforms = transforms.Compose([to_tensor, normalize])

    # Load encoder and decoder
    encoder, decoder = model_loader(model_name, epoch)

    # Define IMage files and Flow files
    imagefiles = sorted(glob.glob(os.path.join(image_dir, '*.png')))[:-1]#<-----change format of image if required
    flowfiles = sorted(glob.glob(os.path.join(flow_dir, '*.png')))#<-----change format of image if required

    with torch.no_grad():
        for imagefile, flowfile in zip(imagefiles, flowfiles):
            image = Image.open(imagefile).convert('RGB')
            flow = Image.open(flowfile).convert('RGB')

            width, height = image.size

            image = imresize(image, inputRes)
            flow = imresize(flow, inputRes)

            image = image_transforms(image)
            flow = image_transforms(flow)

            image = image.unsqueeze(0)
            flow = flow.unsqueeze(0)

            image, flow = image.cuda(), flow.cuda()

            r5, r4, r3, r2 = encoder(image, flow)
            mask_pred, bdry_pred, p2, p3, p4, p5 = decoder(r5, r4, r3, r2)

            if use_flip:
                image_flip = flip(image, 3)
                flow_flip = flip(flow, 3)
                r5, r4, r3, r2 = encoder(image_flip, flow_flip)
                mask_pred_flip, bdry_pred_flip, p2, p3, p4, p5 =\
                    decoder(r5, r4, r3, r2)

                mask_pred_flip = flip(mask_pred_flip, 3)
                bdry_pred_flip = flip(bdry_pred_flip, 3)

                mask_pred = (mask_pred + mask_pred_flip) / 2.0
                bdry_pred = (bdry_pred + bdry_pred_flip) / 2.0

            mask_pred = mask_pred[0, 0, :, :]
            mask_pred = Image.fromarray(
                mask_pred.cpu().detach().numpy() * 255).convert('L')

            save_folder = '{}/{}_epoch{}/{}'.format(davis_result_dir,
                                                    model_name, epoch, video)
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)

            save_file = os.path.join(save_folder,
                                        os.path.basename(imagefile)[:-4] + '.png')
            mask_pred = mask_pred.resize((width, height))
            mask_pred.save(save_file)


if __name__ == '__main__':

    # ==============================================================================================
    # Create the parser
    parser = argparse.ArgumentParser()
    # Add an argument
    parser.add_argument('--davis_result_dir', type=str, required=True)
    # davis_result_dir = 'data/DAVIS2017/DAVIS17_Results'
    parser.add_argument('--val_set', type=str, required=True)
    parser.add_argument('--davis_root_dir', type=str, required=True)
    parser.add_argument('--davis_flow_dir', type=str, required=True)
    # Parse the argument
    args = parser.parse_args()
    # ==============================================================================================

    # val_set = 'data/DAVIS2017/ImageSets/2017/val1.txt'
    # davis_root_dir = 'data/DAVIS2017/JPEGImages/480p'
    # davis_flow_dir = 'data/DAVIS2017/DavisFlow'
    davis_result_dir = args.davis_result_dir
    val_set = args.val_set
    davis_root_dir = args.davis_root_dir
    davis_flow_dir = args.davis_flow_dir

    with open(val_set) as f:
        seqs = f.readlines()
        seqs = [seq.strip() for seq in seqs]

    for video in tqdm(seqs):

        image_dir = os.path.join(davis_root_dir, video)
        flow_dir = os.path.join(davis_flow_dir, video)

        # segment fg motion from background
        segment(image_dir, flow_dir, davis_flow_dir)

# python test_MATNet.py --davis_result_dir <folder-to-save-results> --val_set <txt-file-containing-val-object-names> --davis_root_dir <path-to-folder-JPEGImages/480p> --davis_flow_dir <path-to-folder-of-flow-results>
