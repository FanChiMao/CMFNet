"""

"""

import os
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import utils

from data_RGB import get_test_data
from model.CMFNet import CMFNet
from skimage import img_as_ubyte
from utils.image_utils import rgb2hsv_torch
import gc

model_restoration = CMFNet()
def run():
    torch.multiprocessing.freeze_support()
    parser = argparse.ArgumentParser(description='Image Restoration')

    parser.add_argument('--input_dir', default='./datasets/', type=str, help='Directory of validation images')
    parser.add_argument('--result_dir', default='./test_results/', type=str, help='Directory for results')
    parser.add_argument('--weights',
                        default='./pretrained_model/deraindrop_model.pth', type=str,
                        help='Path to weights')
    parser.add_argument('--dataset', default='deraindrop', type=str, help='[deraindrop, dehaze, deblur]')
    parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus


    utils.load_checkpoint(model_restoration, args.weights)
    print("===>Testing using weights: ", args.weights)
    model_restoration.cuda()
    # model_restoration = nn.DataParallel(model_restoration)
    model_restoration.eval()

    dataset = args.dataset
    rgb_dir_test = os.path.join(args.input_dir, dataset, 'test', 'input')
    test_dataset = get_test_data(rgb_dir_test, img_options={})
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False,
                             pin_memory=False)

    result_dir = os.path.join(args.result_dir, dataset)
    utils.mkdir(result_dir)


    with torch.no_grad():
        for ii, data_test in enumerate(tqdm(test_loader), 0):
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

            input_ = data_test[0].cuda()
            filenames = data_test[1]

            # Padding in case images are not multiples of 8
            if dataset == 'RealBlur_J' or dataset == 'RealBlur_R':
                factor = 8
                h, w = input_.shape[2], input_.shape[3]
                H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
                padh = H - h if h % factor != 0 else 0
                padw = W - w if w % factor != 0 else 0
                input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

            restored = model_restoration(input_)
            # print('factor a: ', restored[2].item())
            restored = torch.clamp(restored[0], 0, 1)


            # Un-pad images to original dimensions
            if dataset == 'RealBlur_J' or dataset == 'RealBlur_R':
                restored = restored[:, :, :h, :w]

            restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()

            for batch in range(len(restored)):
                restored_img = img_as_ubyte(restored[batch])
                utils.save_img((os.path.join(result_dir, filenames[batch] + '.png')), restored_img)

if __name__ == '__main__':
    run()