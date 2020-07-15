import torch

import utility
import data_1 as data
import model_1 as model
import loss
from option import args
from trainer import Trainer
from tqdm import tqdm
import os
import cv2
import numpy as np
from multiprocessing import Queue

loader = data.Data(args)
_model = model.Model(args)

def tensor2img(imgtensor, min_max=(0, args.rgb_range), hr_in=False):
    tensor = imgtensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]

    img_np = tensor.detach().numpy()
    img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR cv的保存顺序是bgr

    img_np = (img_np * 255.0).round()
    # print(img_np)
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(np.uint8)

def test(_model,loader):
    queue = Queue()
    torch.set_grad_enabled(False)
    
    _model = prepare(_model)[0]
    _model.eval()
    save_path = './results/test_whatswrong/'
    os.makedirs(save_path,exist_ok=True)

    # timer_test = utility.timer()
    for lr, hr, filename in tqdm(loader.loader_test, ncols=80):
        # print(lr.shape,hr.shape)
        lr, hr = prepare(lr, hr)
        sr = _model(lr)
        sr = tensor2img(sr)
        cv2.imwrite(save_path+filename[0]+'.png',sr)

    torch.set_grad_enabled(True)

def prepare(*args):
    device = torch.device('cuda')
    def _prepare(tensor):
        return tensor.to(device)

    return [_prepare(a) for a in args]

test(_model,loader)