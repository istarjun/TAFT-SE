import sys

sys.path.append('../')
import argparse

import numpy as np
import scipy.io as sio

import torch
from utils.generator_coco20i_aux import coco20i_generator
from utils.model_TAFT_SE_Deeplab import TAFT_SE_Deeplab

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=3,
                        help='gpu device number. -1 for cpu.')
    parser.add_argument('--split', type=int, default=0,
                        help='split number')
    parser.add_argument('--shot', type=int, default=1,
                        help='number of shots')
    parser.add_argument('--query', type=int, default=10,
                        help='number of queries')
    # set params
    # -----------
    args = parser.parse_args()
    if args.gpu < 0:
        device = "cpu"
    else:
        device = "cuda:%d" % args.gpu
        torch.set_num_threads(1)
    n_shot = args.shot
    n_query = args.query
    n_query_test = 5
    nb_class = 1
    split_number = args.split
    encoder = 'resnet50'

    savefile_name = '/PATH/TO/SAVEFILE/NAME'
    matfile_name=savefile_name + '.mat'

    # set up training
    # ------------------
    model = TAFT_SE_Deeplab(encoder=encoder, nb_class = nb_class,  n_shot=n_shot, device=device, coco=True)


    pixel_acc_h = []
    tps = np.zeros(20)
    fps = np.zeros(20)
    tns = np.zeros(20)
    fns = np.zeros(20)
    checkpoint = torch.load(savefile_name)
    model.load_state_dict(checkpoint['state_dict'])
    print(savefile_name)

    print('Evaluating the model...')
    for i in range(50):
        test_generator = coco20i_generator(img_dir='/PATH/TO/COCO2014', label_dir='/PATH/TO/COCO20i/Label', aux_label_dir='/PATH/TO/COCO/Label', n_epoch=600,
                                            n_way=nb_class, n_sample=n_shot + n_query_test, split_number=split_number, train=False)

        pixel_accs = []
        for j, (images, labels), cl_ind in test_generator:
            with torch.no_grad():
                pixel_acc, tp, fp, tn, fn = model.evaluate(images, labels)
            pixel_acc = pixel_acc.data.cpu().numpy()
            tp = tp.data.cpu().numpy()
            fp = fp.data.cpu().numpy()
            tn = tn.data.cpu().numpy()
            fn = fn.data.cpu().numpy()
            pixel_accs.append(pixel_acc)
            tps[cl_ind] += tp
            fps[cl_ind] += fp
            tns[cl_ind] += tn
            fns[cl_ind] += fn

        pixel_acc_t = 100 * np.mean(np.array(pixel_accs))
        pixel_acc_h.extend([pixel_acc_t.tolist()])
        print('iteration=', i)
    IoU = tps / (tps + fps + fns)
    mIoU = IoU.mean()
    FBIoU = (tps.sum() / (tps.sum() + fps.sum() + fns.sum()) + tns.sum() / (tns.sum() + fps.sum() + fns.sum())) / 2
    print(('Pixel accuracy ={:.2f}%').format(pixel_acc_t))
    print(('mIOU ={:.2f}%').format(100 * mIoU))
    print(('FBIOU ={:.2f}%').format(100 * FBIoU))
    sio.savemat(matfile_name, {'pixel_acc_h': pixel_acc_h, 'tps': tps, 'fps': fps, 'tns': tns, 'fns': fns})

