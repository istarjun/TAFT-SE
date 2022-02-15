import sys
sys.path.append('../')
import argparse

import numpy as np
import scipy.io as sio

import torch
from utils.generator_pascal5i_aux_mse import pascal5i_generator
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

    args = parser.parse_args()        
    if args.gpu < 0:
        device="cpu"
    else:
        device="cuda:%d" %args.gpu
        torch.set_num_threads(1)
    max_iter= 30001
    lr1=1e-2
    lr2=lr1/10
    lr_rate = 1/10
    lrstep = 20000
    momentum = 0.9
    n_shot=args.shot
    n_query=args.query
    n_query_test = 5
    nb_class_train=1
    nb_class_test=1
    split_number = args.split
    encoder='resnet50'
    weight_decay_rate1 = 3e-4
    weight_decay_rate2 = 3e-4

    savefile_name = 'save/TAFT_SE_DeeplabV3+_' + encoder + 'Pascal5i_%d_%ds_%dq' %(split_number, n_shot, n_query)

    matfile_name=savefile_name + '.mat'
    savefile_name_last =savefile_name + '_last'

    model = TAFT_SE_Deeplab(encoder=encoder, nb_class = nb_class_train,  n_shot=n_shot, device=device)

    model.set_optimizer(learning_rate1=lr1, learning_rate2=lr2, momentum=momentum, weight_decay_rate1=weight_decay_rate1,weight_decay_rate2=weight_decay_rate2, lrstep=lrstep, decay_rate=lr_rate)

    train_generator = pascal5i_generator(path_dir= '/PATH/TO/PASCAL5i', aux_label_dir='/PATH/TO/AUX/LABEL', n_epoch = max_iter,
                                       n_way=nb_class_train, n_sample=n_shot + n_query, split_number=split_number, train=True)

    loss_h=[]
    pixel_acc_h_test=[]
    mIOU_h_test=[]
    FBIOU_h_test=[]
    
    mIOU_best=0
    epoch_best=0

    for t, (images, labels, labels_aux) in train_generator:
        loss = model.train(images, labels, labels_aux)
        loss_h.extend([loss.tolist()])
        if (t % 50 == 0):
            print("Episode: %d, Train Loss: %f "%(t, loss))
        if (t!=0) and (t % 500 == 0):
            print('Evaluation in Validation data')
            test_generator = pascal5i_generator(path_dir= '/PATH/TO/PASCAL5i', aux_label_dir='/PATH/TO/AUX/LABEL', n_epoch = 600,
                                    n_way=nb_class_train, n_sample=n_shot + n_query_test, split_number=split_number, train=False)
        
            pixel_accs = []
            tps = np.zeros(5)
            fps = np.zeros(5)
            tns = np.zeros(5)
            fns = np.zeros(5)

            for i, (images, labels), cl_ind in test_generator:
                with torch.no_grad():
                    pixel_acc, tp, fp, tn, fn  = model.evaluate(images, labels)
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
            IoU = tps/(tps+fps+fns)
            mIoU = IoU.mean()
            FBIoU = (tps.sum()/(tps.sum()+fps.sum()+fns.sum()) +tns.sum()/(tns.sum()+fps.sum()+fns.sum()))/2

            pixel_acc_t=100*np.mean(np.array(pixel_accs))
            pixel_acc_h_test.extend([pixel_acc_t.tolist()])
            mIOU_t=100*mIoU
            mIOU_h_test.extend([mIOU_t.tolist()])
            FBIOU_t=100*FBIoU
            FBIOU_h_test.extend([FBIOU_t.tolist()])

            if mIOU_best < mIOU_t:
                mIOU_best = mIOU_t
                epoch_best=t
                torch.save({'state_dict':model.state_dict()}, savefile_name)
            print(('Accuracy ={:.2f}%').format(100*np.mean(np.array(pixel_accs))))
            print(('mIOU  ={:.2f}%').format(mIOU_t))
            print(('FBIOU  ={:.2f}%').format(FBIOU_t))
            print(('Best mIOU  ={:.2f}%, at {}').format(mIOU_best, epoch_best))
            sio.savemat(matfile_name, {'pixel_acc_h_test':pixel_acc_h_test, 'mIOU_h_test':mIOU_h_test, 'FBIOU_h_test':FBIOU_h_test})

    torch.save({'state_dict': model.state_dict()}, savefile_name_last)
