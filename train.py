import os, sys; sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pickle, h5py, time, argparse, itertools, datetime

import torch
import torch.nn as nn
import torch.utils.data
from libs import PolaritySynapseDataset, collate_fn, res_unet, res_unet_plus, res_unet_plus_gn
from libs import FocalLossMul, WeightedMSE
from libs import res_unet_plus_polarity, res_unet_SE

# tensorboardX
from tensorboardX import SummaryWriter
import torchvision.utils as vutils

def get_args():
    parser = argparse.ArgumentParser(description='Training Synapse Detection Model')
    # I/O
    parser.add_argument('-t','--train',  default='/n/coxfs01/',
                        help='Input folder (train)')
    # parser.add_argument('-v','--val',  default='',
    #                     help='input folder (test)')
    parser.add_argument('-dn','--img-name',  default='im_uint8.h5',
                        help='Image data path')
    parser.add_argument('-ln','--seg-name',  default='seg-groundtruth2-malis.h5',
                        help='Ground-truth label path')
    parser.add_argument('-o','--output', default='result/train/',
                        help='Output path')
    parser.add_argument('-mi','--model-input', type=str,  default='31,204,204',
                        help='I/O size of deep network')

    # model option
    parser.add_argument('-ac','--architecture', help='model architecture')                    
    parser.add_argument('-ft','--finetune', default=False,
                        help='Fine-tune on previous model [Default: False]')
    parser.add_argument('-pm','--pre-model', type=str, default='',
                        help='Pre-trained model path')                  

    # optimization option
    parser.add_argument('-lt', '--loss', type=int, default=1,
                        help='Loss function')
    parser.add_argument('-lr', type=float, default=0.0001,
                        help='Learning rate')
    # parser.add_argument('-lr_decay', default='inv,0.0001,0.75',
    #                     help='learning rate decay')
    # parser.add_argument('-betas', default='0.99,0.999',
    #                     help='beta for adam')
    # parser.add_argument('-wd', type=float, default=5e-6,
    #                     help='weight decay')
    parser.add_argument('--volume-total', type=int, default=1000,
                        help='Total number of iteration')
    parser.add_argument('--volume-save', type=int, default=100,
                        help='Number of iteration to save')
    parser.add_argument('-g','--num-gpu', type=int,  default=1,
                        help='Number of gpu')
    parser.add_argument('-c','--num-cpu', type=int,  default=1,
                        help='Number of cpu')
    parser.add_argument('-b','--batch-size', type=int,  default=1,
                        help='Batch size')
    args = parser.parse_args()
    return args

def init(args):
    sn = args.output+'/'
    if not os.path.isdir(sn):
        os.makedirs(sn)
    # I/O size in (z,y,x), no specified channel number
    model_io_size = np.array([int(x) for x in args.model_input.split(',')])

    # select training machine
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return model_io_size, device

def get_input(args, model_io_size, opt='train'):
    # two dataLoader, can't be both multiple-cpu (pytorch issue)

    if opt=='train':
        dir_name = args.train.split('@')
        num_worker = args.num_cpu
        img_name = args.img_name.split('@')
        seg_name = args.seg_name.split('@')
    else:
        dir_name = args.val.split('@')
        num_worker = 1
        img_name = args.img_name_val.split('@')
        seg_name = args.seg_name_val.split('@')

    # may use datasets from multiple folders
    # should be either one or the same as dir_name
    seg_name = [dir_name[0] + x for x in seg_name]
    img_name = [dir_name[0] + x for x in img_name]
    # print(img_name)
    # print(seg_name)
    
    # 1. load data
    train_input = [None]*len(img_name)
    train_label = [None]*len(seg_name)
    assert len(img_name)==len(seg_name)

    # original image is in [0, 255], normalize to [0, 1]
    for i in range(len(img_name)):
        # train_input[i] = np.array(h5py.File(img_name[i], 'r')['main'])/255.0
        # train_label[i] = np.array(h5py.File(seg_name[i], 'r')['main'])
        # train_input[i] = (train_input[i].transpose(2, 1, 0))[100:-100, 200:-200, 200:-200]
        # train_label[i] = (train_label[i].transpose(2, 1, 0))[100:-100, 200:-200, 200:-200]
        train_input[i] = np.array(h5py.File(img_name[i], 'r')['main'])/255.0
        train_label[i] = np.array(h5py.File(seg_name[i], 'r')['main'])
        train_label[i] = (train_label[i] != 0).astype(np.float32)
        train_input[i] = train_input[i].astype(np.float32)
        
        assert train_input[i].shape==train_label[i].shape[1:]
        # print("input type: ", train_input[i].dtype)
        print("synapse pixels: ", np.sum(train_label[i][0]))
        print("volume shape: ", train_input[i].shape)    

    data_aug = True
    print('Data augmentation: ', data_aug)
    dataset = PolaritySynapseDataset(volume=train_input, label=train_label, vol_input_size=model_io_size,
                                 vol_label_size=model_io_size, data_aug = data_aug, mode = 'train')
    # to have evaluation during training (two dataloader), has to set num_worker=0
    SHUFFLE = (opt=='train')
    img_loader =  torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=SHUFFLE, collate_fn = collate_fn,
            num_workers=args.num_cpu, pin_memory=True)
    return img_loader

def get_logger(args):
    log_name = args.output+'/log'
    date = str(datetime.datetime.now()).split(' ')[0]
    time = str(datetime.datetime.now()).split(' ')[1].split('.')[0]
    log_name += '_approx_'+date+'_'+time
    logger = open(log_name+'.txt','w') # unbuffered, write instantly

    # tensorboardX
    writer = SummaryWriter('runs/'+log_name)
    return logger, writer

def get_validation(args):
    dir_name = '/n/coxfs01/zudilin/research/data/JWR/syn_vol1/'
    val_input_name = dir_name + 'val_input.h5'
    val_label_name = dir_name + 'val_label.h5'
    val_input = np.array(h5py.File(val_input_name, 'r')['main'])/255.0 #(z, y, x)
    val_label = np.array(h5py.File(val_label_name, 'r')['main'])/255.0
    return val_input, val_label

def train(args, train_loader, model, device, criterion, optimizer, logger, writer):
    # switch to train mode
    model.train()
    volume_id = 0

    # Visualize training procedure
    val_input, val_label = get_validation(args)
    val_input = torch.from_numpy(val_input.copy())
    val_label = torch.from_numpy(val_label.copy())
    val_input_show = val_input.unsqueeze(1).expand(12, 3, 256, 256) # (z, 3, y, x)
    val_label_show = val_label.unsqueeze(1).expand(12, 3, 256, 256)

    visual_image = vutils.make_grid(val_input_show, nrow=6, normalize=True, scale_each=True)
    visual_label = vutils.make_grid(val_label_show, nrow=6, normalize=True, scale_each=True)

    writer.add_image('EM Image', visual_image, 0)
    writer.add_image('GT Image', visual_label, 0)

    val_model_input = val_input.unsqueeze(0).unsqueeze(0) # (1, 1, 12, 256, 256)
    val_model_input = val_model_input.float()

    for i, (volume, label, class_weight, weight_factor) in enumerate(train_loader):
        volume_id += args.batch_size

        # restrict the weight
        # class_weight.clamp(max=1000)

        # for gpu computing
        # print(weight_factor)
        volume, label = volume.to(device), label.to(device)
        class_weight = class_weight.to(device)
        output = model(volume)
        # assert output.size() == label.size()
        # focal, dice = criterion(output, label, class_weight)
        # loss = focal + dice
        loss = criterion(output, label, class_weight)

        # compute gradient and do Adam step
        optimizer.zero_grad()       
        loss.backward()
        optimizer.step()

        logger.write("[Volume %d] train_loss=%0.4f lr=%.5f\n" % (volume_id, \
                loss.item(), optimizer.param_groups[0]['lr']))
        #writer.add_scalar('train_loss', loss.item(), volume_id)
        #writer.add_scalar('bce_loss', bce.item(), volume_id)
        #writer.add_scalar('dice_loss', dice.item(), volume_id)
        #writer.add_scalar('focal_loss', focal.item(), volume_id)
        writer.add_scalar('mean squared error', loss.item(), volume_id)

        # LR update
        #if args.lr > 0:
            #decay_lr(optimizer, args.lr, volume_id, lr_decay[0], lr_decay[1], lr_decay[2])
        
        if volume_id % 1000 < args.batch_size:
            volume = val_model_input.to(device)
            val_output = model(volume) # (1,3,z,y,x)
            val_output = val_output.squeeze() # (3,z,y,x)
            val_output = val_output.transpose(0, 1).cpu().detach() # (z, 3, y, x)
            volume.cpu().detach()
            torch.cuda.empty_cache()

            val_output_show = vutils.make_grid(val_output, nrow=6, normalize=True, scale_each=True)
            writer.add_image('Prediction', val_output_show, volume_id)
        
        if volume_id % args.volume_save < args.batch_size or volume_id >= args.volume_total:
            torch.save(model.state_dict(), args.output+('/volume_%d.pth' % (volume_id)))
        # Terminate
        if volume_id >= args.volume_total:
            break    #     

def main():
    args = get_args()

    print('0. initial setup')
    model_io_size, device = init(args) 
    logger, writer = get_logger(args)

    print('1. setup data')
    train_loader = get_input(args, model_io_size, 'train')

    print('2.0 setup model')
    #model = res_unet_plus_polarity(in_num=1, out_num=3)  
    model = res_unet_SE(in_num=1, out_num=3)

    if args.num_gpu > 1: model = nn.DataParallel(model, range(args.num_gpu))
    model = model.to(device)

    print('Fine-tune? ', bool(args.finetune))
    if bool(args.finetune):
        #model.load_state_dict(torch.load(args.pre_model))
        updated_params = torch.load(args.pre_model)
        new_params = model.state_dict()
        new_params.update(updated_params)
        model.load_state_dict(new_params)
        print('fine-tune on previous model:')
        print(args.pre_model)
            
    print('2.1 setup loss function')
    # if args.loss == 1:
    #     criterion = WeightedBCELoss()
    # elif args.loss == 2:    
    #     criterion = BCLoss()
    # elif args.loss == 3:
    #     criterion = FocalLoss()
    # elif args.loss == 3:
    #     criterion = BCLoss_focal(smooth=100.0)
    # print('Type of loss function: ', args.loss)
    criterion = WeightedMSE()

    print('3. setup optimizer')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), 
                                 eps=1e-08, weight_decay=1e-7, amsgrad=True)
    #optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.95)

    print('4. start training')
    train(args, train_loader, model, device, criterion, optimizer, logger, writer)
  
    print('5. finish training')
    logger.close()
    writer.close()

if __name__ == "__main__":
    main()
