import numpy as np
import h5py
import argparse
import datetime
import torch
import torch.utils.data

# tensorboardX
from tensorboardX import SummaryWriter

from em_net.data import AffinityDataset, collate_fn
from em_net.model.loss import WeightedBCELoss
from em_net.model.model_seg import UNet3DPniM2
from em_net.libs.sync import DataParallelWithCallback
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_args():
    parser = argparse.ArgumentParser(description='A script for training the PNI 3D UNET model for predicting ' +
                                                 'affinities.')
    # I/O options------------------------------------------------------------------------------------------------------#
    parser.add_argument('-iv', '--input-volume', default='train-input.h5',
                        help='Path to the training volume .h5 file(s). This script assumes that the h5 files have' +
                        'stored the volume in a dataset called "main".')
    parser.add_argument('-lv', '--label-volume', default='train-labels.h5',
                        help='Path to the training segmentation .h5 file(s), in order of correspondence to the passed' +
                        'training volumes. The volume\'s affinity will be generated on the fly as test labels. This ' +
                        'script assumes that the h5 files have stored the volume in a dataset called "main".')
    parser.add_argument('-o', '--output', default='result/train/',
                        help='Output directory used to save the prediction results as .h5 file(s). The directory is ' +
                             'automatically created if already not created.')
    parser.add_argument('-is', '--input-shape', type=str, default='18,160,160',
                        help="Model's input size (shape) formatted 'z, y, x' with no channel number.")
    parser.add_argument('-tv', '--train-data-ratio', type=float, default=0.7,
                        help='The ratio of the data used for training. The rest will be used for validation.')
    # model options----------------------------------------------------------------------------------------------------#
    parser.add_argument('-ft', '--finetune', type=bool, default=False,
                        help='Fine-tune on previous model [Default: False]')
    parser.add_argument('-pm', '--pre-model', type=str, default='',
                        help='Pre-trained model path')
    # optimization options---------------------------------------------------------------------------------------------#
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--volume-total', type=int, default=70000,
                        help='Total number of iteration')
    parser.add_argument('--volume-save', type=int, default=100,
                        help='Number of iterations for the script to save the model.')
    parser.add_argument('-g', '--num-gpu', type=int, default=1,
                        help='Number of CUDA-enabled graphics cards to be used to train the model.')
    parser.add_argument('-j', '--num-procs', type=int, default=1,
                        help='Number of processes to be used for the training data loader. The validation data loader' +
                        'will use only one process to load.')
    parser.add_argument('-bs', '--batch-size', type=int, default=1,
                        help='Batch size.')
    #------------------------------------------------------------------------------------------------------------------#
    args = parser.parse_args()
    return args


def check_output_dir(args):
    """
    Checks whether the output directory exists. If not, creates it.
    """
    # Create the output directory before writing into it.
    sn = args.output + '/'
    if not os.path.isdir(sn):
        os.makedirs(sn)
        print('Output directory was created.')
    print('Output directory: {}.'.format(sn))


def get_preferred_device(args):
    if args.num_gpu < 0:
        raise ValueError("The number of GPUs must be greater than or equal to zero.")
    return torch.device("cuda" if torch.cuda.is_available() and args.num_gpu > 0 else "cpu")


def get_model_input_shape(args):
    shape = np.array([int(x) for x in args.input_shape.split(',')])
    print("Model Shape: {}.".format(shape))
    return shape


def print_volume_stats(volume, name):
    print('Statistics for {}:'.format(name))
    print('Shape: {}.'.format(volume.shape))
    print('Min: {}.'.format(volume.min()))
    print('Max: {}.'.format(volume.max()))
    print('Mean: {}.'.format(volume.mean()))


def load_data(args, model_input_shape):
    # Parse all the input paths to both train and label volumes.------------------------------------------------------ #
    input_volume_paths = args.input_volume.split('@')
    label_volume_paths = args.label_volume.split('@')
    # Parse the ratio of the input data to be used for training.------------------------------------------------------ #
    train_ratio = args.train_data_ratio
    # Make sure they are equal in length.----------------------------------------------------------------------------- #
    assert len(input_volume_paths) == len(label_volume_paths)

    # Load the volumes.----------------------------------------------------------------------------------------------- #
    train_input = []
    train_label = []
    validation_input = []
    validation_label = []
    print("Loading volumes...")
    for i in range(len(input_volume_paths)):
        input_volume = np.array(h5py.File(input_volume_paths[i], 'r')['main']).astype(np.float32) / 255.0
        print("Loaded {}.".format(input_volume_paths[i]))
        print_volume_stats(input_volume, "input_volume")

        label_volume = np.array(h5py.File(label_volume_paths[i], 'r')['main'])
        print("Loaded {}.".format(label_volume_paths[i]))
        print_volume_stats(label_volume, "label_volume")

        assert input_volume.shape == label_volume.shape

        # Divide both input volume and label volume to train and validation sets.------------------------------------- #
        div_point = int(train_ratio * len(input_volume))
        train_input.append(input_volume[: div_point])
        validation_input.append(input_volume[div_point:])

        train_label.append(label_volume[: div_point])
        validation_label.append(label_volume[div_point:])
    # Create Pytorch Datasets.---------------------------------------------------------------------------------------- #
    data_aug = True
    print('Data augmentation: {}.'.format(data_aug))
    train_dataset = AffinityDataset(volume=train_input, label=train_label, vol_input_size=model_input_shape,
                                    vol_label_size=model_input_shape, data_aug=data_aug, mode='train')
    valid_dataset = AffinityDataset(volume=validation_input, label=validation_label, vol_input_size=model_input_shape,
                                    vol_label_size=model_input_shape, data_aug=None, mode='train')
    # Create Pytorch DataLoaders.------------------------------------------------------------------------------------- #
    print('Batch size: {}.'.format(args.batch_size))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=True, collate_fn=collate_fn,
                                               num_workers=args.num_procs, pin_memory=True)
    # TODO: Check whether this will work with args.num_procs.
    validation_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size,
                                                    shuffle=True, collate_fn=collate_fn,
                                                    num_workers=1, pin_memory=True)
    # ---------------------------------------------------------------------------------------------------------------- #
    return train_loader, validation_loader


def load_model(args, device):
    model = UNet3DPniM2(in_num=1, out_num=3)
    model = DataParallelWithCallback(model, device_ids=range(args.num_gpu))
    print("Loading model to device: {}.".format(device))
    model = model.to(device)
    print("Finished.")
    print("Finished loading.")
    print('Fine-tune? {}.'.format(bool(args.finetune)))
    if bool(args.finetune):
        model.load_state_dict(torch.load(args.pre_model))
        print('fine-tune on previous model:')
        print(args.pre_model)
    return model


def get_loggers(args):
    # Set loggers names.
    log_name = args.output + '/log'
    date = str(datetime.datetime.now()).split(' ')[0]
    time = str(datetime.datetime.now()).split(' ')[1].split('.')[0]
    log_name += '_approx_' + date + '_' + time
    logger = open(log_name + '.txt', 'w')  # unbuffered, write instantly
    print("Saving log file in {}.".format(log_name + '.txt'))

    # tensorboardX
    writer = SummaryWriter('runs/' + log_name)
    print("Saving Tensorboard summary to {}.".format('runs/' + log_name))
    return logger, writer


def train(args, train_loader, validation_loader, model, device, criterion, optimizer, logger, writer):
    # switch to train mode
    model.train()
    volume_id = 0
    # Validation dataset iterator:
    val_data_iter = iter(validation_loader)

    for _, volume, label, class_weight, _ in train_loader:
        volume_id += args.batch_size

        volume, label = volume.to(device), label.to(device)
        class_weight = class_weight.to(device)
        output = model(volume)

        loss = criterion(output, label, class_weight)
        writer.add_scalar('Training Loss', loss.item(), volume_id)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("[Volume %d] train_loss=%0.4f lr=%.5f\n" % (volume_id, loss.item(), optimizer.param_groups[0]['lr']))

        logger.write("[Volume %d] train_loss=%0.4f lr=%.5f\n" % (volume_id,
                                                                 loss.item(), optimizer.param_groups[0]['lr']))

        # Get the validation result if it's time. (Every Twenty iterations.)------------------------------------------ #
        if volume_id % args.volume_save < args.batch_size or volume_id >= args.volume_total:
            _, val_vol, val_label, val_class_weight, _ = next(val_data_iter)
            model.eval()
            val_vol, val_label = val_vol.to(device), val_label.to(device)
            val_class_weight = val_class_weight.to(device)
            val_out = model(val_vol)
            val_loss = criterion(val_out, val_label, val_class_weight)
            writer.add_scalar('Validation Loss', val_loss.item(), volume_id)
            print("validation_loss=%0.4f lr=%.5f\n" % (val_loss.item(), optimizer.param_groups[0]['lr']))
            logger.write("validation_loss=%0.4f lr=%.5f\n" % (val_loss.item(), optimizer.param_groups[0]['lr']))
            model.train()
            # Save the model if it's time.
            print("Saving the model in {}....".format(args.output + ('/volume_%d_%f.pth' % (volume_id, val_loss))))
            torch.save(model.state_dict(), args.output + ('/volume_%d_%f.pth' % (volume_id, val_loss)))
        # Terminate
        if volume_id >= args.volume_total:
            break  #


def main():
    args = get_args()

    print('0. initial setup')
    check_output_dir(args)
    model_input_shape = get_model_input_shape(args)
    device = get_preferred_device(args)
    logger, writer = get_loggers(args)

    print('1. setup data')
    train_loader, valid_loader = load_data(args, model_input_shape)

    print('2.0 setup model')
    model = load_model(args, device)

    print('2.1 setup loss function')
    criterion = WeightedBCELoss()

    print('3. setup optimizer')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999),
                                 eps=0.01, weight_decay=1e-6, amsgrad=True)

    print('4. start training')
    train(args, train_loader, valid_loader, model, device, criterion, optimizer, logger, writer)

    print('5. finish training')
    logger.close()
    writer.close()


if __name__ == "__main__":
    main()
