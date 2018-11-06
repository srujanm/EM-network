import os
import numpy as np
import h5py
import time
import argparse

import torch
import torch.utils.data
from em_net.data import AffinityDataset, collate_fn_test
from em_net.model.model_seg import UNet3DPniM2
from em_net.libs.sync import DataParallelWithCallback


def get_args():
    parser = argparse.ArgumentParser(
        description='Tests a previously trained PNI 3DUNet model for SNEMI affinity prediction.')
    # I/O
    parser.add_argument('-tv', '--test-volume', default='test-input.h5',
                        help='Path to the test volume .h5 file(s).')
    parser.add_argument('-o', '--output', default='result/test/',
                        help='Output directory used to save the prediction results as .h5 file(s). The directory is ' +
                             'automatically created if already not created.')
    parser.add_argument('-is', '--input-shape', type=str, default='18,160,160',
                        help="Model's input size (shape) formatted 'z, y, x' with no channel number.")

    # machine option
    parser.add_argument('-g', '--num-gpu', type=int, default=1,
                        help='Number of CUDA capable GPUs used for testing the model. Must be positive.')
    parser.add_argument('-j', '--num-procs', type=int, default=1,
                        help="Number of processes used with Pytorch's dataloader class.")
    parser.add_argument('-b', '--batch-size', type=int, default=1,
                        help='Batch size.')
    parser.add_argument('-m', '--model', help='Path of the model used for testing.')

    args = parser.parse_args()
    return args


def check_output(args):
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


def load_volumes(args, model_input_size):
    in_path = args.test_volume.split('@')

    # 1. load data
    print('Number of volumes passed: {}.'.format(len(in_path)))
    test_input = [None] * len(in_path)
    result = [None] * len(in_path)
    weight = [None] * len(in_path)

    # may use datasets from multiple folders
    # should be either one or the same as dir_name
    # original image is in [0, 255], normalize to [0, 1]
    print("Loading volumes...")
    for i in range(len(in_path)):
        test_input[i] = np.array(h5py.File(in_path[i], 'r')['main']) / 255.0
        print("Loaded {}.".format(in_path[i]))
        print("Volume shape: {}".format(test_input[i].shape))
        result[i] = np.zeros(test_input[i].shape)
        weight[i] = np.zeros(test_input[i].shape)

        result[i] = np.stack([result[i]] * 3)
        print("Result shape: {}.".format(result[i].shape))
        print("Weight shape: {}.".format(weight[i].shape))

    dataset = AffinityDataset(volume=test_input, label=None, vol_input_size=model_input_size,
                              vol_label_size=None, sample_stride=model_input_size / 2,
                              data_aug=None, mode='test')

    img_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_test,
        num_workers=args.num_procs, pin_memory=True)
    return img_loader, result, weight


def gaussian_blend(sz):
    # Gaussian blending
    print('Calculating Gaussian Blending shape...')
    zz, yy, xx = np.meshgrid(np.linspace(-1, 1, sz[0], dtype=np.float32),
                             np.linspace(-1, 1, sz[1], dtype=np.float32),
                             np.linspace(-1, 1, sz[2], dtype=np.float32), indexing='ij')

    dd = np.sqrt(zz * zz + yy * yy + xx * xx)
    sigma, mu = 0.5, 0.0
    ww = 1e-4 + np.exp(-((dd - mu) ** 2 / (2.0 * sigma ** 2)))
    print('Gaussian Blending Weight Shape: {}.'.format(ww.shape))

    return ww


def load_model(args, device):
    print(args.model)
    model = UNet3DPniM2(in_num=1, out_num=3)
    model = DataParallelWithCallback(model, device_ids=range(args.num_gpu))
    print("Loading model to device: {}.".format(device))
    model = model.to(device)
    print("Finished.")
    print("Loading model state from file {}...")
    model.load_state_dict(torch.load(args.model))
    print("Finished loading.")
    return model


def test(args, test_loader, result, weight, model, device, model_io_size):
    # switching model to eval mode
    model.eval()
    volume_id = 0
    ww = gaussian_blend(model_io_size)

    print('Starting prediction...')
    start = time.time()
    with torch.no_grad():
        for i, (pos, volume) in enumerate(test_loader):
            # Transferring both the model and the input volume to the specified device.
            volume = volume.to(device)
            output = model(volume)
            if i == 0:
                print("I/O information:")
                print("Input Volume size: {}.".format(volume.size()))
                print("Output Volume size: {}.".format(output.size()))

            volume_id += args.batch_size
            print('Performing prediction on volume_id: {}...'.format(volume_id))
            print('Statistics:')
            print('Input Volume min: {}.'.format(volume.min()))
            print('Input Volume max: {}.'.format(volume.max()))
            print('Output Volume min: {}.'.format(output.min()))
            print('Output Volume max: {}.'.format(output.max()))
            # sz = (3, 8, 224, 224)
            # sz = (3, 8, 256, 256)
            sz = tuple([3] + list(model_io_size))
            for idx in range(output.size()[0]):
                st = pos[idx]
                result[st[0]][:, st[1]:st[1] + sz[1], st[2]:st[2] + sz[2], st[3]:st[3] + sz[3]] += \
                    output[idx].cpu().detach().numpy().reshape(sz) * np.expand_dims(ww, axis=0)
                weight[st[0]][st[1]:st[1] + sz[1], st[2]:st[2] + sz[2], st[3]:st[3] + sz[3]] += ww

    end = time.time()
    print("Finished.")
    print("Prediction time: {}.".format(end - start))
    print("Saving results...")
    for vol_id in range(len(result)):
        print("Result with vol_id: {}...".format(vol_id))
        result[vol_id] = result[vol_id] / np.expand_dims(weight[vol_id], axis=0)
        print('Statistics:')
        print('Input Volume min: {}.'.format(result[vol_id].min()))
        print('Input Volume max: {}.'.format(result[vol_id].max()))
        # data = (result[vol_id]*255).astype(np.uint8)
        # data[data < 128] = 0
        hf = h5py.File(args.output + '/volume_' + str(vol_id) + '.h5', 'w')
        hf.create_dataset('main', data=result[vol_id])
        hf.close()
        print("Saved vol_id {}.".format(vol_id))


def main():
    args = get_args()

    print('0. initial setup')
    check_output(args)
    model_input_shape = get_model_input_shape(args)
    device = get_preferred_device(args)

    print('1. data setup')
    test_loader, result, weight = load_volumes(args, model_input_shape)

    print('2. model setup')
    model = load_model(args, device)

    print('3. testing')
    test(args, test_loader, result, weight, model, device, model_input_shape)

    print('Finished.')


if __name__ == "__main__":
    main()
