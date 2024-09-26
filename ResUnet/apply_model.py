import os
import time

import numpy as np
import tifffile as tiff
import torch
from tqdm import tqdm

from sliding_window import generateSlices
from utils import min_max_normalization


def apply(net, configs, img_ids, device, slice_img=None):

    print('Apllying of ' + configs['run_name'])

    # Initialize list for the evaluation time
    pred_time = []
    for i, img_id in enumerate(img_ids):
        img = tiff.imread(img_id)
        if slice_img is not None:
            img = img[slice_img]
        since = time.time()

        pred = inference(net,configs, img, device)

        pred_time.append(time.time() - since)
        print('time needed: {}'.format(pred_time[-1]))
    
        # Save the images
        image_name = os.path.basename(img_id)
        tiff.imsave(configs['path_results'] +  '/pp_' + image_name, pred)


    configs['aplly_time'] = np.mean(np.array(pred_time) / len(img_ids))


def inference(net, configs, img, device):
    """ Predict images with specified trained PyTorch model
    
    :param net: Trained PyTorch model used to predict data
        :type net: model
    :param net: List of file path which schould be evaluated
        :type net: list
    :param device: cuda (gpu) or cpu.
        :type device:
    :param configs: Dictionary containing data paths and information for the training and evaluation process.
        :type configs: dict
    :return: None.
    """

    # Load trained model
    net.eval()
    torch.set_grad_enabled(False)
    
    # Normalize image
    img = min_max_normalization(img.astype(np.float32))
    
    # Sliding window approach for images bigger than supported (memory demand)
    windows, overlap = generateSlices(np.shape(img), maxWindowSize=configs["shape"], overlapPercent=0.5)
    prediction = np.zeros(shape=(1, img.shape[0], img.shape[1], img.shape[2]), dtype=np.float32)

    for window in tqdm(windows, ncols=50):

        # Model needs [batch size, channels, depth, height, width]
        himg = np.expand_dims(np.expand_dims(np.copy(img[window]), axis=0), axis=0)

        # Reduce boundary effects --> use no boundary region --> note the required indices
        z_overlap = int(overlap[0]/2) if window[0].start > 0 else 0
        y_overlap = int(overlap[1]/2) if window[1].start > 0 else 0
        x_overlap = int(overlap[2]/2) if window[2].start > 0 else 0

        z_min = window[0].start + z_overlap
        y_min = window[1].start + y_overlap
        x_min = window[2].start + x_overlap

        # Prediction
        himg = torch.from_numpy(himg)
        himg = net(himg.to(torch.float).to(device))

        # if configs['label_type'] == 'binary' or configs['label_type'] == 'binary_ignore':
        #     himg = torch.sigmoid(himg)
        # elif configs['label_type'] == 'dist_label':
        #     himg = himg #linear activation function
        # else:
        #     himg = F.softmax(himg, dim=1)

        himg = himg[0, :, z_overlap:, y_overlap:, x_overlap:].cpu().numpy()
        prediction[:, z_min:window[0].stop, y_min:window[1].stop, x_min:window[2].stop] = himg
        

    # # Clear memory
    # del net
    # torch.cuda.empty_cache()
    # gc.collect()

    # print('Shape of bin: {}'.format(np.shape(prediction_bin)))
    # print('Shape of raw: {}'.format(np.shape(prediction)))

    return prediction