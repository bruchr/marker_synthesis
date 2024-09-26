import json
import os
import random
import shutil
from pathlib import Path

import numpy as np
import tifffile as tiff

from sliding_window import generateSlices
from utils import min_max_normalization

def get_folder_path_list(path:str, pattern:str=None)->tuple:
    file_paths_train = []
    file_paths_vali = []
    file_paths_test = []
    for dirpath, _, files in os.walk(path):
        if pattern is None or pattern in dirpath:
            counter = 0
            for f in files:
                if f.endswith('.tif'):
                    file_path = os.path.join(dirpath, f)
                    if counter == 0:
                        file_paths_test.append(file_path)
                    elif counter == 1:
                        file_paths_vali.append(file_path)
                    else:
                        file_paths_train.append(file_path)
                    counter += 1
    return file_paths_train, file_paths_vali, file_paths_test

def make_split(path_dataset, ratio=0.2):
    path_train = Path(path_dataset) / 'train' / 'patched_images'
    path_vali = Path(path_dataset) / 'vali' / 'patched_images'

    path_vali.mkdir(parents=True, exist_ok=False)

    path_train_list = list(path_train.glob('*.tif'))
    vali_no = int(round(len(path_train_list) * ratio))
    selected_inds = np.random.choice(len(path_train_list), size=vali_no, replace=False)

    for ind in selected_inds:
        shutil.move(path_train_list[ind], path_vali / path_train_list[ind].name)


def create_dataset(paths:list, mode:str, output_folder:str, output_folder_seg:str=None, patch_overlap:float=0, use_threshold:bool=False, threshold:float=0.5, add_noise:bool=False, split_channels:bool=False)->None:
    if split_channels:
        os.makedirs(os.path.join(output_folder, 'trainA'), exist_ok=True)
        os.makedirs(os.path.join(output_folder, 'trainB'), exist_ok=True)
    else:
        os.makedirs(output_folder, exist_ok=True)
    
    if mode=='test' and output_folder_seg is not None:
        os.makedirs(output_folder_seg, exist_ok=True)
    
    parameter_dict = {
        'paths': paths,
        'mode': mode,
        'output_folder': output_folder,
        'output_folder_seg': output_folder_seg,
        'patch_overlap': patch_overlap,
        'use_threshold': use_threshold,
        'threshold': threshold,
        'add_noise': add_noise,
        'split_channels': split_channels,
    }
    with open(os.path.join(os.path.join(output_folder, "parameter.json")), 'w', encoding='utf-8') as outfile:
            json.dump(parameter_dict, outfile, ensure_ascii=False, indent=2)
    
    img_id = 0
    parameter_dict['ID Paths'] = []
    for path in paths:
        file_name = os.path.basename(path)
        folder_path = os.path.dirname(path)
        parameter_dict['ID Paths'].append(f'id{img_id}: {path}')
        if mode=='test':
            if split_channels:
                img = tiff.imread(path)
                img = np.moveaxis(img, 1, -1)
                output_path_A = os.path.join(output_folder, 'trainA', 'id{}.tif'.format(img_id))
                output_path_B = os.path.join(output_folder, 'trainB', 'id{}.tif'.format(img_id))
                tiff.imwrite(output_path_A, img[...,0], imagej=True)
                tiff.imwrite(output_path_B, img[...,1], imagej=True)
            else:
                output_path = os.path.join(output_folder, 'id{}.tif'.format(img_id))
                print('{}: {}'.format(img_id, path))
                shutil.copyfile(path, output_path)
            if output_folder_seg is not None:
                file_path_seg = os.path.join(folder_path.replace('Raw_Data', 'Segmentation'), 'pp_'+file_name)
                output_path_seg = os.path.join(output_folder_seg, 'seg_' + 'id{}.tif'.format(img_id))
                shutil.copyfile(file_path_seg, output_path_seg)
        else:
            img = tiff.imread(path)
            img = np.moveaxis(img, 1, -1)
            img = img.astype(np.float32)
            if add_noise:
                img = img + np.random.normal(5,5, img.shape).astype(np.float32)
            for ch in range(img.shape[-1]):
                img[..., ch] = min_max_normalization(img[..., ch])

            slice_shape = [32,256,256]
            if np.any(np.asarray(img.shape[:-1]) < np.asarray(slice_shape)):
                continue
            slices, _ = generateSlices(img.shape[:-1], [32,256,256], overlapPercent=patch_overlap)

            sl_id = 0
            for sl in slices:
                img_sl = img[sl]
                if use_threshold==False or np.max(img_sl[...,0]) >= threshold or random.random()>0.8:
                    if split_channels:
                        img_sl_int = (img_sl*255).astype(np.uint8)
                        output_path_A = os.path.join(output_folder, 'trainA', 'id{}_sl{}.tif'.format(img_id, sl_id))
                        output_path_B = os.path.join(output_folder, 'trainB', 'id{}_sl{}.tif'.format(img_id, sl_id))
                        tiff.imwrite(output_path_A, img_sl_int[...,0], imagej=True)
                        tiff.imwrite(output_path_B, img_sl_int[...,1], imagej=True)
                    else:
                        img_sl = np.moveaxis(img_sl, -1, 1)
                        output_path = os.path.join(output_folder, 'id{}_sl{}.tif'.format(img_id, sl_id))
                        tiff.imwrite(output_path, img_sl, imagej=True)
                sl_id += 1
        img_id += 1

    with open(os.path.join(os.path.join(output_folder, "parameter.json")), 'w', encoding='utf-8') as outfile:
        json.dump(parameter_dict, outfile, ensure_ascii=False, indent=2)


if __name__ == '__main__':

    paths_train = [
        'path/to/image.tif',
    ]
    paths_vali = ['']
    paths_test = ['']
    
    split_channels = False # generate two folders with single channel images instead of a image file with two channels
    patch_overlap = 0.25
    use_threshold = True # discard dark regions by 80% chance
    threshold = 0.25 # max intensity threshold used to descard the crops
    add_noise = False # add random noise to the crops

    for mode in ['train']:
        if mode=='train':
            output_folder = 'path_dataset/train/patched_images'
            # output_folder = 'D:/Bruch/Mario_Auswertung/Ki67_Prediction/Train_Data/V8_large_Cas3_bright_random/train/patched_images'
            output_folder_seg = None
            paths = paths_train
        elif mode=='vali':
            output_folder = 'path_dataset/vali/patched_images'
            output_folder_seg = None
            paths = paths_vali
        elif mode=='test':
            output_folder = 'path_dataset_test/images'
            output_folder_seg = None
            paths = paths_test
        else:
            continue
        
        create_dataset(paths, mode, output_folder, output_folder_seg, patch_overlap=patch_overlap, use_threshold=True, add_noise=False, split_channels=split_channels)

    # make_split(Path('path/dataset'))