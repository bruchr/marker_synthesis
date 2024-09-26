from datetime import datetime
import gc
import json
import os
import random
import time

import numpy as np
import torch

from dataloader import Dataset_own
from mytransforms import augmentors
from train_model import train as train_model
from unets import build_model


def main():

    random.seed()
    np.random.seed()

    # Load the settings for the model and for the training
    script_dir = os.path.dirname(__file__)
    with open(os.path.join(script_dir,'settings.json')) as infile:
            configs = json.load(infile)

    configs['path_results'] = os.path.join(configs['path_results'], configs['name'] + '_' + datetime.now().strftime('%Y%m%d%H%M'))

    # Set device (cpu or gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if str(device) == 'cuda':
        torch.backends.cudnn.benchmark = True
    configs['num_gpus'] = torch.cuda.device_count()

    # Create the model
    model = build_model(configs, device, print_summary=True)    

    train_dir = os.path.join(configs['path_data'], 'train')
    val_dir = os.path.join(configs['path_data'], 'vali')
    # Data loader
    root_dir = {'train': train_dir, 'val': val_dir}
    data_transforms = augmentors(augmentation=configs['augmentation'], p=configs['augmentation_prob'])
    datasets = {x: Dataset_own(configs, root_dir=root_dir[x], transform=data_transforms[x]) for x in ['train', 'val']}


    # Save config in the out folder
    os.makedirs(configs['path_results'], exist_ok=True)
    with open(os.path.join(configs['path_results'], "settings.json"), 'w', encoding='utf-8') as outfile:
        json.dump(configs, outfile, ensure_ascii=False, indent=2)

    # Train the model
    train_model(model, configs, datasets, device)

    # Save config in the out folder
    with open(os.path.join(configs['path_results'], "settings.json"), 'w', encoding='utf-8') as outfile:
        json.dump(configs, outfile, ensure_ascii=False, indent=2)


    # Clear memory
    del model
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == '__main__':
    s = time.time()
    main()
    d = time.time() - s
    print(f'Overall time required: {d:.0f}s -> {d/3600:.1f}h')