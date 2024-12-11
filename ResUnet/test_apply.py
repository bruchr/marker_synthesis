import json
from pathlib import Path
import time

import tifffile as tiff
import torch

from apply_model import apply as apply_model
from unets import build_model


'''
Script to apply the trained model to new data without calculation of metrics
'''
def load_config_and_model(model_folder:Path):
    model_path = Path(model_folder) / 'state_dict.pth'
    settings_path = Path(model_folder) / 'settings.json'
    if not settings_path.exists():
         print('Warning: No settings file found in model folder. Using settings file in code folder!')
         settings_path = Path(__file__).parent / 'settings.json'
    
    # Load the settings for the model and for the training
    with open(settings_path) as infile:
        configs = json.load(infile)

    configs['run_name'] = configs['name']

    # Set device (cpu or gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if str(device) == 'cuda':
        torch.backends.cudnn.benchmark = True
    configs['num_gpus'] = torch.cuda.device_count()

    net = build_model(configs, device, print_summary=False)

    net.load_state_dict(torch.load(model_path, map_location=device))

    return configs, net, device



if __name__ == "__main__":

    model_folder = Path('path/to/model/folder')
    path_list = [Path('path/to/image/folder')]

    recursive = False
    img_slice = (slice(None), 0, ...) # None ### input should be single channel, e.g. img_slice = (slice(None), 1, ...)


    s = time.time()

    configs, model, device = load_config_and_model(model_folder)
    
    for path_files in path_list:
        output_path = model_folder / ('results_' + path_files.name)
        error_log_file =output_path / 'error.log'
        info_file =output_path / 'info.txt'

        output_path.mkdir(exist_ok=True, parents=True)
        with open(info_file, 'a') as inf_file:
                inf_file.write(f'Data from:\n{path_files}\n')
        
        exp = '**/*.tif' if recursive else '*.tif'
        for f_path in sorted(path_files.glob(exp)):
            try:
                print(f'Processing {f_path.name}')
                # img = tiff.imread(f_path) # Test if image is corrupted

                output_folder = output_path / f_path.relative_to(path_files).parent
                output_folder.mkdir(parents=True, exist_ok=True)
                configs['path_results'] = str(output_folder)
                
                apply_model(model, configs, [f_path], device, slice_img=img_slice)
            
            except tiff.tifffile.TiffFileError as error:
                print(error)
                print('Error during handling of: {}'.format(f_path))
                
                with open(error_log_file, 'a') as err_file:
                    err_file.write('{}\n'.format(error))
                    err_file.write('Error during handling of: {}\n\n'.format(f_path))
        
    d = time.time() - s
    print(f'Required time: {d:.0f}s -> {d/60:.1f}min')