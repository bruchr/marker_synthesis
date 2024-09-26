# Implementation of ResUnet and pix2pix for marker transformation.
**Roman Bruch, Mario Vitacolonna, Rüdiger Rudolf and Markus Reischl**

## Installation

Clone this repository using [`git`](https://git-scm.com/downloads) or download it as `.zip`file and extract it.

Install a conda distribution like [Anaconda](https://www.anaconda.com/products/individual).

Create the environment with conda:
```
conda env create -f environment.yml
```

Activate the environment:
```
conda activate marker_transformation
```
Once the environment is activated, the pipeline can be run as described below.


## pix2pix

A dataset with paired base and target marker images needs to be created with the following folder structure:
```
├── name_of_dataset
│   ├── trainA
│   │   ...
│   ├── trainB
│   │   ...
│   ├── inferenceA
│   │   ...
│   ├── inferenceB
│   │   ...
...
```
The structure and naming is required for the files to be correctly detected. The folders `inference#` are not required for model training, but only for model inference.

Insert the base marker images in `trainA` and the corresponding target marker images in `trainB`. Copy the template settings file `./pix2pix/settings_template.hjson`, rename it to `settings.hjson` and adjust the parameters. Set an experiment name, insert the dataset path at *dataset_folder* and set the *mode* to 'train'.

**Note**: Some parameters specified in the settings file can be overwritten with command line options. See 
```python ./pix2pix/start.py --help``` for more details.

Start the training of the pix2pix model with:
```
python ./pix2pix/start.py --settings ./pix2pix/settings.hjson
```

**Note**: the GPU memory requirements for training the 3D cGAN are quite high. If memory issues occur, reduce batch size or image/crop size to (32, 128, 128). Adjust *inf_patch_size* according to the training image/crop size for optimal results.


Once the training is completed, the network can be used to generate synthetic target marker images based on base marker images. Insert base marker images in `inferenceA` located in the train dataset folder. The trained models can be found at the specified output folder. Note: the experiment name is appended by a timestamp and the epoch after which the modes were saved. Use the settings file located in the same folder as the desired model for inference.

The inference of the model can be started with:
```
python ./pix2pix/start.py --settings ./path/to/model_folder/settings.hjson --mode inference
```
Results will be placed in the model's folder.




## ResUnet

A dataset with paired base and target marker images needs to be created with the following folder structure:
```
├── name_of_dataset
│   ├── train
│   │   ├── patched_images
│   ├── vali
│   │   ├── patched_images
│   │   ...
...
```
The structure and naming is required for the files to be correctly detected.

The script `./ResUnet/create_dataset.py` can be used to generate patched training data. Copy the template settings file `./ResUnet/settings_template.json`, rename it to `settings.json` and adjust the parameters. It is important, that the setting file is located in the code directory: `./ResUnet/settings.json`. Set an experiment name, insert the dataset path at `path_data` and set the output path at `path_results`.

Start the training of the ResUnet model with:
```
python ./ResUnet/main.py
```

Once the training is completed, the network can be used to generate synthetic target marker images based on base marker images. The trained models can be found at the specified output folder. Note: the experiment name is appended by a timestamp. Use the settings file located in the same folder as the desired model for inference.

Before inference of the model, the script located in `./ResUnet/test_apply.py` needs to be adapted. Set the model_folder and path_list to the desired model and the input image folder. The inference of the model can then be started with the script:
```
python ./ResUnet/test_apply.py
```
Results will be placed in the model's folder.
