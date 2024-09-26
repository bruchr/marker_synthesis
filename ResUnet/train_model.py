import os
import time

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch_optimizer as optim_
from tqdm import tqdm

import losses
from metrics import l2_dist
from optimizers import RAdam
from visualization import Visualizer

def train(net, configs, datasets, device):
    """ Train the specified model.
    
    :param net: PyTorch model to train
        :type net: model
    :datasets: Dataloader objects wich contains the training data
        :type net: dataloader object
    :param device: cuda (gpu) or cpu.
        :type device:
    :param configs: Dictionary containing data paths and information for the training and evaluation process.
        :type configs: dict
    :return: None.
    """

    # Data loader for training and validation set
    apply_shuffling = {'train': True, 'val': False}
    dataloader = {x: torch.utils.data.DataLoader(datasets[x], batch_size=configs['batch_size'], shuffle=apply_shuffling[x],
                                                 pin_memory=True, num_workers=0) for x in ['train', 'val']}

    # Loss function
    criterion = losses.get_loss(configs['loss'])

    # Optimizer
    if configs['optimizer'] == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=configs['learning_rate'], betas=(0.9, 0.999), eps=1e-08,
                               weight_decay=0, amsgrad=True)
    elif configs['optimizer'] == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=configs['learning_rate'], momentum=0.9, weight_decay=0,
                              nesterov=True)
    elif configs['optimizer'] == 'lookahead':
        base_optimizer = RAdam(net.parameters(), lr=configs['learning_rate'], betas=(0.9, 0.999), eps=1e-08,
                               weight_decay=0)
        optimizer = optim_.Lookahead(optimizer=base_optimizer, k=5, alpha=0.5)
    elif configs['optimizer'] == 'ranger':
        optimizer = optim_.Ranger(net.parameters(), lr=configs['learning_rate'], betas=(0.95, 0.999), eps=1e-08,
            alpha=0.5, k=5)
    else:
        raise Exception('Unknown optimizer')

    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.25, patience=7, verbose=True, min_lr=1e-5)

    # Auxiliary variables for training process
    epochs_without_improvement = 0
    train_crit, val_crit, train_loss, val_loss, best_loss = [], [], [], [], 1e4
    since = time.time()

    
    # initialize tensorboard writer and define loss groups
    visualizer = Visualizer(configs)
    visualizer.define_loss_group("Train", ['t_Epoch-Loss', 't_L2-Dist'])
    visualizer.define_loss_group("Vali", ['v_Epoch-Loss', 'v_L2-Dist'])

    epoch_loss_dir = {'train': 0, 'val': 0}
    epoch_crit_dir = {'train': 0, 'val': 0}

    # Training process
    for epoch in range(configs['max_epochs']):
        print('-' * 10)
        print('Epoch {}/{}'.format(epoch + 1, configs['max_epochs']))
        print('-' * 10)
        start = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()  # Set model to training mode
            else:
                net.eval()  # Set model to evaluation mode

            running_loss, running_crit = 0.0, 0.0

            # Iterate over data
            for samples in tqdm(dataloader[phase], desc=phase+': Batch-No.'):

                # Get inputs and labels and put them on GPU if available
                inputs, labels = samples
                if phase=='train':
                    inputs_train, labels_train = inputs.clone(), labels.clone()
                if phase=='val':
                    inputs_val, labels_val = inputs.clone(), labels.clone()
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass (track history if only in train)
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)

                    # Backward (optimize only if in training phase)
                    if phase == 'train':
                        # with torch.autograd.detect_anomaly():
                        loss.backward()
                        optimizer.step()

                    # Crit metric
                    crit = l2_dist(outputs, labels, device)

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_crit += crit * inputs.size(0)

            epoch_loss = running_loss / len(datasets[phase])
            epoch_crit = running_crit / len(datasets[phase])

            if phase == 'train':
                train_loss.append(epoch_loss)
                train_crit.append(epoch_crit)
                print('Training loss: {:.4f}, crit: {:.4f}'.format(epoch_loss, epoch_crit))
            else:
                val_loss.append(epoch_loss)
                val_crit.append(epoch_crit)
                print('Validation loss: {:.4f}, crit: {:.4f}'.format(epoch_loss, epoch_crit))

                scheduler.step(epoch_loss)

                if epoch_loss < best_loss:
                    print('Validation loss improved from {:.4f} to {:.4f}. Save model.'.format(best_loss, epoch_loss))
                    best_loss = epoch_loss
                    torch.save(net.state_dict(), os.path.join(configs['path_results'],'state_dict.pth'))
                    epochs_without_improvement = 0
                else:
                    print('Validation loss did not improve.')
                    epochs_without_improvement += 1

            epoch_loss_dir[phase] = epoch_loss
            epoch_crit_dir[phase] = epoch_crit

        visualizer.write_losses({
            't_Epoch-Loss': epoch_loss_dir['train'], 't_L2-Dist': epoch_crit_dir['train'],
            'v_Epoch-Loss': epoch_loss_dir['val'], 'v_L2-Dist': epoch_crit_dir['val']}, epoch)
        visualizer.write_images({'val input': inputs[0,...], 'val prediction': outputs[0,...], 'val label': labels[0,...]}, epoch)

        # Epoch training time
        print('Epoch training time: {:.0f}s'.format(time.time() - start))

        # Break training if plateau is reached
        if epochs_without_improvement == configs['break_condition']:
            print(str(epochs_without_improvement) + ' epochs without validation loss improvement --> break')
            break

    # Total training time
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    configs['training_time'], configs['trained_epochs'] = time_elapsed, epoch + 1

    # Save loss and metrics
    stats = np.transpose(np.array([list(range(1, len(train_loss) + 1)), train_loss, train_crit, val_loss, val_crit]))
    np.savetxt(fname=os.path.join(configs['path_results'],'train.txt'), X=stats,
               fmt=['%3i', '%2.5f', '%1.4f', '%2.5f', '%1.4f'],
               header='Epoch, training loss, training crit, validation loss, validation crit', delimiter=',')

    # # Clear memory
    # del net
    # gc.collect()

    return None