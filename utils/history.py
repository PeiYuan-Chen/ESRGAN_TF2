"""
history = {
    'iteration': [],
    'loss': [],
    'val_loss': [],
    'val_psnr': [],
    'val_ssim': [],
    'best_iteration': 0,
    'best_val_psnr': float('inf'),
}
"""
import os
import json
import matplotlib.pyplot as plt


def is_history_complete(history):
    keys = ['iteration', 'loss', 'val_loss', 'val_psnr',
            'val_ssim', 'best_iteration', 'best_val_psnr']
    for key in keys:
        if key not in history or not history[key]:
            return False
    return True


def is_gan_history_complete(history):
    keys = ['iteration', 'gen_loss', 'disc_loss']
    for key in keys:
        if key not in history or not history[key]:
            return False
    return True


def create_or_continue_history(history_file):
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            history = json.load(f)
            if is_history_complete(history):
                start_iteration = history['iteration'][-1]
                print(
                    f'Restoring from latest history,start from iteration{start_iteration}')
            else:
                print('No history found, recording from scratch')
                history = {
                    'iteration': [],
                    'loss': [],
                    'val_loss': [],
                    'val_psnr': [],
                    'val_ssim': [],
                    'best_iteration': 0,
                    'best_val_psnr': float('inf'),
                }
                with open(history_file, 'w') as f:
                    json.dump(history, f)
                start_iteration = 0

    else:
        print('No history found, recording from scratch')
        history = {
            'iteration': [],
            'loss': [],
            'val_loss': [],
            'val_psnr': [],
            'val_ssim': [],
            'best_iteration': 0,
            'best_val_psnr': float('inf'),
        }
        with open(history_file, 'w') as f:
            json.dump(history, f)
        start_iteration = 0
    return history, start_iteration


def create_or_continue_gan_history(history_file):
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            history = json.load(f)
            if is_gan_history_complete(history):
                start_iteration = history['iteration'][-1]
                print(
                    f'Restoring from latest history,start from iteration{start_iteration}')
            else:
                print('No history found, recording from scratch')
                history = {
                    'iteration': [],
                    'gen_loss': [],
                    'disc_loss': [],
                }
                with open(history_file, 'w') as f:
                    json.dump(history, f)
                start_iteration = 0

    else:
        print('No history found, recording from scratch')
        history = {
            'iteration': [],
            'gen_loss': [],
            'disc_loss': [],
        }
        with open(history_file, 'w') as f:
            json.dump(history, f)
        start_iteration = 0
    return history, start_iteration


def save_history(history, history_file):
    with open(history_file, 'w') as f:
        json.dump(history, f)


def plot_history(history):
    # plot loss and metrics after train_val
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(history['val_psnr'], label='Validation PSNR')
    plt.xlabel('Iteration')
    plt.ylabel('PSNR')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(history['val_ssim'], label='Validation SSIM')
    plt.xlabel('Iteration')
    plt.ylabel('SSIM')
    plt.legend()
    plt.savefig()
