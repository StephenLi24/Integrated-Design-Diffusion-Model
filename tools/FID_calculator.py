#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2024/4/22 18:07
    @Author : egoist945402376
    @Site   : https://github.com/chairc
"""
import subprocess


def run_fid_command(generated_image_folder, dataset_image_folder, dim=2048):

    command = f'python -m pytorch_fid {generated_image_folder} {dataset_image_folder} --dim={dim}'

    try:

        result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)

        print("Successfully calculate FID score!")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("Failed to calculate the FID score:")
        print(e.stderr)


# Modify your path to dataset images folder and generated images folder.
# generated_image_folder = 'path_to_generated_image_folder'
# dataset_image_folder = 'path_to_dataset_image_folder'
generated_image_folder = '/home/llb/Integrated-Design-Diffusion-Model/results/cifar_horse_disentangle1/vis/1733382796.0701885'
dataset_image_folder = '/home/llb/Integrated-Design-Diffusion-Model/datasets/cifar10_folder/train_images/'



# dimensions options: 768/ 2048
# Choose 768 for fast calculation and reduced memory requirement
# choose 2048 for better calculation
# default: 2048
dimensions = 2048
# run
run_fid_command(generated_image_folder, dataset_image_folder, dimensions)
