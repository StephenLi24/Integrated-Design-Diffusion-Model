#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2023/6/15 17:12
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import os
import logging
import shutil
import time

import coloredlogs
import torch
import torchvision

from PIL import Image
from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)
coloredlogs.install(level="INFO")


def plot_images(images, fig_size=(64, 64)):
    """
    Draw images
    :param images: Image
    :param fig_size: Draw image size
    :return: None
    """
    plt.figure(figsize=fig_size)
    plt.imshow(X=torch.cat([torch.cat([i for i in images.cpu()], dim=-1), ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()


def plot_one_image_in_images(images, fig_size=(64, 64)):
    """
    Draw one image in images
    :param images: Image
    :param fig_size: Draw image size
    :return: None
    """
    plt.figure(figsize=fig_size)
    for i in images.cpu():
        plt.imshow(X=i)
        plt.show()


def save_images(images, path, **kwargs):
    """
    Save images
    :param images: Image
    :param path: Save path
    :param kwargs: Other parameters
    :return: None
    """
    grid = torchvision.utils.make_grid(tensor=images, **kwargs)
    image_array = grid.permute(1, 2, 0).to("cpu").numpy()
    im = Image.fromarray(obj=image_array)
    im.save(fp=path)


def save_one_image_in_images(images, path, generate_name, image_size=None, image_format="jpg", **kwargs):
    """
    Save one image in images
    :param images: Image
    :param generate_name: generate image name
    :param path: Save path
    :param image_size: Resize image size
    :param image_format: Format of the output image
    :param kwargs: Other parameters
    :return: None
    """
    # This is counter
    count = 0
    # Show image in images
    for i in images.cpu():
        grid = torchvision.utils.make_grid(tensor=i, **kwargs)
        image_array = grid.permute(1, 2, 0).to("cpu").numpy()
        im = Image.fromarray(obj=image_array)
        # Rename every images
        im.save(fp=os.path.join(path, f"{generate_name}_{count}.{image_format}"))
        if image_size is not None:
            logger.info(msg=f"Image is resizing {image_size}.")
            # Resize
            # TODO: Super-resolution algorithm replacement
            im = im.resize(size=image_size, resample=Image.LANCZOS)
            im.save(fp=os.path.join(path, f"{generate_name}_{image_size}_{count}.{image_format}"))
        count += 1


def setup_logging(save_path, run_name):
    """
    Set log saving path
    :param save_path: Saving path
    :param run_name: Saving name
    :return: List of file paths
    """
    results_root_dir = save_path
    results_dir = os.path.join(save_path, run_name)
    results_vis_dir = os.path.join(save_path, run_name, "vis")
    results_tb_dir = os.path.join(save_path, run_name, "tensorboard")
    # Root folder
    os.makedirs(name=results_root_dir, exist_ok=True)
    # Saving folder
    os.makedirs(name=results_dir, exist_ok=True)
    # Visualization folder
    os.makedirs(name=results_vis_dir, exist_ok=True)
    # Visualization folder for Tensorboard
    os.makedirs(name=results_tb_dir, exist_ok=True)
    return [results_root_dir, results_dir, results_vis_dir, results_tb_dir]


def delete_files(path):
    """
    Clear files
    :param path: Path
    :return: None
    """
    if os.path.exists(path):
        if os.path.isfile(path):
            os.remove(path)
        else:
            shutil.rmtree(path=path)
        logger.info(msg=f"Folder '{path}' deleted.")
    else:
        logger.warning(msg=f"Folder '{path}' does not exist.")


def save_train_logging(arg, save_path):
    """
    Save train log
    :param arg: Argparse
    :param save_path: Save path
    :return: None
    """
    with open(file=f"{save_path}/train.log", mode="a") as f:
        current_time = time.strftime("%H:%M:%S", time.localtime())
        f.write(f"{current_time}: {arg}\n")
    f.close()


def check_and_create_dir(path):
    """
    Check and create not exist folder
    :param path: Create path
    :return: None
    """
    logger.info(msg=f"Check and create folder '{path}'.")
    os.makedirs(name=path, exist_ok=True)
