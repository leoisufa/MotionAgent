# -*- coding:utf-8 -*-
import os
import sys
import shutil
import logging
import colorlog
from tqdm import tqdm
import time
import yaml
import random
import importlib
from PIL import Image
from warnings import simplefilter
import imageio
import math
import collections
import json
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, Dataset
from einops import rearrange, repeat
import torch.distributed as dist
from torchvision import datasets, transforms, utils

logging.getLogger().setLevel(logging.WARNING)
simplefilter(action='ignore', category=FutureWarning)

def get_logger(filename=None):
    """
    examples:
        logger = get_logger('try_logging.txt')

        logger.debug("Do something.")
        logger.info("Start print log.")
        logger.warning("Something maybe fail.")
        try:
            raise ValueError()
        except ValueError:
            logger.error("Error", exc_info=True)

        tips:
        DO NOT logger.inf(some big tensors since color may not helpful.)
    """
    logger = logging.getLogger('utils')
    level = logging.DEBUG
    logger.setLevel(level=level)
    # Use propagate to avoid multiple loggings.
    logger.propagate = False
    # Remove %(levelname)s since we have colorlog to represent levelname.
    format_str = '[%(asctime)s <%(filename)s:%(lineno)d> %(funcName)s] %(message)s'

    streamHandler = logging.StreamHandler()
    streamHandler.setLevel(level)
    coloredFormatter = colorlog.ColoredFormatter(
        '%(log_color)s' + format_str,
        datefmt='%Y-%m-%d %H:%M:%S',
        reset=True,
        log_colors={
            'DEBUG': 'cyan',
            # 'INFO': 'white',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'reg,bg_white',
        }
    )

    streamHandler.setFormatter(coloredFormatter)
    logger.addHandler(streamHandler)

    if filename:
        fileHandler = logging.FileHandler(filename)
        fileHandler.setLevel(level)
        formatter = logging.Formatter(format_str)
        fileHandler.setFormatter(formatter)
        logger.addHandler(fileHandler)

    # Fix multiple logging for torch.distributed
    try:
        class UniqueLogger:
            def __init__(self, logger):
                self.logger = logger
                self.local_rank = torch.distributed.get_rank()

            def info(self, msg, *args, **kwargs):
                if self.local_rank == 0:
                    return self.logger.info(msg, *args, **kwargs)

            def warning(self, msg, *args, **kwargs):
                if self.local_rank == 0:
                    return self.logger.warning(msg, *args, **kwargs)

        logger = UniqueLogger(logger)
    # AssertionError for gpu with no distributed
    # AttributeError for no gpu.
    except Exception:
        pass
    return logger


logger = get_logger()

def split_filename(filename):
    absname = os.path.abspath(filename)
    dirname, basename = os.path.split(absname)
    split_tmp = basename.rsplit('.', maxsplit=1)
    if len(split_tmp) == 2:
        rootname, extname = split_tmp
    elif len(split_tmp) == 1:
        rootname = split_tmp[0]
        extname = None
    else:
        raise ValueError("programming error!")
    return dirname, rootname, extname

def data2file(data, filename, type=None, override=False, printable=False, **kwargs):
    dirname, rootname, extname = split_filename(filename)
    print_did_not_save_flag = True
    if type:
        extname = type
    if not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)

    if not os.path.exists(filename) or override:
        if extname in ['jpg', 'png', 'jpeg']:
            utils.save_image(data, filename, **kwargs)
        elif extname == 'gif':
            imageio.mimsave(filename, data, format='GIF', duration=kwargs.get('duration'), loop=0)
        elif extname == 'txt':
            if kwargs is None:
                kwargs = {}
            max_step = kwargs.get('max_step')
            if max_step is None:
                max_step = np.Infinity

            with open(filename, 'w', encoding='utf-8') as f:
                for i, e in enumerate(data):
                    if i < max_step:
                        f.write(str(e) + '\n')
                    else:
                        break
        else:
            raise ValueError('Do not support this type')
        if printable: logger.info('Saved data to %s' % os.path.abspath(filename))
    else:
        if print_did_not_save_flag: logger.info(
            'Did not save data to %s because file exists and override is False' % os.path.abspath(
                filename))


def file2data(filename, type=None, printable=True, **kwargs):
    dirname, rootname, extname = split_filename(filename)
    print_load_flag = True
    if type:
        extname = type
    
    if extname in ['pth', 'ckpt']:
        data = torch.load(filename, map_location=kwargs.get('map_location'))
    elif extname == 'txt':
        top = kwargs.get('top', None)
        with open(filename, encoding='utf-8') as f:
            if top:
                data = [f.readline() for _ in range(top)]
            else:
                data = [e for e in f.read().split('\n') if e]
    elif extname == 'yaml':
        with open(filename, 'r') as f:
            data = yaml.load(f)
    else:
        raise ValueError('type can only support h5, npy, json, txt')
    if printable:
        if print_load_flag:
            logger.info('Loaded data from %s' % os.path.abspath(filename))
    return data


def ensure_dirname(dirname, override=False):
    if os.path.exists(dirname) and override:
        logger.info('Removing dirname: %s' % os.path.abspath(dirname))
        try:
            shutil.rmtree(dirname)
        except OSError as e:
            raise ValueError('Failed to delete %s because %s' % (dirname, e))

    if not os.path.exists(dirname):
        logger.info('Making dirname: %s' % os.path.abspath(dirname))
        os.makedirs(dirname, exist_ok=True)


def import_filename(filename):
    spec = importlib.util.spec_from_file_location("mymodule", filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def adaptively_load_state_dict(target, state_dict):
    target_dict = target.state_dict()

    try:
        common_dict = {k: v for k, v in state_dict.items() if k in target_dict and v.size() == target_dict[k].size()}
    except Exception as e:
        logger.warning('load error %s', e)
        common_dict = {k: v for k, v in state_dict.items() if k in target_dict}

    if 'param_groups' in common_dict and common_dict['param_groups'][0]['params'] != \
            target.state_dict()['param_groups'][0]['params']:
        logger.warning('Detected mismatch params, auto adapte state_dict to current')
        common_dict['param_groups'][0]['params'] = target.state_dict()['param_groups'][0]['params']
    target_dict.update(common_dict)
    target.load_state_dict(target_dict)

    missing_keys = [k for k in target_dict.keys() if k not in common_dict]
    unexpected_keys = [k for k in state_dict.keys() if k not in common_dict]

    if len(unexpected_keys) != 0:
        logger.warning(
            f"Some weights of state_dict were not used in target: {unexpected_keys}"
        )
    if len(missing_keys) != 0:
        logger.warning(
            f"Some weights of state_dict are missing used in target {missing_keys}"
        )
    if len(unexpected_keys) == 0 and len(missing_keys) == 0:
        logger.warning("Strictly Loaded state_dict.")

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def image2pil(filename):
    return Image.open(filename)


def image2arr(filename):
    pil = image2pil(filename)
    return pil2arr(pil)


# 格式转换
def pil2arr(pil):
    if isinstance(pil, list):
        arr = np.array(
            [np.array(e.convert('RGB').getdata(), dtype=np.uint8).reshape(e.size[1], e.size[0], 3) for e in pil])
    else:
        arr = np.array(pil)
    return arr

def arr2pil(arr):
    if arr.ndim == 3:
        return Image.fromarray(arr.astype('uint8'), 'RGB')
    elif arr.ndim == 4:
        return [Image.fromarray(e.astype('uint8'), 'RGB') for e in list(arr)]
    else:
        raise ValueError('arr must has ndim of 3 or 4, but got %s' % arr.ndim)

def notebook_show(*images):
    from IPython.display import Image
    from IPython.display import display
    display(*[Image(e) for e in images])

def calc_cam_cone_pts_3d(R_W2C, T_W2C, fov_deg, scale=0.1, set_canonical=False, first_frame_RT=None):
    fov_rad = np.deg2rad(fov_deg)
    R_W2C_inv = np.linalg.inv(R_W2C)

    # Camera pose center:
    T = np.zeros_like(T_W2C) - T_W2C
    T = np.dot(R_W2C_inv, T)
    cam_x = T[0]
    cam_y = T[1]
    cam_z = T[2]
    if set_canonical:
        T = np.zeros_like(T_W2C)
        T = np.dot(first_frame_RT[:,:3], T) + first_frame_RT[:,-1]
        T = T - T_W2C 
        T = np.dot(R_W2C_inv, T)
        cam_x = T[0]
        cam_y = T[1]
        cam_z = T[2]

    # vertex
    corn1 = np.array([np.tan(fov_rad / 2.0), 0.5*np.tan(fov_rad / 2.0), 1.0]) *scale 
    corn2 = np.array([-np.tan(fov_rad / 2.0), 0.5*np.tan(fov_rad / 2.0), 1.0]) *scale
    corn3 = np.array([0, -0.25*np.tan(fov_rad / 2.0), 1.0]) *scale
    corn4 = np.array([0, -0.5*np.tan(fov_rad / 2.0), 1.0]) *scale

    corn1 = corn1 - T_W2C
    corn2 = corn2 - T_W2C
    corn3 = corn3 - T_W2C
    corn4 = corn4 - T_W2C
    
    corn1 = np.dot(R_W2C_inv, corn1)
    corn2 = np.dot(R_W2C_inv, corn2)
    corn3 = np.dot(R_W2C_inv, corn3) 
    corn4 = np.dot(R_W2C_inv, corn4) 

    # Now attach as offset to actual 3D camera position:
    corn_x1 = corn1[0]
    corn_y1 = corn1[1]
    corn_z1 = corn1[2]
    
    corn_x2 = corn2[0]
    corn_y2 = corn2[1]
    corn_z2 = corn2[2]
    
    corn_x3 = corn3[0]
    corn_y3 = corn3[1]
    corn_z3 = corn3[2]
    
    corn_x4 = corn4[0]
    corn_y4 = corn4[1]
    corn_z4 = corn4[2]
            

    xs = [cam_x, corn_x1, corn_x2, corn_x3, corn_x4, ]
    ys = [cam_y, corn_y1, corn_y2, corn_y3, corn_y4, ]
    zs = [cam_z, corn_z1, corn_z2, corn_z3, corn_z4, ]

    return np.array([xs, ys, zs]).T



    # T_base = [
    #             [1.,0.,0.],             ## W2C  x 的正方向： 相机朝左  left
    #             [-1.,0.,0.],            ## W2C  x 的负方向： 相机朝右  right
    #             [0., 1., 0.],           ## W2C  y 的正方向： 相机朝上  up     
    #             [0.,-1.,0.],            ## W2C  y 的负方向： 相机朝下  down
    #             [0.,0.,1.],             ## W2C  z 的正方向： 相机往前  zoom out
    #             [0.,0.,-1.],            ## W2C  z 的负方向： 相机往前  zoom in
    #         ]   
    # radius = 1
    # n = 16
    # # step = 
    # look_at = np.array([0, 0, 0.8]).reshape(3,1)
    # # look_at = np.array([0, 0, 0.2]).reshape(3,1)

    # T_list = []
    # base_R = np.array([[1., 0., 0.],
    #                 [0., 1., 0.],
    #                 [0., 0., 1.]])
    # res = [] 
    # res_forsave = []
    # T_range = 1.8



    # for i in range(0, 16):
    #     # theta = (1)*np.pi*i/n

    #     R = base_R[:,:3]
    #     T = np.array([0.,0.,1.]).reshape(3,1) * (i/n)*2
    #     RT = np.concatenate([R,T], axis=1)
    #     res.append(RT)
        
    # fig = vis_camera(res)

import plotly.graph_objects as go
import plotly.express as px
def vis_camera(RT_list, rescale_T=1):
    fig = go.Figure()
    showticklabels = True
    visible = True
    scene_bounds = 2
    base_radius = 2.5
    zoom_scale = 1.5
    fov_deg = 50.0
    
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (2, 3), (3, 1), (3, 4)] 
    
    colors = px.colors.qualitative.Plotly
    
    cone_list = []
    n = len(RT_list)
    for i, RT in enumerate(RT_list):
        R = RT[:,:3]
        T = RT[:,-1]/rescale_T
        cone = calc_cam_cone_pts_3d(R, T, fov_deg)
        cone_list.append((cone, (i*1/n, "green"), f"view_{i}"))

    
    for (cone, clr, legend) in cone_list:
        for (i, edge) in enumerate(edges):
            (x1, x2) = (cone[edge[0], 0], cone[edge[1], 0])
            (y1, y2) = (cone[edge[0], 1], cone[edge[1], 1])
            (z1, z2) = (cone[edge[0], 2], cone[edge[1], 2])
            fig.add_trace(go.Scatter3d(
                x=[x1, x2], y=[y1, y2], z=[z1, z2], mode='lines',
                line=dict(color=clr, width=3),
                name=legend, showlegend=(i == 0))) 
    fig.update_layout(
                    height=500,
                    autosize=True,
                    # hovermode=False,
                    margin=go.layout.Margin(l=0, r=0, b=0, t=0),
                    
                    showlegend=True,
                    legend=dict(
                        yanchor='bottom',
                        y=0.01,
                        xanchor='right',
                        x=0.99,
                    ),
                    scene=dict(
                        aspectmode='manual',
                        aspectratio=dict(x=1, y=1, z=1.0),
                        camera=dict(
                            center=dict(x=0.0, y=0.0, z=0.0),
                            up=dict(x=0.0, y=-1.0, z=0.0),
                            eye=dict(x=scene_bounds/2, y=-scene_bounds/2, z=-scene_bounds/2),
                            ),

                        xaxis=dict(
                            range=[-scene_bounds, scene_bounds],
                            showticklabels=showticklabels,
                            visible=visible,
                        ),
                            
                        
                        yaxis=dict(
                            range=[-scene_bounds, scene_bounds],
                            showticklabels=showticklabels,
                            visible=visible,
                        ),
                            
                        
                        zaxis=dict(
                            range=[-scene_bounds, scene_bounds],
                            showticklabels=showticklabels,
                            visible=visible,
                        )
                    ))
    return fig