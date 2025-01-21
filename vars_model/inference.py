#!/usr/bin/env python

from PyQt5 import QtCore
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import * 
from PyQt5.QtGui import * 
from PyQt5.QtCore import *
import pandas as pd
import json
import os
from moviepy.editor import *
from moviepy.config import get_setting
import torch
from torchvision.io.video import read_video
import torch.nn as nn
from torchvision.models.video import MViT_V2_S_Weights
from model import MVNetwork

# Load model
model = MVNetwork(net_name="mvit_v2_s", agr_type="attention")
rootdir = os.getcwd()
path = os.path.join(rootdir, '14_model.pth.tar')
path =path.replace('\\', '/' )

        # Load weights
load = torch.load(path, map_location=torch.device('cpu'))
model.load_state_dict(load['state_dict'])
model.eval()
softmax = nn.Softmax(dim=1)

video, _, _ = read_video(files[num_view], output_format="THWC")
final_frames = None
transforms_model = MViT_V2_S_Weights.KINETICS400_V1.transforms()
