#!/usr/bin/env python3

import argparse
import math

from utils.pre_processing import *
from utils.training import *
from utils.models import *
from utils.properties import *

parser = argparse.ArgumentParser(description='OSM2Vec')
parser.add_argument("-c", type=str, required=True, help='City')
parser.add_argument("-e", type=int, default=config.n_epochs, help='Epochs per stage')
parser.add_argument("-m", type=int, default=int(config.m_index), help='Samples per epoch (stages 1 and 2)')
parser.add_argument("-s", type=str, default='123', help='Stages to run, in order (default 123)')
parser.add_argument("-f", type=str, default=None, help='Start from the model in this folder instead of a new one')
parser.add_argument("--reset-image", action='store_true',
                    help='Re-initialise the footprint branch (image network, image and area projections)')
parser.add_argument("--no-names", action='store_true',
                    help='Leave names out of all text, in training and when the model is used later')
parser.add_argument("--no-names-stage2", action='store_true', help='Stage 2 only: leave names out of the tag text (as v4)')

hp = parser.parse_args()
config.n_epochs = hp.e
config.m_index = hp.m
config.stage2_names = not hp.no_names_stage2
filter_w()

check_data_exists(hp.c)

if not os.path.isdir(hp.c + '/Model'):
    os.mkdir(hp.c + '/Model')

cd = CityData(hp.c)
voc = build_category_vocab(cd.entities)
n_polylines = len(set(cd.data.keys()))
model = read_model(hp.f) if hp.f else SSModel(hp.c, config.lm, 768)
if hp.no_names:
    config.names_in_text = False
    model.names_in_text = False

if hp.reset_image:
    fresh = SSModel(hp.c, config.lm, 768)
    model.cv, model.projection_img, model.projection_s = fresh.cv, fresh.projection_img, fresh.projection_s
    del fresh

if '2' in hp.s:
    model.stage2_names = config.stage2_names

stages = {'1': train_SS_im, '2': train_SS_raster, '3': train_SS_rel}
for n, s in enumerate(hp.s):
    if n:
        model = read_model(hp.c)
    stages[s](cd, model)
