#!/usr/bin/env python3

import argparse
import shutil

from utils.pre_processing import *

parser = argparse.ArgumentParser(description='Replace residential buildings in contexts with one summary entity')
parser.add_argument("-c", type=str, required=True, help='City (input folder)')
parser.add_argument("-o", type=str, required=True, help='Output folder')

hp = parser.parse_args()

src = hp.c + '/Data/'
dst = hp.o + '/Data/'
os.makedirs(dst, exist_ok=True)

with open(src + 'entities.json', encoding='utf-8') as f:
    entities = json.load(f)

with open(src + 'data.json', encoding='utf-8') as f:
    data = json.load(f)

for d in data.values():
    d['context'] = collapse_residential(d['context'], entities)

save(data, dst + 'data.json')

if os.path.isfile(src + 'buildings_context.json'):
    with open(src + 'buildings_context.json', encoding='utf-8') as f:
        b_ctx = json.load(f)

    save({b: collapse_residential(ctx, entities) for b, ctx in b_ctx.items()}, dst + 'buildings_context.json')

entities.update(residential_summaries())
save(entities, dst + 'entities.json')

for name in listdir(src):
    if not os.path.isfile(dst + name):
        shutil.copy(src + name, dst + name)
