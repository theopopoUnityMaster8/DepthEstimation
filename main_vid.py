#Main for run MiDaS model

import run_vid
import argparse
import torch


parser = argparse.ArgumentParser()

parser.add_argument('-i', '--input_path', 
    default='input',
    help='folder with input images'
    )

parser.add_argument('-o', '--output_path', 
    default='output',
    help='folder for output images'
    )
    
parser.add_argument('-m', '--model_weights', 
        default=None,
        help='path to the trained weights of model'
    )

parser.add_argument('-t', '--model_type', 
        default='midas_v21_small',
        #default = 'midas_v21',
        #default = 'dpt_hybrid',
        help='model type: dpt_large, dpt_hybrid, midas_v21_large or midas_v21_small'
    )

args = parser.parse_args()

default_models = {
        "midas_v21_small": "weights/midas_v21_small-70d6b9c8.pt",
        "midas_v21": "weights/midas_v21-f6b98070.pt",
        "dpt_large": "weights/dpt_large-midas-2f21e586.pt",
        "dpt_hybrid": "weights/dpt_hybrid-midas-501f0c75.pt",
    }

if args.model_weights is None:
    args.model_weights = default_models[args.model_type]

# set torch options
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


#cam temps réel
device, transform, model = run_vid.init(args.model_weights, args.model_type)

run_vid.start(device, transform, model)


