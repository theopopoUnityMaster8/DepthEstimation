"""Compute depth maps for video.
"""
import os
import glob
import torch
import utils
import cv2
import argparse

from torchvision.transforms import Compose
from midas.dpt_depth import DPTDepthModel
from midas.midas_net import MidasNet
from midas.midas_net_custom import MidasNet_small
from midas.transforms import Resize, NormalizeImage, PrepareForNet

def init(model_path, model_type, optimize=True):
    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: %s" % device)

    # load network
    if model_type == "dpt_large": # DPT-Large
        model = DPTDepthModel(
            path=model_path,
            backbone="vitl16_384",
            non_negative=True,
        )
        net_w, net_h = 384, 384
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif model_type == "dpt_hybrid": #DPT-Hybrid
        model = DPTDepthModel(
            path=model_path,
            backbone="vitb_rn50_384",
            non_negative=True,
        )
        net_w, net_h = 384, 384
        resize_mode="minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif model_type == "midas_v21":
        model = MidasNet(model_path, non_negative=True)
        net_w, net_h = 384, 384
        resize_mode="upper_bound"
        normalization = NormalizeImage(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    elif model_type == "midas_v21_small":
        model = MidasNet_small(model_path, features=64, backbone="efficientnet_lite3", exportable=True, non_negative=True, blocks={'expand': True})
        net_w, net_h = 256, 256
        resize_mode="upper_bound"
        normalization = NormalizeImage(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    else:
        print(f"model_type '{model_type}' not implemented, use: --model_type large")
        assert False
    
    transform = Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method=resize_mode,
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet(),
        ]
    )

    model.eval()
    
    if optimize==True:
        # rand_example = torch.rand(1, 3, net_h, net_w)
        # model(rand_example)
        # traced_script_module = torch.jit.trace(model, rand_example)
        # model = traced_script_module
    
        if device == torch.device("cuda"):
            model = model.to(memory_format=torch.channels_last)  
            model = model.half()

    model.to(device)
    
    print("Model ready")
    
    return device, transform, model

def start(device, transform, model, optimize=True):

    # create output folder
    output_path = 'output'
    os.makedirs(output_path, exist_ok=True)

    print("start processing")
    
    cam = cv2.VideoCapture(0)

    #Boucle sur l'ensemble des images d'entrées (traitement 1par1)
    while True:
        
        ret, img_cam = cam.read()
        img = utils.prepare_image(img_cam)
        img_input = transform({"image": img})["image"]

        # compute
        with torch.no_grad():
            sample = torch.from_numpy(img_input).to(device).unsqueeze(0)
            if optimize==True and device == torch.device("cuda"):
                sample = sample.to(memory_format=torch.channels_last)  
                sample = sample.half()
            prediction = model.forward(sample)
            # prediction = (
            #     torch.nn.functional.interpolate(
            #         prediction.unsqueeze(1),
            #         size=img.shape[:2],
            #         mode="bicubic",
            #         align_corners=False,
            #     )
            #     .squeeze()
            #     .cpu()
            #     .numpy()
            # )
            prediction = prediction.squeeze().cpu().numpy()
        #Prediction Array f32
        # output
        # filename = os.path.join(
        #     output_path, os.path.splitext(os.path.basename('video'))[0]
        # )
        # utils.write_depth(filename, prediction, bits=2) #Save 
        # img_depth = cv2.imread('output/video.png')
        
        img_depth = utils.read_depth(prediction, bits=2)
        if ret:
            cv2.imshow('ORIGIN', img_cam)
            cv2.imshow('DEPTH', img_depth)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

    print("finished")
    cv2.destroyAllWindows()
