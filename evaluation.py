from diffusers import StableDiffusionInstructPix2PixPipeline,StableDiffusionPipeline
import torch
from h5reader import *
import random
import os
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from PIL import Image
from tqdm import tqdm
import json
from torch.utils.data import random_split
random.seed(42)

origin_pipe = StableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    torch_dtype=torch.float16, 
    safety_checker = None
).to("cuda")


model_path = "p2p_test_2"  
trained_pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.float16,  
    safety_checker = None
).to("cuda")

model_large = "p2p_test"

trained_large = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    model_large,
    torch_dtype=torch.float16,  
    safety_checker = None
).to("cuda")

dataset = FrameDataset(step = 30,begin = random.randint(0,30))
test_dataset,_ = random_split(dataset,[1000,len(dataset)-1000])
guidance_scale = 7.5
num_inference_steps = 50
image_guidance_scale = 1.5  


def load_image(img, size=(256, 256)):
    try:
        img = img.resize(size)
    except:
        pass
    img_np = np.array(img).astype(np.float32) / 255.0  # scale to [0,1]
    return img_np


def test_epoch(index):
    pair = test_dataset.__getitem__(index)
    prompt = pair['instruction']
    src = load_image(pair['src'])
    ground_truth = load_image(pair['tgt'])

    non_trained = origin_pipe(prompt = prompt,
                              image = src,
                              guidance_rescale= guidance_scale,
                              num_inference_steps=num_inference_steps,
                              
                              image_guidance_scale=image_guidance_scale).images[0]
    trained = trained_pipe(prompt = prompt,
                              image = src,
                              guidance_rescale= guidance_scale,
                              num_inference_steps=num_inference_steps,
                              image_guidance_scale=image_guidance_scale).images[0]
    
    large = trained_large(prompt = prompt,
                              image = src,
                              guidance_rescale= guidance_scale,
                              num_inference_steps=num_inference_steps,
                              image_guidance_scale=image_guidance_scale).images[0]
    record = {}
    
    record["idx"] = index
    record["prompt"] = prompt

    origin_img = load_image(non_trained)
    trained_img = load_image(trained)
    large_img = load_image(large)

    cv2.imwrite(f"test_imgs/begin_{index}.jpg",src*255)
    cv2.imwrite(f"test_imgs/gt_{index}.jpg",ground_truth*255)
    cv2.imwrite(f"test_imgs/origin_{index}.jpg",origin_img*255)
    cv2.imwrite(f"test_imgs/trained_{index}.jpg",trained_img*255)
    cv2.imwrite(f"test_imgs/large_{index}.jpg",large_img*255)

    ssim_val = ssim(ground_truth, origin_img, channel_axis=2, data_range=1.0)
    psnr_val = psnr(ground_truth, origin_img, data_range=1.0)
    record["origin_model"] = {"SSIM":float(ssim_val),"PSNR (dB)":float(psnr_val)}

    ssim_val = ssim(ground_truth, trained_img, channel_axis=2, data_range=1.0)
    psnr_val = psnr(ground_truth, trained_img, data_range=1.0)
    record["trained_model"] = {"SSIM":float(ssim_val),"PSNR (dB)":float(psnr_val)}

    ssim_val = ssim(ground_truth, large_img, channel_axis=2, data_range=1.0)
    psnr_val = psnr(ground_truth, large_img, data_range=1.0)
    record["large_model"] = {"SSIM":float(ssim_val),"PSNR (dB)":float(psnr_val)}

    return record
    
print(test_epoch(0))

summary = {"orgin":{"SSIM":0,"PSNR (dB)":0},
           "trained":{"SSIM":0,"PSNR (dB)":0},
           "large":{"SSIM":0,"PSNR (dB)":0}}
records = []
for i in range(len(test_dataset)):
    print(f"test epoch {i}/{len(test_dataset)}")
    ret = test_epoch(i)
    summary["orgin"]["SSIM"] += ret["origin_model"]["SSIM"]
    summary["orgin"]["PSNR (dB)"] += ret["origin_model"]["PSNR (dB)"]

    summary["trained"]["SSIM"] += ret["trained_model"]["SSIM"]
    summary["trained"]["PSNR (dB)"] += ret["trained_model"]["PSNR (dB)"]

    summary["large"]["SSIM"] += ret["large_model"]["SSIM"]
    summary["large"]["PSNR (dB)"] += ret["large_model"]["PSNR (dB)"]
    print(json.dumps(ret))
    records.append(ret)

length = len(test_dataset)
summary["orgin"]["SSIM"] /= length
summary["orgin"]["PSNR (dB)"] /= length

summary["trained"]["SSIM"] /= length
summary["trained"]["PSNR (dB)"] /= length

summary["large"]["SSIM"] /= length
summary["large"]["PSNR (dB)"] /= length

print(summary)


with open("testrecord.json","w+") as f:
    json.dump(records,f)

with open("summary.json","w+") as f:
    json.dump(summary,f)