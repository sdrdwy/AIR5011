from diffusers import StableDiffusionInstructPix2PixPipeline,StableDiffusionPipeline
import torch
from h5reader import *


model_path = "p2p_test_2"  
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.float16,  
).to("cuda")

# model_path = "stable-diffusion-v1-5/stable-diffusion-v1-5"
# pipe = StableDiffusionPipeline.from_pretrained(
#     model_path,
#     torch_dtype=torch.float16, 
# ).to("cuda")


path = "processed_data/blocks_stack_easy_D435_pkl_95/episode_9.hdf5"
data = RoboTwinH5(path)
width, height = 256, 256
ind = data.size//2
img = cv2.resize(data.fetch_pair(ind)["cam_high"],(256,256))
prompt = "stack the blocks together"
ground_truth = cv2.resize(data.fetch_pair(min(ind+50,data.size))["cam_high"],(256,256))

guidance_scale = 7.5
num_inference_steps = 50
image_guidance_scale = 1.5  

edited_images = pipe(
    prompt=prompt,
    image=img,
    guidance_scale=guidance_scale,
    num_inference_steps=num_inference_steps,
    image_guidance_scale=image_guidance_scale,
)
edited_image = np.array(edited_images.images[0])
cv2.imwrite("begin.jpg",img)
cv2.imwrite("end.jpg",edited_image)
cv2.imwrite("ground_truth.jpg",ground_truth)
