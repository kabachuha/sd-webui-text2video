# ModelScope text2video Extension for AUTOMATIC1111's StableDiffusion WebUI

Auto1111 extension consisting of implementation of ModelScope text2video using only Auto1111 webui dependencies and downloadable models (so no logins required anywhere)

8gbs vram should be enough to run on GPU with low vram vae on at 256x256 (some opts are not working properly rn, however we are already getting reports of people launching 192x192 videos with 4gbs of vram). 24 frames length 256x256 video definitely fits into 12gbs of NVIDIA GeForce RTX 2080 Ti. We will appreciate *any* help with this extension, *especially* pull-requests.

There is a known issue with ffmpeg stitching, if ffmpeg fails and it outputs something like 'tuple split failed', go to 'stable-diffusion-webui/outputs/img2img-images/text2video-modelscope' and grab the frames from there until it's fixed.

Test examples:

Prompt: `flowers turning into lava`

https://user-images.githubusercontent.com/14872007/226214023-2d3892d8-64d4-4312-baab-575aafedae09.mp4

Prompt: `cinematic explosion by greg rutkowski`

https://user-images.githubusercontent.com/14872007/226345611-a1f0601f-db32-41bd-b983-80d363eca4d5.mp4

Prompt: `really attractive anime girl skating, by makoto shinkai, cinematic lighting`

https://user-images.githubusercontent.com/14872007/226468406-ce43fa0c-35f2-4625-a892-9fb3411d96bb.mp4


## Where to get the weights

Download the following files from [HuggingFace](https://huggingface.co/damo-vilab/modelscope-damo-text-to-video-synthesis/tree/main):
- VQGAN_autoencoder.pth
- configuration.json
- open_clip_pytorch_model.bin
- text2video_pytorch_model.pth

And put them in `stable-diffusion-webui/models/ModelScope/t2v`. Create those 2 folders if they are missing. 

## Screenshots

![Screenshot 2023-03-20 at 15-52-21 Stable Diffusion](https://user-images.githubusercontent.com/14872007/226345377-bad6dda5-f921-4233-b832-843e78854cbb.png)

![Screenshot 2023-03-20 at 15-52-15 Stable Diffusion](https://user-images.githubusercontent.com/14872007/226345398-d37133a8-3e5f-43f3-ae13-37dc609cd14c.png)


## Dev resources

HuggingFace space:

https://huggingface.co/spaces/damo-vilab/modelscope-text-to-video-synthesis

The model PyTorch implementation from ModelScope:

https://github.com/modelscope/modelscope/tree/master/modelscope/models/multi_modal/video_synthesis

Google Colab from the devs:

https://colab.research.google.com/drive/1uW1ZqswkQ9Z9bp5Nbo5z59cAn7I0hE6R?usp=sharing
