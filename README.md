# ModelScope text2video Extension for AUTOMATIC1111's StableDiffusion WebUI

## WIP!!!

Auto1111 extension consisting of implementation of ModelScope text2video using only Auto1111 webui dependencies.

8gbs vram should be enough to run on GPU with low vram vae on at 256x256. (some opts are not working properly rn) But, 24 frames length 256x256 video definitely fits into 12gbs of NVIDIA GeForce RTX 2080 Ti.

Prompt: `flowers turning into lava`

Test example:

https://user-images.githubusercontent.com/14872007/226214023-2d3892d8-64d4-4312-baab-575aafedae09.mp4


---

HuggingFace space:

https://huggingface.co/spaces/damo-vilab/modelscope-text-to-video-synthesis

All the parts of the model at HuggingFace:

https://huggingface.co/damo-vilab/modelscope-damo-text-to-video-synthesis/tree/main

The model PyTorch implementation from ModelScope:

https://github.com/modelscope/modelscope/tree/master/modelscope/models/multi_modal/video_synthesis

Google Colab from the devs:

https://colab.research.google.com/drive/1uW1ZqswkQ9Z9bp5Nbo5z59cAn7I0hE6R?usp=sharing
