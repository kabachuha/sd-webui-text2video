import sys, os
from scripts.key_frames import T2VAnimKeys #TODO: move to deforum_tools
basedirs = [os.getcwd()]
if 'google.colab' in sys.modules:
    basedirs.append('/content/gdrive/MyDrive/sd/stable-diffusion-webui') #hardcode as TheLastBen's colab seems to be the primal source

for basedir in basedirs:
    deforum_paths_to_ensure = [basedir + '/extensions/sd-webui-text2video/scripts', basedir + '/extensions/sd-webui-modelscope-text2video/scripts', basedir]

    for deforum_scripts_path_fix in deforum_paths_to_ensure:
        if not deforum_scripts_path_fix in sys.path:
            sys.path.extend([deforum_scripts_path_fix])

# See https://github.com/modelscope/modelscope/tree/master/modelscope/pipelines/multi_modal
import cv2
import gc
import random
import subprocess
import time, math
from PIL import Image
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm
from base64 import b64encode
from types import SimpleNamespace
import gradio as gr
from webui import wrap_gradio_gpu_call
import modules.paths as ph
from modules import devices, lowvram, script_callbacks, sd_hijack, shared
from modules.shared import cmd_opts, opts, state
from scripts.error_hardcode import get_error
from modelscope.t2v_pipeline import TextToVideoSynthesis, tensor2vid
from scripts.video_audio_utils import ffmpeg_stitch_video, find_ffmpeg_binary, get_quick_vid_info, vid2frames, duplicate_pngs_from_folder, clean_folder_name

outdir = os.path.join(opts.outdir_img2img_samples, 'text2video')
outdir = os.path.join(os.getcwd(), outdir)

pipe = None

def setup_pipeline():
    return TextToVideoSynthesis(ph.models_path+'/ModelScope/t2v')

i1_store_t2v = f"<p style=\"text-align:center;font-weight:bold;margin-bottom:0em\">text2video extension for auto1111 — version 1.0b. The video will be shown below this label when ready</p>"

welcome_text_videocrafter = '''
    Download pretrained T2V models via this link https://drive.google.com/file/d/13ZZTXyAKM3x0tObRQOQWdtnrI2ARWYf_/view?usp=share_link, and put the model.ckpt in models/VideoCrafter/model.ckpt.
    Then use the same GUI pipeline as ModelScope does.
'''

welcome_text_modelscope = '''Put your models from <a style="color:SteelBlue" href="https://huggingface.co/damo-vilab/modelscope-damo-text-to-video-synthesis/tree/main">https://huggingface.co/damo-vilab/modelscope-damo-text-to-video-synthesis/tree/main</a> to stable-diffusion-webui/models/ModelScope/t2v/. Make sure, you downloaded the file named 'configuration.json' in its raw text form (click on the ⬇️ character to the right, don't save via right-click).

8gbs of VRAM on top of SD should be enough to launch (when the VAE unloading will be fixed, before that orient around ~12 gbs).

Join the development or report issues and feature requests here <a style="color:SteelBlue" href="https://github.com/deforum-art/sd-webui-modelscope-text2video">https://github.com/deforum-art/sd-webui-modelscope-text2video</a>

<italic>If you liked this extension, please <a style="color:SteelBlue" href="https://github.com/deforum-art/sd-webui-modelscope-text2video">give it a star on GitHub</a>!</italic> 😊

'''

welcome_text = '''VideoCrafter:

''' + welcome_text_videocrafter + '''

ModelScope:

''' + welcome_text_modelscope
import traceback

def process(skip_video_creation, ffmpeg_location, ffmpeg_crf, ffmpeg_preset, fps, add_soundtrack, soundtrack_path, \
                prompt, n_prompt, steps, frames, seed, cfg_scale, width, height, eta, \
                prompt_v, n_prompt_v, steps_v, frames_v, seed_v, cfg_scale_v, width_v, height_v, eta_v, batch_count_v=1, \
                batch_count=1, do_img2img=False, img2img_frames=None, img2img_frames_path="", strength=0,img2img_startFrame=0, \
                inpainting_image=None,inpainting_frames=0,inpainting_weights="", \
                model_type='ModelScope',
            ):
    
    # weird PATH stuff
    for basedir in basedirs:
            sys.path.extend([
                basedir + '/scripts',
                basedir + '/extensions/sd-webui-text2video/scripts',
                basedir + '/extensions/sd-webui-modelscope-text2video/scripts',
            ])

    global pipe
    global i1_store_t2v
    dataurl = get_error()
    i1_store_t2v = f'<p style=\"font-weight:bold;margin-bottom:0em\">text2video extension for auto1111 — version 1.1b </p><video controls loop><source src="{dataurl}" type="video/mp4"></video>'
    keep_pipe_in_vram = opts.data.get("modelscope_deforum_keep_model_in_vram") if opts.data is not None and opts.data.get("modelscope_deforum_keep_model_in_vram") is not None else False
    try:
        print('text2video — The model selected is: ', model_type)
        if model_type == 'ModelScope':
            process_modelscope(skip_video_creation, ffmpeg_location, ffmpeg_crf, ffmpeg_preset, fps, add_soundtrack, soundtrack_path, \
                    prompt, n_prompt, steps, frames, seed, cfg_scale, width, height, eta, \
                    prompt_v, n_prompt_v, steps_v, frames_v, seed_v, cfg_scale_v, width_v, height_v, eta_v, batch_count_v, \
                    batch_count, do_img2img, img2img_frames, img2img_frames_path, strength,img2img_startFrame, \
                    inpainting_image,inpainting_frames,inpainting_weights,)
        elif model_type == 'VideoCrafter':
            process_videocrafter(skip_video_creation, ffmpeg_location, ffmpeg_crf, ffmpeg_preset, fps, add_soundtrack, soundtrack_path, \
                    prompt, n_prompt, steps, frames, seed, cfg_scale, width, height, eta, \
                    prompt_v, n_prompt_v, steps_v, frames_v, seed_v, cfg_scale_v, width_v, height_v, eta_v, batch_count_v, \
                    batch_count, do_img2img, img2img_frames, img2img_frames_path, strength,img2img_startFrame, \
                    inpainting_image,inpainting_frames,inpainting_weights,)
        else:
            raise NotImplementedError(f"Unknown model type: {model_type}")
    except Exception as e:
        traceback.print_exc()
        print('Exception occurred:', e)
    # except Exception as e:
        # print('Exception occured')
        # print(e)
    finally:
        #optionally store pipe in global between runs, if not, remove it
        if not keep_pipe_in_vram:
            pipe = None
        devices.torch_gc()
        gc.collect()
    return f'Video at ready!'

def process_modelscope(skip_video_creation, ffmpeg_location, ffmpeg_crf, ffmpeg_preset, fps, add_soundtrack, soundtrack_path, \
                prompt, n_prompt, steps, frames, seed, cfg_scale, width, height, eta, \
                prompt_v, n_prompt_v, steps_v, frames_v, seed_v, cfg_scale_v, width_v, height_v, eta_v, batch_count_v=1, \
                 batch_count=1, do_img2img=False, img2img_frames=None, img2img_frames_path="", strength=0,img2img_startFrame=0, \
                 inpainting_image=None,inpainting_frames=0,inpainting_weights="",
            ):
    
    global pipe
    print(f"\033[4;33m text2video extension for auto1111 webui\033[0m")
    print(f"Git commit: {get_t2v_version()}")
    global i1_store_t2v
    init_timestring = time.strftime('%Y%m%d%H%M%S')
    outdir_current = os.path.join(outdir, f"{init_timestring}")
    
    cpu_vae = opts.data.get("modelscope_deforum_vae_settings") if opts.data is not None and opts.data.get("modelscope_deforum_vae_settings") is not None else 'GPU (half precision)'
    if shared.sd_model is not None:
        sd_hijack.model_hijack.undo_hijack(shared.sd_model)
        try:
            lowvram.send_everything_to_cpu()
        except Exception as e:
            ...
        del shared.sd_model
        shared.sd_model = None
    gc.collect()
    devices.torch_gc()

    print('Starting text2video')
    print('Pipeline setup')

    # optionally store pipe in global between runs
    if pipe is None:
        pipe = setup_pipeline()

    device=devices.get_optimal_device()
    print('device',device)

    mask=None

    if do_img2img:
        
        prompt, n_prompt, steps, frames, seed, cfg_scale, width, height, eta = prompt_v, n_prompt_v, steps_v, frames_v, seed_v, cfg_scale_v, width_v, height_v, eta_v
        
        batch_count = batch_count_v # update generarl batch_count from batch_count_video

        if img2img_frames is None and img2img_frames_path == "":
            raise FileNotFoundError("Please upload a video :()")

        # Overrides
        if img2img_frames is not None:
            img2img_frames_path = img2img_frames.name

        print("got a request to *vid2vid* an existing video.")

        in_vid_fps, _, _ = get_quick_vid_info(img2img_frames_path)
        folder_name = clean_folder_name(Path(img2img_frames_path).stem)
        outdir_no_tmp = os.path.join(os.getcwd(), 'outputs', 'frame-vid2vid', folder_name)
        i = 1
        while os.path.exists(outdir_no_tmp):
            outdir_no_tmp = os.path.join(os.getcwd(), 'outputs', 'frame-vid2vid', folder_name + '_' + str(i))
            i += 1

        outdir_v2v = os.path.join(outdir_no_tmp, 'tmp_input_frames')
        os.makedirs(outdir_v2v, exist_ok=True)
        
        vid2frames(video_path=img2img_frames_path, video_in_frame_path=outdir_v2v, overwrite=True, extract_from_frame=img2img_startFrame, extract_to_frame=img2img_startFrame+frames, numeric_files_output=True, out_img_format='png')
        
        temp_convert_raw_png_path = os.path.join(outdir_v2v, "tmp_vid2vid_folder")
        duplicate_pngs_from_folder(outdir_v2v, temp_convert_raw_png_path, None, folder_name)

        videogen = []
        for f in os.listdir(temp_convert_raw_png_path):
            # double check for old _depth_ files, not really needed probably but keeping it for now
            if '_depth_' not in f:
                videogen.append(f)
                
        videogen.sort(key= lambda x:int(x.split('.')[0]))

        images=[]
        for file in tqdm(videogen, desc="Loading frames"):
            image=Image.open(os.path.join(temp_convert_raw_png_path, file))
            image=image.resize((width,height), Image.ANTIALIAS)
            array = np.array(image)
            images+=[array]

        #print(images)

        images=np.stack(images)# f h w c
        batches=1
        n_images=np.tile(images[np.newaxis, ...], (batches, 1, 1, 1, 1)) # n f h w c
        bcfhw=n_images.transpose(0,4,1,2,3)
        #convert to 0-1 float
        bcfhw=bcfhw.astype(np.float32)/255
        bfchw=bcfhw.transpose(0,2,1,3,4)#b c f h w

        print(f"Converted the frames to tensor {bfchw.shape}")

        vd_out=torch.from_numpy(bcfhw).to("cuda")

        #should be -1,1, not 0,1
        vd_out=2*vd_out-1

        #latents should have shape num_sample, 4, max_frames, latent_h,latent_w
        print("Computing latents")
        latents = pipe.compute_latents(vd_out).to(device)
    else:
        latents = None
        strength=1

    print('Working in txt2vid mode' if not do_img2img else 'Working in vid2vid mode')


    # Start the batch count loop
    pbar = tqdm(range(batch_count), leave=False)
    if batch_count == 1:
        pbar.disable=True
    
    for batch in pbar:

        # TODO: move to a separate function
        if inpainting_frames > 0:
            keys = T2VAnimKeys(SimpleNamespace(**{'max_frames':frames, 'inpainting_weights':inpainting_weights}), seed, inpainting_frames)
            images=[]
            print("gir",inpainting_image)
            print(inpainting_image.name)
            for i in range(frames):
                image=Image.open(inpainting_image.name).convert("RGB")
                image=image.resize((width,height), Image.ANTIALIAS)
                array = np.array(image)
                images+=[array]

            images=np.stack(images)# f h w c
            batches=1
            n_images=np.tile(images[np.newaxis, ...], (batches, 1, 1, 1, 1)) # n f h w c
            bcfhw=n_images.transpose(0,4,1,2,3)
            #convert to 0-1 float
            bcfhw=bcfhw.astype(np.float32)/255
            bfchw=bcfhw.transpose(0,2,1,3,4)#b c f h w

            print(f"Converted the frames to tensor {bfchw.shape}")

            vd_out=torch.from_numpy(bcfhw).to("cuda")

            #should be -1,1, not 0,1
            vd_out=2*vd_out-1

            #latents should have shape num_sample, 4, max_frames, latent_h,latent_w
            #but right now they have shape num_sample=1,4, 1 (only used 1 img), latent_h, latent_w
            print("Computing latents")
            image_latents = pipe.compute_latents(vd_out).numpy()
            # padding_width = [(0, 0), (0, 0), (0, frames-inpainting_frames), (0, 0), (0, 0)]
            # padded_latents = np.pad(image_latents, pad_width=padding_width, mode='constant', constant_values=0)

            latent_h=height//8
            latent_w=width//8
            latent_noise=np.random.normal(size=(1,4,frames,latent_h,latent_w))
            mask=np.ones(shape=(1,4,frames,latent_h,latent_w))

            mask_weights = [keys.inpainting_weights_series[frame_idx] for frame_idx in range(frames)]

            for i in range(frames):
                v=mask_weights[i]
                mask[:,:,i,:,:]=v

            masked_latents=image_latents*(1-mask)+latent_noise*mask

            latents=torch.tensor(masked_latents).to(device)

            mask=torch.tensor(mask).to(device)

            strength=1

        samples, _ = pipe.infer(prompt, n_prompt, steps, frames, seed + batch if seed != -1 else -1, cfg_scale,
                                width, height, eta, cpu_vae, device, latents,skip_steps=int(math.floor(steps*max(0, min(1 - strength, 1)))), mask=mask)

        if batch > 0:
            outdir_current = os.path.join(outdir, f"{init_timestring}_{batch}")
        print(f'text2video finished, saving frames to {outdir_current}')

        # just deleted the folder so we need to make it again
        os.makedirs(outdir_current, exist_ok=True)
        for i in range(len(samples)):
            cv2.imwrite(outdir_current + os.path.sep +
                        f"{i:06}.png", samples[i])

        # TODO: add params to the GUI
        if not skip_video_creation:
            ffmpeg_stitch_video(ffmpeg_location=ffmpeg_location, fps=fps, outmp4_path=outdir_current + os.path.sep + f"vid.mp4", imgs_path=os.path.join(outdir_current,
                                "%06d.png"), stitch_from_frame=0, stitch_to_frame=-1, add_soundtrack=add_soundtrack, audio_path=img2img_frames_path if add_soundtrack == 'Init Video' else soundtrack_path, crf=ffmpeg_crf, preset=ffmpeg_preset)
        print(f't2v complete, result saved at {outdir_current}')

        mp4 = open(outdir_current + os.path.sep + f"vid.mp4", 'rb').read()
        dataurl = "data:video/mp4;base64," + b64encode(mp4).decode()
        i1_store_t2v = f'<p style=\"font-weight:bold;margin-bottom:0em\">text2video extension for auto1111 — version 1.1b </p><video controls loop><source src="{dataurl}" type="video/mp4"></video>'
    pbar.close()


def process_videocrafter(skip_video_creation, ffmpeg_location, ffmpeg_crf, ffmpeg_preset, fps, add_soundtrack, soundtrack_path, \
                prompt, n_prompt, steps, frames, seed, cfg_scale, width, height, eta, \
                prompt_v, n_prompt_v, steps_v, frames_v, seed_v, cfg_scale_v, width_v, height_v, eta_v, batch_count_v=1, \
                batch_count=1, do_img2img=False, img2img_frames=None, img2img_frames_path="", strength=0,img2img_startFrame=0, \
                inpainting_image=None,inpainting_frames=0,inpainting_weights="",
            ):
    print(f"\033[4;33m text2video extension for auto1111 webui\033[0m")
    print(f"Git commit: {get_t2v_version()}")
    global i1_store_t2v
    init_timestring = time.strftime('%Y%m%d%H%M%S')
    outdir_current = os.path.join(outdir, f"{init_timestring}")

    os.makedirs(outdir_current, exist_ok=True)

    # load & merge config

    config_path = os.path.join(ph.models_path, "models/VideoCrafter/model_config.yaml")
    if not os.path.exists(config_path):
        config_path = os.path.join(os.getcwd(), "extensions/sd-webui-modelscope-text2video/scripts/videocrafter/base_t2v/model_config.yaml")
    if not os.path.exists(config_path):
        config_path = os.path.join(os.getcwd(), "extensions/sd-webui-text2video/scripts/videocrafter/base_t2v/model_config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f'Could not find config file at {os.path.join(ph.models_path, "models/VideoCrafter/model_config.yaml")}, nor at {os.path.join(os.getcwd(), "extensions/sd-webui-modelscope-text2video/scripts/videocrafter/base_t2v/model_config.yaml")}, nor at {os.path.join(os.getcwd(), "extensions/sd-webui-text2video/scripts/videocrafter/base_t2v/model_config.yaml")}')

    config = OmegaConf.load(config_path)
    print("VideoCrafter config: \n", config)

    from videocrafter.lvdm.samplers.ddim import DDIMSampler
    from videocrafter.sample_utils import load_model, get_conditions, make_model_input_shape, torch_to_np
    from videocrafter.sample_text2video import sample_text2video
    from videocrafter.lvdm.utils.saving_utils import npz_to_video_grid
    from video_audio_utils import add_soundtrack

    # get model & sampler
    model, _, _ = load_model(config, ph.models_path+'/VideoCrafter/model.ckpt', #TODO: support safetensors and stuff
                             inject_lora=False, # TODO
                             lora_scale=1, # TODO
                             lora_path=ph.models_path+'/VideoCrafter/LoRA/LoRA.ckpt', #TODO: support LoRA and stuff
                             )
    ddim_sampler = DDIMSampler(model)# if opt.sample_type == "ddim" else None

    # if opt.inject_lora:
    #     assert(opt.lora_trigger_word != '')
    #     prompts = [p + opt.lora_trigger_word for p in prompts]
    
    # go
    start = time.time()  

    pbar = tqdm(range(batch_count), leave=False)
    if batch_count == 1:
        pbar.disable=True
    
    for batch in pbar:
        ddim_sampler.noise_gen.manual_seed(seed + batch if seed != -1 else -1)
        # sample
        samples = sample_text2video(model, prompt, n_prompt, 1, 1,# todo:add batch size support
                        sample_type='ddim', sampler=ddim_sampler,
                        ddim_steps=steps, eta=eta, 
                        cfg_scale=cfg_scale,
                        decode_frame_bs=1,
                        ddp=False, show_denoising_progress=False,
                        )
        # save
        if batch > 0:
            outdir_current = os.path.join(outdir, f"{init_timestring}_{batch}")
        print(f'text2video finished, saving frames to {outdir_current}')

        # just deleted the folder so we need to make it again
        # os.makedirs(outdir_current, exist_ok=True)
        # for i in range(len(samples)):
        #     cv2.imwrite(outdir_current + os.path.sep +
        #                 f"{i:06}.png", samples[i])

        # # TODO: add params to the GUI
        # if not skip_video_creation:
        #     ffmpeg_stitch_video(ffmpeg_location=ffmpeg_location, fps=fps, outmp4_path=outdir_current + os.path.sep + f"vid.mp4", imgs_path=os.path.join(outdir_current,
        #                         "%06d.png"), stitch_from_frame=0, stitch_to_frame=-1, add_soundtrack=add_soundtrack, audio_path=img2img_frames_path if add_soundtrack == 'Init Video' else soundtrack_path, crf=ffmpeg_crf, preset=ffmpeg_preset)

        npz_to_video_grid(samples[0:1,...], 
                              os.path.join(outdir_current, f"vid.mp4"), 
                              fps=fps)
        add_soundtrack(ffmpeg_location, fps, os.path.join(outdir_current, f"vid.mp4"), 0, -1, None, add_soundtrack, soundtrack_path, ffmpeg_crf, ffmpeg_preset)
        print(f't2v complete, result saved at {outdir_current}')

        mp4 = open(outdir_current + os.path.sep + f"vid.mp4", 'rb').read()
        dataurl = "data:video/mp4;base64," + b64encode(mp4).decode()
        i1_store_t2v = f'<p style=\"font-weight:bold;margin-bottom:0em\">text2video extension for auto1111 — version 1.1b </p><video controls loop><source src="{dataurl}" type="video/mp4"></video>'
        print("Finish sampling!")
        print(f"Run time = {(time.time() - start):.2f} seconds")
    pbar.close()

    # if opt.ddp:
    #     dist.destroy_process_group()


def setup_common_values(mode):
    with gr.Row(elem_id=f'{mode}_prompt_toprow'):
        prompt = gr.Textbox(label='Prompt', lines=3, interactive=True, elem_id=f"{mode}_prompt", placeholder="Enter your prompt here...")
    with gr.Row(elem_id=f'{mode}_n_prompt_toprow'):
        n_prompt = gr.Textbox(label='Negative prompt', lines=2, interactive=True, elem_id=f"{mode}_n_prompt", value='text, watermark, copyright, blurry')
    with gr.Row():
        steps = gr.Slider(label='Steps', minimum=1, maximum=100, step=1, value=30)
        cfg_scale = gr.Slider(label='CFG scale', minimum=1, maximum=100, step=1, value=7)
    # with gr.Row():
        # frames = gr.Slider(label="Frames", value=24, minimum=2, maximum=125, step=1, interactive=True, precision=0)
        # seed = gr.Number(label='Seed', value = -1, Interactive = True, precision=0)
    with gr.Row():
        width = gr.Slider(label='Width', minimum=64, maximum=1024, step=64, value=256)
        height = gr.Slider(label='Height', minimum=64, maximum=1024, step=64, value=256)
    with gr.Row():
        seed = gr.Number(label='Seed', value = -1, Interactive = True, precision=0)
        eta = gr.Number(label="ETA", value=0, interactive=True)
    with gr.Row():
        frames = gr.Slider(label="Frames", value=24, minimum=2, maximum=125, step=1, interactive=True, precision=0)
        batch_count = gr.Slider(label="Batch count", value=1, minimum=1, maximum=100, step=1, interactive=True)
    
    return prompt, n_prompt, steps, seed, cfg_scale, width, height, eta, frames, batch_count

def on_ui_tabs():
    global i1_store_t2v
    # Uses only SD-requirements + ffmpeg
    dv = SimpleNamespace(**DeforumOutputArgs())
    with gr.Blocks(analytics_enabled=False) as deforum_interface:
        with gr.Row(elem_id='t2v-core').style(equal_height=False, variant='compact'):
            with gr.Column(scale=1, variant='panel'):
                with gr.Row(elem_id='model-switcher'):
                    model_type = gr.Radio(label='Model type', choices=['ModelScope', 'VideoCrafter'], value='ModelScope', elem_id='model-type')
                with gr.Tabs():
                    do_img2img = gr.State(value=0)
                    with gr.Tab('txt2vid') as tab_txt2vid:
                        # TODO: make it how it's done in Deforum/WebUI, so we won't have to track individual vars
                        prompt, n_prompt, steps, seed, cfg_scale, width, height, eta, frames, batch_count = setup_common_values('txt2vid')
                        with gr.Accordion('img2vid', open=False):
                            inpainting_image = gr.File(label="Inpainting image", interactive=True, file_count="single", file_types=["image"], elem_id="inpainting_chosen_file")
                            # TODO: should be tied to the total frame count dynamically
                            inpainting_frames=gr.Slider(label='inpainting frames',value=dv.inpainting_frames,minimum=0, maximum=200, step=1)
                            with gr.Row():
                                gr.Markdown('''`inpainting frames` is the number of frames inpainting is applied to (counting from the beginning)

The following parameters are exposed in this keyframe: max frames as `max_f`, inpainting frames as `max_i_f`, current frame number as `t`, seed as `s`

The weigths of `0:(t/max_i_f), "max_i_f":(1)` will *continue* the initial pic

To *loop it back*, set the weight to 0 for the first and for the last frame

Example: `0:(0), "max_i_f/4":(1), "3*max_i_f/4":(1), "max_i_f-1":(0)` ''')
                            with gr.Row():
                                inpainting_weights = gr.Textbox(label="Inpainting weights", value=dv.inpainting_weights, interactive=True)
                    with gr.Tab('vid2vid') as tab_vid2vid:
                        with gr.Row():
                            gr.HTML('Put your video here')
                            gr.HTML('<strong>Vid2vid for VideoCrafter is to be done!</strong>')
                        img2img_frames = gr.File(label="Input video", interactive=True, file_count="single", file_types=["video"], elem_id="vid_to_vid_chosen_file")
                        with gr.Row():
                            gr.HTML('Alternative: enter the relative (to the webui) path to the file')
                        with gr.Row():
                            img2img_frames_path = gr.Textbox(label="Input video path", interactive=True, elem_id="vid_to_vid_chosen_path", placeholder='Enter your video path here, or upload in the box above ^')
                        # TODO: here too
                        prompt_v, n_prompt_v, steps_v, seed_v, cfg_scale_v, width_v, height_v, eta_v, frames_v, batch_count_v = setup_common_values('vid2vid')
                        with gr.Row():
                            strength = gr.Slider(label="denoising strength", value=dv.strength, minimum=0, maximum=1, step=0.05, interactive=True)
                            img2img_startFrame=gr.Number(label='vid2vid start frame',value=dv.img2img_startFrame)
                    
                    tab_txt2vid.select(fn=lambda: 0, inputs=[], outputs=[do_img2img])
                    tab_vid2vid.select(fn=lambda: 1, inputs=[], outputs=[do_img2img])

                    with gr.Tab('Output settings'):
                        with gr.Row(variant='compact') as fps_out_format_row:
                            fps = gr.Slider(label="FPS", value=dv.fps, minimum=1, maximum=240, step=1)
                        with gr.Row(variant='compact') as soundtrack_row:
                            add_soundtrack = gr.Radio(['None', 'File', 'Init Video'], label="Add soundtrack", value=dv.add_soundtrack)
                            soundtrack_path = gr.Textbox(label="Soundtrack path", lines=1, interactive=True, value=dv.soundtrack_path)

                        with gr.Row(variant='compact'):
                            skip_video_creation = gr.Checkbox(label="Skip video creation", value=dv.skip_video_creation, interactive=True)
                        with gr.Row(equal_height=True, variant='compact', visible=True) as ffmpeg_set_row:
                            ffmpeg_crf = gr.Slider(minimum=0, maximum=51, step=1, label="CRF", value=dv.ffmpeg_crf, interactive=True)
                            ffmpeg_preset = gr.Dropdown(label="Preset", choices=['veryslow', 'slower', 'slow', 'medium', 'fast', 'faster', 'veryfast', 'superfast', 'ultrafast'], interactive=True, value=dv.ffmpeg_preset, type="value")
                        with gr.Row(equal_height=True, variant='compact', visible=True) as ffmpeg_location_row:
                            ffmpeg_location = gr.Textbox(label="Location", lines=1, interactive=True, value=dv.ffmpeg_location)
                    with gr.Tab('How to install? Where to get help, how to help?'):
                        gr.Markdown(welcome_text)
            with gr.Column(scale=1, variant='compact'):
                with gr.Row(variant='compact'):
                    run_button = gr.Button('Generate', elem_id=f"text2vid_generate", variant='primary')
                with gr.Row(variant='compact'):
                    i1 = gr.HTML(i1_store_t2v, elem_id='deforum_header')
                with gr.Row(visible=False):
                    result = gr.Label("")
                    result2 = gr.Label("")
                with gr.Row(variant='compact'):
                    btn = gr.Button("Click here after the generation to show the video")
                with gr.Row(variant='compact'):
                    i1 = gr.HTML(i1_store_t2v, elem_id='deforum_header')
                    def show_vid(): # Show video
                        return {
                            i1: gr.update(value=i1_store_t2v, visible=True),
                            btn: gr.update(value="Update the video", visible=True),
                        }
                    btn.click(
                        show_vid,
                        [],
                        [i1, btn],
                    )
            dummy_component = gr.Label(visible=False)
            run_button.click(
                # , extra_outputs=[None, '', '']),
                fn=wrap_gradio_gpu_call(process),
                # _js="submit_deforum",
                inputs=[skip_video_creation, ffmpeg_location, ffmpeg_crf, ffmpeg_preset, fps, add_soundtrack, soundtrack_path,
                        prompt, n_prompt, steps, frames, seed, cfg_scale, width, height, eta,\
                        prompt_v, n_prompt_v, steps_v, frames_v, seed_v, cfg_scale_v, width_v, height_v, eta_v, batch_count_v, \
                        batch_count, do_img2img, img2img_frames, img2img_frames_path, strength,img2img_startFrame, \
                        inpainting_image,inpainting_frames, inpainting_weights, \
                        model_type],  # [dummy_component, dummy_component] +
                outputs=[
                    result, result2,
                ],
            )

    return [(deforum_interface, "txt2video", "t2v_interface")]

def get_t2v_version():
    from modules import extensions as mext
    try:
        for ext in mext.extensions:
            if (ext.name in ["sd-webui-modelscope-text2video"] or ext.name in ["sd-webui-text2video"]) and ext.enabled:
                return ext.version
        return "Unknown"
    except:
        return "Unknown"

def DeforumOutputArgs():
    strength = 0.75
    img2img_startFrame = 0
    skip_video_creation = False
    fps = 15
    make_gif = False
    delete_imgs = False  # True will delete all imgs after a successful mp4 creation
    image_path = "C:/SD/20230124234916_%09d.png"
    mp4_path = "testvidmanualsettings.mp4"
    ffmpeg_location = find_ffmpeg_binary()
    ffmpeg_crf = '17'
    ffmpeg_preset = 'slow'
    add_soundtrack = 'None'  # ["File","Init Video"]
    soundtrack_path = "https://deforum.github.io/a1/A1.mp3"
    # End-Run upscaling
    r_upscale_video = False
    r_upscale_factor = 'x2'  # ['2x', 'x3', 'x4']
    # 'realesr-animevideov3' (default of realesrgan engine, does 2-4x), the rest do only 4x: 'realesrgan-x4plus', 'realesrgan-x4plus-anime'
    r_upscale_model = 'realesr-animevideov3'
    r_upscale_keep_imgs = True

    render_steps = False
    path_name_modifier = "x0_pred"  # ["x0_pred","x"]
    store_frames_in_ram = False
    # **Interpolate Video Settings**
    frame_interpolation_engine = "None"  # ["None", "RIFE v4.6", "FILM"]
    frame_interpolation_x_amount = 2  # [2 to 1000 depends on the engine]
    frame_interpolation_slow_mo_enabled = False
    frame_interpolation_slow_mo_amount = 2  # [2 to 10]
    frame_interpolation_keep_imgs = False
    inpainting_weights = '0:(t/max_i_f), "max_i_f":(1)' # linear growth weights (as they used to be in the original variant)
    inpainting_frames=0
    return locals()
    
def on_ui_settings():
    section = ('modelscope_deforum', "Text2Video")
    shared.opts.add_option("modelscope_deforum_keep_model_in_vram", shared.OptionInfo(
        False, "Keep model in VRAM between runs", gr.Checkbox, {"interactive": True, "visible": True if not (cmd_opts.lowvram or cmd_opts.medvram) else False}, section=section))
    shared.opts.add_option("modelscope_deforum_vae_settings", shared.OptionInfo(
        "GPU (half precision)", "VAE Mode:", gr.Radio, {"interactive": True, "choices": ['GPU (half precision)', 'GPU', 'CPU (Low VRAM)']}, section=section))

script_callbacks.on_ui_tabs(on_ui_tabs)
script_callbacks.on_ui_settings(on_ui_settings)
