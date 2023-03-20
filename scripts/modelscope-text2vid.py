# See https://github.com/modelscope/modelscope/tree/master/modelscope/pipelines/multi_modal

from modules import script_callbacks
from types import SimpleNamespace
import gradio as gr
import torch
import random
from pkg_resources import resource_filename
import modules.paths as ph
from modules import lowvram, devices, sd_hijack
from modules import shared
import gc
from modules.shared import opts, cmd_opts, state, sd_model
from scripts.t2v_pipeline import TextToVideoSynthesis, tensor2vid
from webui import wrap_gradio_gpu_call
import cv2
import os, subprocess, time

outdir = os.path.join(opts.outdir_img2img_samples, 'text2video-modelscope')
outdir = os.path.join(os.getcwd(), outdir)

def setup_pipeline():
    pipe = TextToVideoSynthesis(ph.models_path+'/ModelScope/t2v')
    return pipe

def process(skip_video_creation, ffmpeg_location, ffmpeg_crf, ffmpeg_preset, fps, add_soundtrack, soundtrack_path, prompt, n_prompt, steps, frames, cfg_scale, width=256, height=256, eta=0.0, cpu_vae=False):
    outdir_current = os.path.join(outdir, f"{time.strftime('%Y%m%d%H%M%S')}")
    try:
        latents=None

        sd_hijack.model_hijack.undo_hijack(shared.sd_model)
        shared.sd_model = None
        gc.collect()
        devices.torch_gc()

        print('Starting text2video')
        print('Pipeline setup')
        pipe = setup_pipeline()
        print('Starting text2video')
        #print(pipe.infer(prompt, n_prompt, steps, frames, cfg_scale, width, height, eta, cpu_vae, latents))
        samples, _ = pipe.infer(prompt, n_prompt, steps, frames, cfg_scale, width, height, eta, cpu_vae, latents)
        print(f'text2video finished, saving frames to {outdir_current}')
        os.makedirs(outdir_current, exist_ok=True) # just deleted the folder so we need to make it again
        for i in range(len(samples)):
            cv2.imwrite(outdir_current + os.path.sep + f"{i:09}.png", samples[i])
        
        # TODO: add params to the GUI
        if not skip_video_creation:
            ffmpeg_stitch_video(ffmpeg_location=ffmpeg_location, fps=fps, outmp4_path=outdir_current + os.path.sep + f"vid.mp4", imgs_path=outdir_current, stitch_from_frame=0, stitch_to_frame=-1, add_soundtrack=add_soundtrack, audio_path=soundtrack_path, crf=ffmpeg_crf, preset=ffmpeg_preset)
        print(f't2v complete, result saved at {outdir_current}')
    except Exception as e:
        print('Exception occured')
        print(e)
    finally:
        devices.torch_gc()
        gc.collect()
        devices.torch_gc()
    return outdir_current + os.path.sep + f"vid.mp4"

def on_ui_tabs():
    # Uses only SD-requirements + ffmpeg
    dv = SimpleNamespace(**DeforumOutputArgs())
    with gr.Blocks(analytics_enabled=False) as deforum_interface:
        gr.Markdown('Put your models from https://huggingface.co/damo-vilab/modelscope-damo-text-to-video-synthesis/tree/main to stable-diffusion-webui/models/ModelScope/t2v/. 8gbs of VRAM on top of SD (TODO: unload SD on launch) should be enough to launch.\n\n Btw, This is all going to be HACKABLE at some point. Join the development https://github.com/deforum-art/sd-webui-modelscope-text2video \n\n')
        with gr.Row(elem_id='t2v-core').style(equal_height=False, variant='compact'):
            with gr.Column(scale=1, variant='panel'):
                with gr.Tabs():
                    with gr.Tab('text2video'):
                        with gr.Row():
                            prompt = gr.Text(label='Prompt', max_lines=1, interactive=True)
                        with gr.Row():
                            n_prompt = gr.Text(label='Negative prompt', max_lines=1, interactive=True, value='text, watermark, copyright, blurry')
                        with gr.Row():
                            steps = gr.Slider(
                                label='Steps',
                                minimum=1,
                                maximum=100,
                                step=1,
                                value=30,
                                info='Steps')
                            cfg_scale = gr.Slider(
                                label='cfg_scale',
                                minimum=1,
                                maximum=100,
                                step=1,
                                value=12.5,
                                info='Steps')
                        with gr.Row():
                            frames = gr.Slider(label="frames", value=24, minimum=2, maximum=125, step=1, interactive=True, precision=0)
                            seed = gr.Slider(
                                label='Seed',
                                minimum=-1,
                                maximum=1000000,
                                step=1,
                                value=-1,
                                info='If set to -1, a different seed will be used each time.')
                        with gr.Row():
                            width = gr.Slider(
                                label='width',
                                minimum=64,
                                maximum=1024,
                                step=64,
                                value=256,
                                info='If set to -1, a different seed will be used each time.')
                            height = gr.Slider(
                                label='height',
                                minimum=64,
                                maximum=1024,
                                step=64,
                                value=256,
                                info='If set to -1, a different seed will be used each time.')
                        with gr.Row():
                            eta = gr.Number(label="eta", value=0, interactive=True)
                        with gr.Row():
                            cpu_vae = gr.Checkbox(label='Low VRAM VAE', value=False)
                    with gr.Tab('Output'):
                        with gr.Row(variant='compact') as fps_out_format_row:
                            fps = gr.Slider(label="FPS", value=dv.fps, minimum=1, maximum=240, step=1)
                        with gr.Row(variant='compact') as soundtrack_row:
                            add_soundtrack = gr.Radio(['None', 'File', 'Init Video'], label="Add soundtrack", value=dv.add_soundtrack)
                            soundtrack_path = gr.Textbox(label="Soundtrack path", lines=1, interactive=True, value = dv.soundtrack_path)

                        with gr.Row(variant='compact'):
                            skip_video_creation = gr.Checkbox(label="Skip video creation", value=dv.skip_video_creation, interactive=True)
                        with gr.Row(equal_height=True, variant='compact', visible=True) as ffmpeg_set_row:
                            ffmpeg_crf = gr.Slider(minimum=0, maximum=51, step=1, label="CRF", value=dv.ffmpeg_crf, interactive=True)
                            ffmpeg_preset = gr.Dropdown(label="Preset", choices=['veryslow', 'slower', 'slow', 'medium', 'fast', 'faster', 'veryfast', 'superfast', 'ultrafast'], interactive=True, value = dv.ffmpeg_preset, type="value")
                        with gr.Row(equal_height=True, variant='compact', visible=True) as ffmpeg_location_row:
                            ffmpeg_location = gr.Textbox(label="Location", lines=1, interactive=True, value = dv.ffmpeg_location)
            with gr.Column(scale=1, variant='compact'):
                with gr.Row():
                    run_button = gr.Button('Generate', variant='primary')
                with gr.Row():
                    result = gr.PlayableVideo(label='Result')
            dummy_component = gr.Label(visible=False)
            run_button.click(
                fn=wrap_gradio_gpu_call(process, extra_outputs=[None, '', '']),
                #_js="submit_deforum",
                inputs=[skip_video_creation, ffmpeg_location, ffmpeg_crf, ffmpeg_preset, fps, add_soundtrack, soundtrack_path, prompt, n_prompt, steps, frames, cfg_scale, width, height, eta, cpu_vae],#[dummy_component, dummy_component] + 
                outputs=[
                        result,
                ],
            )

    return [(deforum_interface, "ModelScope text2video", "t2v_interface")]

script_callbacks.on_ui_tabs(on_ui_tabs)

def find_ffmpeg_binary():
    try:
        import google.colab
        return 'ffmpeg'
    except:
        pass
    for package in ['imageio_ffmpeg', 'imageio-ffmpeg']:
        try:
            package_path = resource_filename(package, 'binaries')
            files = [os.path.join(package_path, f) for f in os.listdir(package_path) if f.startswith("ffmpeg-")]
            files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            return files[0] if files else 'ffmpeg'
        except:
            return 'ffmpeg'
          
def DeforumOutputArgs():
    skip_video_creation = False
    fps = 15 
    make_gif = False
    delete_imgs = False # True will delete all imgs after a successful mp4 creation
    image_path = "C:/SD/20230124234916_%09d.png" 
    mp4_path = "testvidmanualsettings.mp4" 
    ffmpeg_location = find_ffmpeg_binary()
    ffmpeg_crf = '17'
    ffmpeg_preset = 'slow'
    add_soundtrack = 'None' # ["File","Init Video"]
    soundtrack_path = "https://deforum.github.io/a1/A1.mp3"
    # End-Run upscaling
    r_upscale_video = False
    r_upscale_factor = 'x2' # ['2x', 'x3', 'x4']
    r_upscale_model = 'realesr-animevideov3' # 'realesr-animevideov3' (default of realesrgan engine, does 2-4x), the rest do only 4x: 'realesrgan-x4plus', 'realesrgan-x4plus-anime'
    r_upscale_keep_imgs = True
    
    render_steps = False
    path_name_modifier = "x0_pred" #["x0_pred","x"]
    store_frames_in_ram = False
    #**Interpolate Video Settings**
    frame_interpolation_engine = "None" # ["None", "RIFE v4.6", "FILM"]
    frame_interpolation_x_amount = 2 # [2 to 1000 depends on the engine]
    frame_interpolation_slow_mo_enabled = False
    frame_interpolation_slow_mo_amount = 2 #[2 to 10]
    frame_interpolation_keep_imgs = False
    return locals()


# Stitch images to a h264 mp4 video using ffmpeg
def ffmpeg_stitch_video(ffmpeg_location=None, fps=None, outmp4_path=None, stitch_from_frame=0, stitch_to_frame=None, imgs_path=None, add_soundtrack=None, audio_path=None, crf=17, preset='veryslow'):
    start_time = time.time()

    print(f"Got a request to stitch frames to video using FFmpeg.\nFrames:\n{imgs_path}\nTo Video:\n{outmp4_path}")
    msg_to_print = f"Stitching *video*..."
    print(msg_to_print)
    if stitch_to_frame == -1:
        stitch_to_frame = 999999999
    try:
        cmd = [
            ffmpeg_location,
            '-y',
            '-vcodec', 'png',
            '-r', str(float(fps)),
            '-start_number', str(stitch_from_frame),
            '-i', imgs_path,
            '-frames:v', str(stitch_to_frame),
            '-c:v', 'libx264',
            '-vf',
            f'fps={float(fps)}',
            '-pix_fmt', 'yuv420p',
            '-crf', str(crf),
            '-preset', preset,
            '-pattern_type', 'sequence',
            outmp4_path
        ]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
    except FileNotFoundError:
        print("\r" + " " * len(msg_to_print), end="", flush=True)
        print(f"\r{msg_to_print}", flush=True)
        raise FileNotFoundError("FFmpeg not found. Please make sure you have a working ffmpeg path under 'ffmpeg_location' parameter.")
    except Exception as e:
        print("\r" + " " * len(msg_to_print), end="", flush=True)
        print(f"\r{msg_to_print}", flush=True)
        raise Exception(f'Error stitching frames to video. Actual runtime error:{e}')
    
    if add_soundtrack != 'None':
        audio_add_start_time = time.time()
        try:
            cmd = [
                ffmpeg_location,
                '-i',
                outmp4_path,
                '-i',
                audio_path,
                '-map', '0:v',
                '-map', '1:a',
                '-c:v', 'copy',
                '-shortest',
                outmp4_path+'.temp.mp4'
            ]
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            if process.returncode != 0:
                print("\r" + " " * len(msg_to_print), end="", flush=True)
                print(f"\r{msg_to_print}", flush=True)
                raise RuntimeError(stderr)
            os.replace(outmp4_path+'.temp.mp4', outmp4_path)
            print("\r" + " " * len(msg_to_print), end="", flush=True)
            print(f"\r{msg_to_print}", flush=True)
            print(f"\rFFmpeg Video+Audio stitching \033[0;32mdone\033[0m in {time.time() - start_time:.2f} seconds!", flush=True)
        except Exception as e:
            print("\r" + " " * len(msg_to_print), end="", flush=True)
            print(f"\r{msg_to_print}", flush=True)
            print(f'\rError adding audio to video. Actual error: {e}', flush=True)
            print(f"FFMPEG Video (sorry, no audio) stitching \033[33mdone\033[0m in {time.time() - start_time:.2f} seconds!", flush=True)
    else:
        print("\r" + " " * len(msg_to_print), end="", flush=True)
        print(f"\r{msg_to_print}", flush=True)
        print(f"\rVideo stitching \033[0;32mdone\033[0m in {time.time() - start_time:.2f} seconds!", flush=True)
