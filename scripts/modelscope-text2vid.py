# See https://github.com/modelscope/modelscope/tree/master/modelscope/pipelines/multi_modal

from modules import script_callbacks
import gradio as gr
import torch
import random
import modules.paths as ph
from modules import lowvram, devices, sd_hijack
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

def process(prompt, n_prompt, steps, frames, cfg_scale, width=256, height=256, eta=0.0, cpu_vae=False):
    latents=None

    # FIXME sd unload
    #lowvram.send_everything_to_cpu()
    #sd_hijack.model_hijack.undo_hijack(sd_model)
    #devices.torch_gc()

    print('Starting text2video')
    print('Pipeline setup')
    pipe = setup_pipeline()
    print('Starting text2video')
    #print(pipe.infer(prompt, n_prompt, steps, frames, cfg_scale, width, height, eta, cpu_vae, latents))
    samples, _ = pipe.infer(prompt, n_prompt, steps, frames, cfg_scale, width, height, eta, cpu_vae, latents)
    outdir_current = os.path.join(outdir, f"{time.strftime('%Y%m%d%H%M%S')}")
    print('text2video finished, saving frames')
    os.makedirs(outdir_current, exist_ok=True) # just deleted the folder so we need to make it again
    for i in range(len(samples)):
        cv2.imwrite(outdir_current + os.path.sep + f"{i:09}.png", samples[i])
    
    # TODO: add params to the GUI

    ffmpeg_stitch_video('ffmpeg', 24, outdir_current + os.path.sep + f"vid.mp4", 0, None, outdir_current)# add timestring
    print(f't2v complete, result saved at {outdir_current}')

    devices.torch_gc()
    #lowvram.setup_for_low_vram(sd_model, cmd_opts.medvram)
    #sd_hijack.model_hijack.hijack(sd_model)
    return outdir_current + os.path.sep + f"vid.mp4"

def on_ui_tabs():
    # Uses only SD-requirements
    
    with gr.Blocks(analytics_enabled=False) as deforum_interface:
        gr.Markdown('Put your models from https://huggingface.co/damo-vilab/modelscope-damo-text-to-video-synthesis/tree/main to stable-diffusion-webui/models/ModelScope/t2v/. 8gbs of VRAM on top of SD (TODO: unload SD on launch) should be enough to launch.\n\n Btw, This is all going to be HACKABLE at some point. Join the development https://github.com/deforum-art/sd-webui-modelscope-text2video \n\n')
        with gr.Column(scale=1, variant='panel'):
            with gr.Row():
                prompt = gr.Text(label='Prompt', max_lines=1)
            with gr.Row():
                n_prompt = gr.Text(label='Negative prompt', max_lines=1)
            with gr.Row():
                steps = gr.Slider(
                    label='Steps',
                    minimum=1,
                    maximum=100,
                    step=1,
                    value=20,
                    info='Steps')
                cfg_scale = gr.Slider(
                    label='cfg_scale',
                    minimum=1,
                    maximum=100,
                    step=1,
                    value=7,
                    info='Steps')
            with gr.Row():
                frames = gr.Number(label="frames", value=24, interactive=True, precision=0)
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
                    value=512,
                    info='If set to -1, a different seed will be used each time.')
                height = gr.Slider(
                    label='height',
                    minimum=64,
                    maximum=1024,
                    step=64,
                    value=512,
                    info='If set to -1, a different seed will be used each time.')
            with gr.Row():
                eta = gr.Number(label="eta", value=0, interactive=True)
            with gr.Row():
                cpu_vae = gr.Checkbox(label='Low VRAM VAE', value=False)
        with gr.Column(scale=1, variant='compact'):
            with gr.Row():
                run_button = gr.Button('Generate', variant='primary')
            with gr.Row():
                result = gr.PlayableVideo(label='Result')
        dummy_component = gr.Label(visible=False)
        run_button.click(
            fn=wrap_gradio_gpu_call(process, extra_outputs=[None, '', '']),
            #_js="submit_deforum",
            inputs=[prompt, n_prompt, steps, frames, cfg_scale, width, height, eta, cpu_vae],#[dummy_component, dummy_component] + 
            outputs=[
                    result,
            ],
        )

    return [(deforum_interface, "ModelScope text2video", "t2v_interface")]

script_callbacks.on_ui_tabs(on_ui_tabs)

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