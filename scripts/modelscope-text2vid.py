# See https://github.com/modelscope/modelscope/tree/master/modelscope/pipelines/multi_modal
import cv2
import gc
import os
import random
import subprocess
import time
import torch
from base64 import b64encode
from types import SimpleNamespace
import gradio as gr
from webui import wrap_gradio_gpu_call
import modules.paths as ph
from modules import devices, lowvram, script_callbacks, sd_hijack, shared
from modules.shared import cmd_opts, opts, state
from scripts.error_hardcode import get_error
from scripts.t2v_pipeline import TextToVideoSynthesis, tensor2vid
from scripts.video_audio_utils import ffmpeg_stitch_video, find_ffmpeg_binary

outdir = os.path.join(opts.outdir_img2img_samples, 'text2video-modelscope')
outdir = os.path.join(os.getcwd(), outdir)

pipe = None

def setup_pipeline():
    return TextToVideoSynthesis(ph.models_path+'/ModelScope/t2v')

i1_store_t2v = f"<p style=\"text-align:center;font-weight:bold;margin-bottom:0em\">ModelScope text2video extension for auto1111 — version 1.0b. The video will be shown below this label when ready</p>"

welcome_text = '''Put your models from <a style="color:SteelBlue" href="https://huggingface.co/damo-vilab/modelscope-damo-text-to-video-synthesis/tree/main">https://huggingface.co/damo-vilab/modelscope-damo-text-to-video-synthesis/tree/main</a> to stable-diffusion-webui/models/ModelScope/t2v/. Make sure, you downloaded the file named 'configuration.json' in its raw text form (click on the ⬇️ character to the right, don't save via right-click).

8gbs of VRAM on top of SD should be enough to launch (when the VAE unloading will be fixed, before that orient around ~12 gbs).

Join the development or report issues and feature requests here <a style="color:SteelBlue" href="https://github.com/deforum-art/sd-webui-modelscope-text2video">https://github.com/deforum-art/sd-webui-modelscope-text2video</a>

<italic>If you liked this extension, please <a style="color:SteelBlue" href="https://github.com/deforum-art/sd-webui-modelscope-text2video">give it a star on GitHub</a>!</italic> 😊

'''

def process(skip_video_creation, ffmpeg_location, ffmpeg_crf, ffmpeg_preset, fps, add_soundtrack, soundtrack_path, prompt, n_prompt, steps, frames, cfg_scale, width=256, height=256, eta=0.0, cpu_vae='GPU (half precision)', keep_pipe=False):
    global pipe
    print(f"\033[4;33mModelScope text2video extension for auto1111 webui\033[0m")
    print(f"Git commit: {get_t2v_version()}")
    global i1_store_t2v
    outdir_current = os.path.join(outdir, f"{time.strftime('%Y%m%d%H%M%S')}")
    dataurl = get_error()
    try:
        latents = None

        if shared.sd_model is not None:
            sd_hijack.model_hijack.undo_hijack(shared.sd_model)
            try:
                lowvram.send_everything_to_cpu()
            except e:
                ...
            del shared.sd_model
            shared.sd_model = None
        gc.collect()
        devices.torch_gc()

        print('Starting text2video')
        print('Pipeline setup')

        if pipe is None:
            pipe = setup_pipeline()

        print('Starting text2video')

        samples, _ = pipe.infer(prompt, n_prompt, steps, frames, cfg_scale,
                                width, height, eta, cpu_vae, devices.get_optimal_device(), latents)

        print(f'text2video finished, saving frames to {outdir_current}')

        # just deleted the folder so we need to make it again
        os.makedirs(outdir_current, exist_ok=True)
        for i in range(len(samples)):
            cv2.imwrite(outdir_current + os.path.sep +
                        f"{i:06}.png", samples[i])

        # TODO: add params to the GUI
        if not skip_video_creation:
            ffmpeg_stitch_video(ffmpeg_location=ffmpeg_location, fps=fps, outmp4_path=outdir_current + os.path.sep + f"vid.mp4", imgs_path=os.path.join(outdir_current,
                                "%06d.png"), stitch_from_frame=0, stitch_to_frame=-1, add_soundtrack=add_soundtrack, audio_path=soundtrack_path, crf=ffmpeg_crf, preset=ffmpeg_preset)
        print(f't2v complete, result saved at {outdir_current}')

        mp4 = open(outdir_current + os.path.sep + f"vid.mp4", 'rb').read()
        dataurl = "data:video/mp4;base64," + b64encode(mp4).decode()
    except Exception as e:
        print('Exception occured')
        print(e)
    finally:
        #optionally store pipe in global between runs, if not, remove it
        if not keep_pipe:
            pipe = None
        devices.torch_gc()
        gc.collect()
        devices.torch_gc()
        i1_store_t2v = f'<p style=\"font-weight:bold;margin-bottom:0em\">ModelScope text2video extension for auto1111 — version 1.0b </p><video controls loop><source src="{dataurl}" type="video/mp4"></video>'
    return f'Video at {outdir_current} ready!'

def on_ui_tabs():
    global i1_store_t2v
    # Uses only SD-requirements + ffmpeg
    dv = SimpleNamespace(**DeforumOutputArgs())
    with gr.Blocks(analytics_enabled=False) as deforum_interface:
        with gr.Row(elem_id='t2v-core').style(equal_height=False, variant='compact'):
            with gr.Column(scale=1, variant='panel'):
                with gr.Tabs():
                    with gr.Tab('text2video'):
                        with gr.Row(variant='compact'):
                            prompt = gr.Text(
                                label='Prompt', max_lines=1, interactive=True)
                        with gr.Row(variant='compact'):
                            n_prompt = gr.Text(label='Negative prompt', max_lines=1,
                                               interactive=True, value='text, watermark, copyright, blurry')
                        with gr.Row(variant='compact'):
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
                        with gr.Row(variant='compact'):
                            frames = gr.Slider(
                                label="frames", value=24, minimum=2, maximum=125, step=1, interactive=True, precision=0)
                            seed = gr.Slider(
                                label='Seed',
                                minimum=-1,
                                maximum=1000000,
                                step=1,
                                value=-1,
                                info='If set to -1, a different seed will be used each time.')
                        with gr.Row(variant='compact'):
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
                        with gr.Row(variant='compact'):
                            eta = gr.Number(
                                label="eta", value=0, interactive=True)
                        with gr.Row(variant='compact'):
                            cpu_vae = gr.Radio(label='VAE Mode', value='GPU (half precision)', choices=[
                                               'GPU (half precision)', 'GPU', 'CPU (Low VRAM)'], interactive=True)
                        with gr.Row(variant='compact'):
                            keep_pipe = gr.Checkbox(label="Keep pipe in VRAM", value=dv.keep_pipe_in_memory, interactive=True)
                    with gr.Tab('Output settings'):
                        with gr.Row(variant='compact') as fps_out_format_row:
                            fps = gr.Slider(label="FPS", value=dv.fps, minimum=1, maximum=240, step=1)
                        with gr.Row(variant='compact') as soundtrack_row:
                            add_soundtrack = gr.Radio(
                                ['None', 'File', 'Init Video'], label="Add soundtrack", value=dv.add_soundtrack)
                            soundtrack_path = gr.Textbox(
                                label="Soundtrack path", lines=1, interactive=True, value=dv.soundtrack_path)

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
                    run_button = gr.Button('Generate', variant='primary')
                with gr.Row(variant='compact'):
                    i1 = gr.HTML(i1_store_t2v, elem_id='deforum_header')
                with gr.Row(visible=False):
                    result = gr.Label("")
                    result2 = gr.Label("")
                with gr.Row(variant='compact'):
                    btn = gr.Button("Click here after the generation to show the video")
                with gr.Row(variant='compact'):
                    i1 = gr.HTML(i1_store_t2v, elem_id='deforum_header')
                    # Show video

                    def show_vid():
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
                inputs=[skip_video_creation, ffmpeg_location, ffmpeg_crf, ffmpeg_preset, fps, add_soundtrack, soundtrack_path, prompt,
                        n_prompt, steps, frames, cfg_scale, width, height, eta, cpu_vae, keep_pipe],  # [dummy_component, dummy_component] +
                outputs=[result, result2]
            )

    return [(deforum_interface, "ModelScope text2video", "t2v_interface")]

script_callbacks.on_ui_tabs(on_ui_tabs)

def get_t2v_version():
    from modules import extensions as mext
    try:
        for ext in mext.extensions:
            if ext.name in ["sd-webui-modelscope-text2video"] and ext.enabled:
                return ext.version
        return "Unknown"
    except:
        return "Unknown"

def DeforumOutputArgs():
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
    keep_pipe_in_memory = False
    return locals()