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

def setup_pipeline():
    pipe = TextToVideoSynthesis(ph.models_path+'/ModelScope/t2v')
    return pipe

def process(prompt, n_prompt, steps, frames, cfg_scale, width=256, height=256, eta=0.0, cpu_vae=False):
    latents=None
    # FIXME
    #lowvram.send_everything_to_cpu()
    #sd_hijack.model_hijack.undo_hijack(sd_model)
    #devices.torch_gc()

    print('Starting text2video')
    print('Pipeline setup')
    pipe = setup_pipeline()
    print('Starting text2video')
    print(pipe.infer(prompt, n_prompt, steps, frames, cfg_scale, width, height, eta, cpu_vae, latents))
    raise 'd'
    video_path, _ = pipe.infer(prompt, n_prompt, steps, frames, cfg_scale, width, height, eta, cpu_vae, latents)
    print(f't2v complete, result saved at {video_path}')
    print(video_path)

    #devices.torch_gc()
    #lowvram.setup_for_low_vram(sd_model, cmd_opts.medvram)
    #sd_hijack.model_hijack.hijack(sd_model)

def on_ui_tabs():
    # Uses only SD-requirements
    
    with gr.Blocks(analytics_enabled=False) as deforum_interface:
        gr.Markdown('Put your models from https://huggingface.co/damo-vilab/modelscope-damo-text-to-video-synthesis/tree/main to stable-diffusion-webui/models/ModelScope/t2v/. 8gbs of VRAM on top of SD (TODO: unload SD on launch) should be enough to launch.\n\n Btw, This is all going to be HACKABLE at some point. Join the development https://github.com/deforum-art/sd-webui-modelscope-text2video')
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
