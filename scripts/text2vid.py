# Copyright (C) 2023 by Artem Khrapov (kabachuha)
# Read LICENSE for usage terms.

import sys, os

basedirs = [os.getcwd()]
if 'google.colab' in sys.modules:
    basedirs.append('/content/gdrive/MyDrive/sd/stable-diffusion-webui')  # hardcode as TheLastBen's colab seems to be the primal source

for basedir in basedirs:
    deforum_paths_to_ensure = [basedir + '/extensions/sd-webui-text2video/scripts', basedir + '/extensions/sd-webui-modelscope-text2video/scripts', basedir]

    for deforum_scripts_path_fix in deforum_paths_to_ensure:
        if not deforum_scripts_path_fix in sys.path:
            sys.path.extend([deforum_scripts_path_fix])

current_directory = os.path.dirname(os.path.abspath(__file__))
if current_directory not in sys.path:
    sys.path.append(current_directory)

import gradio as gr
from modules import script_callbacks, shared
from modules.shared import cmd_opts, opts
from t2v_helpers.render import run
import t2v_helpers.args as args
from t2v_helpers.args import setup_text2video_settings_dictionary
from webui import wrap_gradio_gpu_call
from stable_lora.scripts.lora_webui import StableLoraScriptInstance
StableLoraScript = StableLoraScriptInstance

def process(*args):
    # weird PATH stuff
    for basedir in basedirs:
        sys.path.extend([
            basedir + '/scripts',
            basedir + '/extensions/sd-webui-text2video/scripts',
            basedir + '/extensions/sd-webui-modelscope-text2video/scripts',
        ])
    if current_directory not in sys.path:
        sys.path.append(current_directory)

    run(*args)
    return f'Video ready'

def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as deforum_interface:
        components = {}
        with gr.Row(elem_id='t2v-core').style(equal_height=False, variant='compact'):
            with gr.Column(scale=1, variant='panel'):
                components = setup_text2video_settings_dictionary()
                stable_lora_ui = StableLoraScript.ui()
            with gr.Column(scale=1, variant='compact'):
                with gr.Row(elem_id=f"text2vid_generate_box", variant='compact', elem_classes="generate-box"):
                    interrupt = gr.Button('Interrupt', elem_id=f"text2vid_interrupt", elem_classes="generate-box-interrupt")
                    skip = gr.Button('Skip', elem_id=f"text2vid_skip", elem_classes="generate-box-skip")
                    run_button = gr.Button('Generate', elem_id=f"text2vid_generate", variant='primary')

                    skip.click(
                        fn=lambda: shared.state.skip(),
                        inputs=[],
                        outputs=[],
                    )

                    interrupt.click(
                        fn=lambda: shared.state.interrupt(),
                        inputs=[],
                        outputs=[],
                    )
                with gr.Row(variant='compact'):
                    i1 = gr.HTML(args.i1_store_t2v, elem_id='deforum_header')
                with gr.Row(visible=False):
                    dummy_component1 = gr.Label("")
                    dummy_component2 = gr.Label("")
                with gr.Row(variant='compact'):
                    btn = gr.Button("Click here after the generation to show the video")
                with gr.Row(variant='compact', elem_id='text2vid_results_panel'):
                    ...
                    # gr.Label("", visible=False)
                with gr.Row(variant='compact'):
                    i1 = gr.HTML(args.i1_store_t2v, elem_id='deforum_header')

                    def show_vid():  # Show video1
                        return {
                            i1: gr.update(value=args.i1_store_t2v, visible=True),
                            btn: gr.update(value="Update the video", visible=True),
                        }

                    btn.click(
                        show_vid,
                        [],
                        [i1, btn],
                    )
            run_button.click(
                # , extra_outputs=[None, '', '']),
                fn=wrap_gradio_gpu_call(process),
                _js="submit_txt2vid",
                inputs=[dummy_component1, dummy_component2] + [components[name] for name in args.get_component_names()] + stable_lora_ui,
                outputs=[
                    dummy_component1, dummy_component2,
                ],
            )
    return [(deforum_interface, "txt2video", "t2v_interface")]

def on_ui_settings():
    section = ('modelscope_deforum', "Text2Video")
    shared.opts.add_option("modelscope_deforum_keep_model_in_vram", shared.OptionInfo(
        'None', "Keep model in VRAM between runs", gr.Radio,
        {"interactive": True, "choices": ['None', 'Main Model Only', 'All'], "visible": True if not (cmd_opts.lowvram or cmd_opts.medvram) else False}, section=section))
    shared.opts.add_option("modelscope_deforum_vae_settings", shared.OptionInfo(
        "GPU (half precision)", "VAE Mode:", gr.Radio, {"interactive": True, "choices": ['GPU (half precision)', 'GPU', 'CPU (Low VRAM)']}, section=section))
    shared.opts.add_option("modelscope_deforum_show_n_videos", shared.OptionInfo(
        -1, "How many videos to show on the right panel on completion (-1 = show all)", gr.Number, {"interactive": True, "visible": True}, section=section))
    shared.opts.add_option("modelscope_save_info_to_file", shared.OptionInfo(
        False, "Save generation params to a text file near the video", gr.Checkbox, {'interactive':True, 'visible':True}, section=section))

script_callbacks.on_ui_tabs(on_ui_tabs)
script_callbacks.on_ui_settings(on_ui_settings)
