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
from t2v_helpers.args import setup_text2video_settings_dictionary, setup_model_switcher
from modules.call_queue import wrap_gradio_gpu_call
from stable_lora.scripts.lora_webui import StableLoraScriptInstance
StableLoraScript = StableLoraScriptInstance

def process(*argss):
    # weird PATH stuff
    for basedir in basedirs:
        sys.path.extend([
            basedir + '/scripts',
            basedir + '/extensions/sd-webui-text2video/scripts',
            basedir + '/extensions/sd-webui-modelscope-text2video/scripts',
        ])
    if current_directory not in sys.path:
        sys.path.append(current_directory)

    run(*argss)
    return [args.i1_store_t2v]

def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as deforum_interface:
        components = {}
        with gr.Row(elem_id='t2v-model-switcher').style(equal_height=False, variant='compact'):
            with gr.Column(scale=1, variant='default'):
                model, model_type = setup_model_switcher()
            with gr.Column(scale=1, variant='default'):
                ...
        
        with gr.Row(elem_id='t2v-core').style(equal_height=False, variant='compact'):
            components = setup_text2video_settings_dictionary(model, model_type, process)
            #stable_lora_ui = StableLoraScript.ui()

            components_t2v = {**components, **components['components_t2v']}
            run_button_t2v = components_t2v.pop('run_button')
            run_button_t2v.click(
                fn=wrap_gradio_gpu_call(process),
                _js="submit_txt2vid",
                inputs=[components_t2v['dummy_component1'], components_t2v['dummy_component2']] + [components_t2v[name] for name in args.get_txt2vid_component_names()],# + stable_lora_ui,
                outputs=[
                        components_t2v['output']
                ],
            )

            components_v2v = {**components, **components['components_v2v']}
            run_button_v2v = components_v2v.pop('run_button')
            run_button_v2v.click(
                fn=wrap_gradio_gpu_call(process),
                _js="submit_txt2vid",
                inputs=[components_v2v['dummy_component1'], components_v2v['dummy_component2']] + [components_v2v[name] for name in args.get_vid2vid_component_names()],# + stable_lora_ui,
                outputs=[
                        components_v2v['output']
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
    shared.opts.add_option("modelscope_save_metadata", shared.OptionInfo(
        True, "Save generation params as video metadata", gr.Checkbox, {'interactive':True, 'visible':True}, section=section))

script_callbacks.on_ui_tabs(on_ui_tabs)
script_callbacks.on_ui_settings(on_ui_settings)
