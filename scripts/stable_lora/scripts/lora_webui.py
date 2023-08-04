import torch
import glob

from safetensors.torch import load_file
from types import SimpleNamespace
from safetensors import safe_open
from einops import rearrange
import gradio as gr
import os
import json

from modules import images, script_callbacks
from modules.shared import opts, state, cmd_opts
from stable_lora.stable_utils.lora_processor import StableLoraProcessor
from t2v_helpers.extensions_utils import Text2VideoExtension

EXTENSION_TITLE = "Stable LoRA"
EXTENSION_NAME = EXTENSION_TITLE.replace(' ', '_').lower()

gr_inputs_list = [
    "lora_file_selection", 
    "lora_alpha",
    "refresh_button",
    "use_bias",
    "use_linear",
    "use_conv",
    "use_emb",
    "use_time",
    "use_multiplier"
    ]

gr_inputs_dict = {v: v for v in gr_inputs_list}
GradioInputsIds = SimpleNamespace(**gr_inputs_dict)

class StableLoraScript(Text2VideoExtension, StableLoraProcessor):
    
    def __init__(self):
        StableLoraProcessor.__init__(self)
        Text2VideoExtension.__init__(self, EXTENSION_NAME, EXTENSION_TITLE)
        self.device = 'cuda'
        self.dtype = torch.float16

    def title(self):
            return EXTENSION_TITLE
            
    def refresh_models(self, *args):
        paths_with_metadata, lora_names = self.get_lora_files()
        self.lora_files = paths_with_metadata.copy()

        return gr.Dropdown.update(value=[], choices=lora_names)

    def ui(self):
        paths_with_metadata, lora_names = self.get_lora_files()
        self.lora_files = paths_with_metadata.copy()
        REPOSITORY_LINK = "https://github.com/ExponentialML/Text-To-Video-Finetuning"

        with gr.Accordion(label=EXTENSION_TITLE, open=False) as stable_lora_section:
            with gr.Blocks(analytics_enabled=False):
                with gr.Row():
                    with gr.Column():
                        gr.HTML("<h2>Load a Trained LoRA File.</h2>")
                        gr.HTML(
                            """
                            <h3 style='color: crimson; font-weight: bold;'>
                                Only Stable LoRA files are supported.
                            </h3>
                            """
                            )
                        gr.HTML(f"""
                        <a href='{REPOSITORY_LINK}'>
                            To train a Stable LoRA file, use the finetune repository by clicking here.
                        </a>"""
                        )
                        gr.HTML(f"<span> Place your LoRA files in {cmd_opts.lora_dir}")
                        lora_files_selection = gr.Dropdown(
                            label="Available Models",
                            elem_id=GradioInputsIds.lora_file_selection,
                            choices=lora_names,
                            value=[],
                            multiselect=True,
                        )
                        lora_alpha = gr.Slider(
                            minimum=0,
                            maximum=1,
                            value=1,
                            step=0.05,
                            elem_id=GradioInputsIds.lora_alpha,
                            label="LoRA Weight"
                        )
                        refresh_button = gr.Button(
                                value="Refresh Models",
                                elem_id=GradioInputsIds.refresh_button
                            )                   
                        refresh_button.click(
                            self.refresh_models, 
                            lora_files_selection, 
                            lora_files_selection
                        )
                    with gr.Accordion(label="Advanced Settings", open=False, visible=False):
                            with gr.Column():
                                use_bias = gr.Checkbox(label="Enable Bias", elem_id=GradioInputsIds.use_bias, value=lambda: True)
                                use_linear = gr.Checkbox(label="Enable Linears", elem_id=GradioInputsIds.use_linear, value=lambda: True)
                                use_conv = gr.Checkbox(label="Enable Convolutions", elem_id=GradioInputsIds.use_conv, value=lambda: True)
                                use_emb = gr.Checkbox(label="Enable Embeddings", elem_id=GradioInputsIds.use_emb, value=lambda: True)
                                use_time = gr.Checkbox(label="Enable Time", elem_id=GradioInputsIds.use_time, value=lambda: True)
                            with gr.Column():
                                use_multiplier = gr.Number(
                                    label="Alpha Multiplier",
                                    elem_id=GradioInputsIds.use_multiplier,
                                    value=1,
                                )      


        return self.return_ui_inputs(
            return_args=[
                lora_files_selection, 
                lora_alpha, 
                use_bias, 
                use_linear, 
                use_conv, 
                use_emb, 
                use_multiplier,
                use_time
            ]
        )
    
    @torch.no_grad()
    def process(
        self, 
        p, 
        lora_files_selection, 
        lora_alpha, 
        use_bias, 
        use_linear, 
        use_conv, 
        use_emb, 
        use_multiplier,
        use_time
    ):

        # Get the list of LoRA files based off of filepath.
        lora_file_names = [x for x in lora_files_selection if x != "None"]   

        if len(self.lora_files) <= 0:
            paths_with_metadata, lora_names = self.get_lora_files()
            self.lora_files = paths_with_metadata.copy()
  
        lora_files = self.get_loras_to_process(lora_file_names)

        # Load multiple LoRAs
        lora_files_list = []    

        # Load our advanced options in a list
        advanced_options = [
            use_bias,
            use_linear,
            use_conv,
            use_emb,
            use_multiplier,
            use_time
        ]

        # Save the previous alpha value so we can re-run the LoRA with new values.        
        alpha_changed = self.handle_alpha_change(lora_alpha, p.sd_model)

        # If an advanced option changes, re-run with new options
        options_changed = self.handle_options_change(advanced_options, p.sd_model)

        # Check if we changed our LoRA models we are loading
        lora_changed = self.previous_lora_file_names != lora_file_names

        first_lora_init = not self.is_lora_loaded(p.sd_model)

        # If the LoRA is still loaded, unload it.
        unload_args = [p.sd_model, None, use_bias, use_time, use_conv, use_emb, use_linear, None]
        self.handle_lora_start(lora_files, p.sd_model, unload_args)    

        can_use_lora = self.can_use_lora(p.sd_model)
        
        lora_params_changed = any([alpha_changed, lora_changed, options_changed])

        # Process LoRA
        if can_use_lora or lora_params_changed:

            if len(lora_files) == 0: return

            for i, model in enumerate([p.sd_model, p.clip_encoder.model.transformer]):
                lora_alpha = (lora_alpha * use_multiplier) / len(lora_files)

                lora_files_list = self.load_loras_from_list(lora_files)

                args = [model, lora_files_list, use_bias, use_time, use_conv, use_emb, use_linear, lora_alpha]

                if lora_params_changed and not first_lora_init :
                    if i == 0:
                        self.log("Resetting weights to reflect changed options.")

                    undo_args = args.copy()
                    undo_args[1], undo_args[-1] = self.undo_merge_preprocess()

                    self.process_lora(*undo_args, undo_merge=True)

                self.process_lora(*args, undo_merge=False)
                    
            self.handle_after_lora_load(
                p.sd_model, 
                lora_files,
                lora_file_names, 
                advanced_options,
                lora_alpha
            )
        
        if len(lora_files) > 0 and not all([can_use_lora, lora_params_changed]):
            self.log(f"Using loaded LoRAs: {', '.join(lora_file_names)}")
            
StableLoraScriptInstance = StableLoraScript()
