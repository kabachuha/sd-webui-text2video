# Copyright (C) 2023 by Artem Khrapov (kabachuha)
# Read LICENSE for usage terms.

import traceback
from modelscope.process_modelscope import process_modelscope
import modelscope.process_modelscope as pm
from videocrafter.process_videocrafter import process_videocrafter
from modules.shared import opts
from .error_hardcode import get_error
from modules import lowvram, devices, sd_hijack
import logging
import gc
import t2v_helpers.args as t2v_helpers_args


def run(*args):
    dataurl = get_error()
    vids_pack = [dataurl]
    component_names = t2v_helpers_args.get_component_names()
    # api check
    affected_args = args[2:] if len(args) > 36 else args
    # TODO: change to i+2 when we will add the progress bar
    args_dict = {
        component_names[i]: affected_args[i] for i in range(0, len(component_names))
    }
    model_type = args_dict["model_type"]
    t2v_helpers_args.i1_store_t2v = f'<p style="font-weight:bold;margin-bottom:0em">text2video extension for auto1111 — version 1.2b </p><video controls loop><source src="{dataurl}" type="video/mp4"></video>'
    keep_pipe_in_vram = (
        opts.data.get("modelscope_deforum_keep_model_in_vram")
        if opts.data is not None
        and opts.data.get("modelscope_deforum_keep_model_in_vram") is not None
        else "None"
    )
    try:
        print("text2video — The model selected is: ", args_dict["model_type"])
        if model_type == "ModelScope":
            vids_pack = process_modelscope(args_dict)
        elif model_type == "VideoCrafter (WIP)":
            vids_pack = process_videocrafter(args_dict)
        else:
            raise NotImplementedError(f"Unknown model type: {model_type}")
    except Exception as e:
        traceback.print_exc()
        print("Exception occurred:", e)
    finally:
        # optionally store pipe in global between runs, if not, remove it
        if keep_pipe_in_vram == "None":
            pm.pipe = None
        devices.torch_gc()
        gc.collect()
    return vids_pack
