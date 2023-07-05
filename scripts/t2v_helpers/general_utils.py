# Copyright (C) 2023 by Artem Khrapov (kabachuha)
# Read LICENSE for usage terms.
from modules.prompt_parser import reconstruct_cond_batch
import os
import modules.paths as ph

def get_t2v_version():
    from modules import extensions as mext
    try:
        for ext in mext.extensions:
            if (ext.name in ["sd-webui-modelscope-text2video"] or ext.name in ["sd-webui-text2video"]) and ext.enabled:
                return ext.version
        return "Unknown"
    except:
        return "Unknown"

def get_model_location(model_name):
    assert model_name is not None
    # Split model_name string with '/' as separator to find ModelScope model path
    model_name_part = str(model_name).split('/')[0]
    model_path = str(model_name).split('/')[-1]
    if model_name_part == "<modelscope>":
        return os.path.join(ph.models_path, 'ModelScope/t2v/' + model_path)
    elif model_name_part == "<videocrafter>":
        return os.path.join(ph.models_path, 'VideoCrafter')
    else:
        return os.path.join(ph.models_path, 'text2video/', model_name_part)
    
def get_model_type(model_name):
    assert model_name is not None

    if "<modelscope>" in model_name:
        return "ModelScope"
    elif model_name == "<videocrafter>":
        return "VideoCrafter (WIP)"
    else:
        return "VideoCrafter (WIP)"

def reconstruct_conds(cond, uncond, step):
    c = reconstruct_cond_batch(cond, step)
    uc = reconstruct_cond_batch(uncond, step)
    return c, uc
