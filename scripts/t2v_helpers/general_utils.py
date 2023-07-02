# Copyright (C) 2023 by Artem Khrapov (kabachuha)
# Read LICENSE for usage terms.
from modules.prompt_parser import reconstruct_cond_batch

def get_t2v_version():
    from modules import extensions as mext
    try:
        for ext in mext.extensions:
            if (ext.name in ["sd-webui-modelscope-text2video"] or ext.name in ["sd-webui-text2video"]) and ext.enabled:
                return ext.version
        return "Unknown"
    except:
        return "Unknown"

def reconstruct_conds(cond, uncond, step):
    c = reconstruct_cond_batch(cond, step)
    uc = reconstruct_cond_batch(uncond, step)
    return c, uc
