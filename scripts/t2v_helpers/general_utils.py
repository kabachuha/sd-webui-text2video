

def get_t2v_version():
    from modules import extensions as mext
    try:
        for ext in mext.extensions:
            if (ext.name in ["sd-webui-modelscope-text2video"] or ext.name in ["sd-webui-text2video"]) and ext.enabled:
                return ext.version
        return "Unknown"
    except:
        return "Unknown"