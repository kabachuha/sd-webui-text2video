

# Some dirty stuff copypasted from webui, because it's too hardcoded there
from modules import devices, prompt_parser
from modules import shared
from modules.shared import *
import torch

def preprocess_webui_inner(prompt, negative_prompt, steps):
    """this is the main loop that both txt2img and img2img use; it calls func_init once inside all the scopes and func_sample once per batch"""

    prompt = [prompt]
    negative_prompt = [negative_prompt]

    n = 0

    if type(prompt) == list:
        assert(len(prompt) > 0)
    else:
        assert prompt is not None
    
    if os.path.exists(cmd_opts.embeddings_dir):
        modules.sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings()

    cached_uc = [None, None]
    cached_c = [None, None]

    def get_conds_with_caching(function, required_prompts, steps, cache):
        """
        Returns the result of calling function(shared.sd_model, required_prompts, steps)
        using a cache to store the result if the same arguments have been used before.

        cache is an array containing two elements. The first element is a tuple
        representing the previously used arguments, or None if no arguments
        have been used before. The second element is where the previously
        computed result is stored.
        """

        if cache[0] is not None and (required_prompts, steps) == cache[0]:
            return cache[1]

        with devices.autocast():
            cache[1] = function(shared.sd_model, required_prompts, steps)

        cache[0] = (required_prompts, steps)
        return cache[1]

    with torch.no_grad(), shared.sd_model.ema_scope():
        if len(prompt) == 0:
            return

        uc = get_conds_with_caching(prompt_parser.get_learned_conditioning, negative_prompt, steps, cached_uc)
        c = get_conds_with_caching(prompt_parser.get_learned_conditioning, prompt, steps, cached_c)

    return uc, c
