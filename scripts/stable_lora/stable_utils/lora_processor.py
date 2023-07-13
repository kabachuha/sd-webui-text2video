import os
import glob
import torch

from safetensors.torch import load_file
from safetensors import safe_open
from modules.shared import opts, cmd_opts, state

class StableLoraProcessor:
    def __init__(self):
        self.lora_loaded = 'lora_loaded' 
        self.previous_lora_alpha = 1
        self.current_sd_checkpoint = ""
        self.previous_lora_file_names = []
        self.previous_advanced_options = []
        self.lora_files = []

    def get_lora_files(self):
        paths_with_metadata = []
        paths = glob.glob(os.path.join(cmd_opts.lora_dir, '**/*.safetensors'), recursive=True)
        lora_names = []
        
        for lora_path in paths:
            with safe_open(lora_path, 'pt') as lora_file:
                metadata = lora_file.metadata()
                if metadata is not None and 'stable_lora_text_to_video' in metadata.keys():
                    metadata['path'] = lora_path
                    metadata['lora_name'] = os.path.splitext(os.path.basename(lora_path))[0]
                    paths_with_metadata.append(metadata)

        if len(paths_with_metadata) > 0:
            lora_names = [x['lora_name'] for x in paths_with_metadata]

        return paths_with_metadata, lora_names

    def key_name_match(self, value, key, name):
        return value in key and name == key.split(f".{value}")[0]

    def is_lora_match(self, key, name):
        return self.key_name_match('lora_A', key, name)

    def is_bias_match(self, key, name):
        return self.key_name_match("bias", key, name)

    def lora_rank(self, weight): return min(weight.shape)

    def get_lora_alpha(self, alpha): 
        return alpha

    def process_lora_weight(self, weight, lora_weight, alpha, undo_merge=False):
        new_weight = weight.detach().clone()
        
        if not undo_merge:
            new_weight += lora_weight.to(weight.device, weight.dtype) * alpha
        else:
            new_weight -= lora_weight.to(weight.device, weight.dtype) * alpha

        return torch.nn.Parameter(new_weight.to(weight.device, weight.dtype))

    def lora_linear_forward(
        self, 
        weight, 
        lora_A, 
        lora_B, 
        alpha, 
        undo_merge=False, 
        *args
    ):
        l_alpha = self.get_lora_alpha(alpha)
        lora_weight = (lora_B @ lora_A)

        return self.process_lora_weight(weight, lora_weight, l_alpha, undo_merge=undo_merge)

    def lora_conv_forward(
        self, 
        weight, 
        lora_A, 
        lora_B, 
        alpha, 
        undo_merge=False, 
        is_temporal=False, 
        *args
    ):
        l_alpha = self.get_lora_alpha(alpha)
        view_shape = weight.shape

        if is_temporal:
            i, o, k = weight.shape[:3]
            view_shape = (i, o, k, k, 1)
            
        lora_weight = (lora_B @ lora_A).view(view_shape) 
        
        if is_temporal:
            lora_weight = torch.mean(lora_weight, dim=-2, keepdim=True)

        return self.process_lora_weight(weight, lora_weight, l_alpha, undo_merge=undo_merge)

    def lora_emb_forward(self, lora_A, lora_B, alpha, undo_merge=False, *args):
        l_alpha = self.get_lora_alpha(alpha)

        return (lora_B @ lora_A).transpose(0, 1) * l_alpha

    def is_lora_loaded(self, sd_model):
        return hasattr(sd_model, self.lora_loaded)

    def get_loras_to_process(self, lora_files):
        lora_files_to_load = []

        for file_name in lora_files:
            if len(self.lora_files) > 0:
                for f in self.lora_files:
                    if file_name == f['lora_name']:
                        lora_files_to_load.append(f['path'])
    
        return lora_files_to_load

    def handle_lora_load(self, sd_model, lora_files_list, set_lora_loaded=False):
        if not hasattr(sd_model, self.lora_loaded) and set_lora_loaded:
            setattr(sd_model, self.lora_loaded, True)

        if self.is_lora_loaded(sd_model) and not set_lora_loaded:
            self.process_lora(p, lora_files_list, undo_merge=True)
            delattr(sd_model, self.lora_loaded)

    def handle_alpha_change(self, lora_alpha, model):
        return (lora_alpha != self.previous_lora_alpha) \
            and self.is_lora_loaded(model)

    def handle_options_change(self, options, model):
        return (options != self.previous_advanced_options) \
            and self.is_lora_loaded(model)
    
    def handle_lora_start(self, lora_files, model):
        if len(lora_files) == 0 and self.is_lora_loaded(model):
            self.handle_lora_load(model, lora_files, set_lora_loaded=False)
    
            self.log(f"Unloaded previously loaded LoRA files")
            return

    def can_use_lora(self, model):
        return not self.is_lora_loaded(model)

    def load_loras_from_list(self, lora_files):
        lora_files_list = []

        for lora_file in lora_files:
            LORA_FILE = lora_file.split('/')[-1]
            LORA_DIR = cmd_opts.lora_dir
            LORA_PATH = f"{LORA_DIR}/{LORA_FILE}"

            lora_model_text_path = f"{LORA_DIR}/text_{LORA_FILE}"
            lora_text_exists = os.path.exists(lora_model_text_path)
            
            is_safetensors = LORA_PATH.endswith('.safetensors')
            load_method = load_file if is_safetensors else torch.load
            
            lora_model = load_method(LORA_PATH)

            lora_files_list.append(lora_model)

        return lora_files_list

    def handle_after_lora_load(
        self, 
        model, 
        lora_files,
        lora_file_names, 
        advanced_options, 
        alpha_changed,
        lora_alpha
    ):
        lora_summary = []
        self.handle_lora_load(model, lora_files, set_lora_loaded=True)
        self.previous_lora_file_names = lora_file_names
        self.previous_advanced_options = advanced_options
        self.previous_lora_alpha = lora_alpha

        for lora_file_name in lora_file_names:
            if self.is_lora_loaded(model):
                lora_summary.append(f"{lora_file_name.split('.')[0]}")
        
        if len(lora_summary) > 0:
            self.log(f"Using {model.__class__.__name__} LoRA(s):", *lora_summary)

        if alpha_changed:
            self.log("Alpha changed successfully.")

    def undo_merge_preprocess(self):
        previous_lora_files_list = self.get_loras_to_process(self.previous_lora_file_names)
        previous_lora_files = self.load_loras_from_list(previous_lora_files_list)

        return previous_lora_files, self.previous_lora_alpha

    @torch.autocast('cuda')
    def process_lora(
        self, 
        model, 
        lora_files_list, 
        use_bias, 
        use_time, 
        use_conv, 
        use_emb, 
        use_linear,
        lora_alpha, 
        undo_merge=False
    ):
        for n, m in model.named_modules():
            for lora_model in lora_files_list:
                for k, v in lora_model.items():
                    
                    # If there is bias in the LoRA, add it here.
                    if self.is_bias_match(k, n) and use_bias:
                        if m.bias is None:
                            m.bias = torch.nn.Parameter(v.to(self.device, dtype=self.dtype))
                        else:
                            m.bias.weight = v.to(self.device, dtype=self.dtype)
    
                    if self.is_lora_match(k, n):
                        lora_A = lora_model[f"{n}.lora_A"].to(self.device, dtype=self.dtype)
                        lora_B = lora_model[f"{n}.lora_B"].to(self.device, dtype=self.dtype)

                        forward_args = [m.weight, lora_A, lora_B, lora_alpha]

                        if isinstance(m, torch.nn.Linear) and use_linear:
                            if 'proj' in n:
                                forward_args[1], forward_args[2] = map(lambda l: l.squeeze(-1), (lora_A, lora_B))

                            m.weight = self.lora_linear_forward(*forward_args, undo_merge=undo_merge)

                        if isinstance(m, torch.nn.Conv2d) and use_conv:
                            m.weight = self.lora_conv_forward(*forward_args, undo_merge=undo_merge, is_temporal=False) 

                        if isinstance(m, torch.nn.Conv3d) and use_conv and use_time:
                            m.weight = self.lora_conv_forward(*forward_args, undo_merge=undo_merge, is_temporal=True) 

                        if isinstance(m, torch.nn.Embedding) and use_emb:
                            embedding_weight = self.lora_emb_forward(lora_A, lora_B, lora_alpha, undo_merge=undo_merge)
                            new_embedding_weight = torch.nn.Embedding.from_pretrained(embedding_weight)
