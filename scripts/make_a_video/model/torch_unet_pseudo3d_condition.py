from typing import Optional, Tuple, Union

import torch
from torch import nn
import torch.nn as nn

from torch_embeddings import TimestepEmbedding, Timesteps
from torch_unet_pseudo3d_blocks import (
    UNetMidBlock2DCrossAttn,
    get_down_block,
    get_up_block,
)

from torch_resnet_pseudo3d import Pseudo3DConv

class UNetPseudo3DConditionOutput:
    sample: torch.FloatTensor
    def __init__(self, sample: torch.FloatTensor) -> None:
        self.sample = sample


class UNetPseudo3DConditionModel(nn.Module):
    def __init__(self,
            sample_size: Optional[int] = None,
            in_channels: int = 9,
            out_channels: int = 4,
            flip_sin_to_cos: bool = True,
            freq_shift: int = 0,
            down_block_types: Tuple[str] = (
                    "CrossAttnDownBlock2D",
                    "CrossAttnDownBlock2D",
                    "CrossAttnDownBlock2D",
                    "DownBlock2D",
            ),
            up_block_types: Tuple[str] = (
                    "UpBlock2D",
                    "CrossAttnUpBlock2D",
                    "CrossAttnUpBlock2D",
                    "CrossAttnUpBlock2D"
            ),
            block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
            layers_per_block: int = 2,
            downsample_padding: int = 1,
            mid_block_scale_factor: float = 1,
            act_fn: str = "silu",
            norm_num_groups: int = 32,
            norm_eps: float = 1e-5,
            cross_attention_dim: int = 768,
            attention_head_dim: int = 8,
            **kwargs
    ) -> None:
        super().__init__()
        self.dtype = torch.float32
        self.sample_size = sample_size
        time_embed_dim = block_out_channels[0] * 4

        # input
        self.conv_in = Pseudo3DConv(in_channels, block_out_channels[0], kernel_size=3, padding=(1, 1))

        # time
        self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
        timestep_input_dim = block_out_channels[0]

        self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)

        self.down_blocks = nn.ModuleList([])
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                    down_block_type,
                    num_layers = layers_per_block,
                    in_channels = input_channel,
                    out_channels = output_channel,
                    temb_channels = time_embed_dim,
                    add_downsample = not is_final_block,
                    resnet_eps = norm_eps,
                    resnet_act_fn = act_fn,
                    resnet_groups = norm_num_groups,
                    cross_attention_dim = cross_attention_dim,
                    attn_num_head_channels = attention_head_dim,
                    downsample_padding = downsample_padding
            )
            self.down_blocks.append(down_block)

        # mid
        self.mid_block = UNetMidBlock2DCrossAttn(
                in_channels = block_out_channels[-1],
                temb_channels = time_embed_dim,
                resnet_eps = norm_eps,
                resnet_act_fn = act_fn,
                output_scale_factor = mid_block_scale_factor,
                resnet_time_scale_shift = "default",
                cross_attention_dim = cross_attention_dim,
                attn_num_head_channels = attention_head_dim,
                resnet_groups = norm_num_groups
        )

        # count how many layers upsample the images
        self.num_upsamplers = 0

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            is_final_block = i == len(block_out_channels) - 1

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            # add upsample block for all BUT final layer
            if not is_final_block:
                add_upsample = True
                self.num_upsamplers += 1
            else:
                add_upsample = False

            up_block = get_up_block(
                    up_block_type,
                    num_layers = layers_per_block + 1,
                    in_channels = input_channel,
                    out_channels = output_channel,
                    prev_output_channel = prev_output_channel,
                    temb_channels = time_embed_dim,
                    add_upsample = add_upsample,
                    resnet_eps = norm_eps,
                    resnet_act_fn = act_fn,
                    resnet_groups = norm_num_groups,
                    cross_attention_dim = cross_attention_dim,
                    attn_num_head_channels = attention_head_dim
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        self.conv_norm_out = nn.GroupNorm(
                num_channels = block_out_channels[0],
                num_groups = norm_num_groups,
                eps = norm_eps
        )
        self.conv_act = nn.SiLU()
        self.conv_out = Pseudo3DConv(block_out_channels[0], out_channels, 3, padding = 1)


    def forward(
        self,
        sample: torch.FloatTensor,
        timesteps: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor
    ) -> Union[UNetPseudo3DConditionOutput, Tuple]:
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layears).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2**self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            forward_upsample_size = True

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)
        emb = self.time_embedding(t_emb)

        # 2. pre-process
        sample = self.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "attentions") and downsample_block.attentions is not None:
                sample, res_samples = downsample_block(
                    hidden_states = sample,
                    temb = emb,
                    encoder_hidden_states = encoder_hidden_states,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        # 4. mid
        sample = self.mid_block(sample, emb, encoder_hidden_states=encoder_hidden_states)

        # 5. up
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "attentions") and upsample_block.attentions is not None:
                sample = upsample_block(
                        hidden_states = sample,
                        temb = emb,
                        res_hidden_states_tuple = res_samples,
                        encoder_hidden_states = encoder_hidden_states,
                        upsample_size = upsample_size,
                )
            else:
                sample = upsample_block(
                        hidden_states = sample,
                        temb = emb,
                        res_hidden_states_tuple = res_samples,
                        upsample_size = upsample_size
                )
        # 6. post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return UNetPseudo3DConditionOutput(sample = sample)
