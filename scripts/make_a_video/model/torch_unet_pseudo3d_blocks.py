from typing import Union, Optional
import torch
from torch import nn

from torch_attention_pseudo3d import TransformerPseudo3DModel
from torch_resnet_pseudo3d import Downsample2D, ResnetBlockPseudo3D, Upsample2D


class UNetMidBlock2DCrossAttn(nn.Module):
    def __init__(self,
            in_channels: int,
            temb_channels: int,
            dropout: float = 0.0,
            num_layers: int = 1,
            resnet_eps: float = 1e-6,
            resnet_time_scale_shift: str = "default",
            resnet_act_fn: str = "swish",
            resnet_groups: Optional[int] = 32,
            resnet_pre_norm: bool = True,
            attn_num_head_channels: int = 1,
            attention_type: str = "default",
            output_scale_factor: float =1.0,
            cross_attention_dim: int = 1280,
            **kwargs
    ) -> None:
        super().__init__()

        self.attention_type = attention_type
        self.attn_num_head_channels = attn_num_head_channels
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)

        # there is always at least one resnet
        resnets = [
            ResnetBlockPseudo3D(
                    in_channels = in_channels,
                    out_channels = in_channels,
                    temb_channels = temb_channels,
                    eps = resnet_eps,
                    groups = resnet_groups,
                    dropout = dropout,
                    time_embedding_norm = resnet_time_scale_shift,
                    #non_linearity = resnet_act_fn,
                    output_scale_factor = output_scale_factor,
                    pre_norm = resnet_pre_norm
            )
        ]
        attentions = []

        for _ in range(num_layers):
            attentions.append(
                TransformerPseudo3DModel(
                        in_channels = in_channels,
                        num_attention_heads = attn_num_head_channels,
                        attention_head_dim = in_channels // attn_num_head_channels,
                        num_layers = 1,
                        cross_attention_dim = cross_attention_dim,
                        norm_num_groups = resnet_groups
                )
            )
            resnets.append(
                ResnetBlockPseudo3D(
                        in_channels = in_channels,
                        out_channels = in_channels,
                        temb_channels = temb_channels,
                        eps = resnet_eps,
                        groups = resnet_groups,
                        dropout = dropout,
                        time_embedding_norm = resnet_time_scale_shift,
                        #non_linearity = resnet_act_fn,
                        output_scale_factor = output_scale_factor,
                        pre_norm = resnet_pre_norm
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(self, hidden_states, temb = None, encoder_hidden_states = None):
        hidden_states = self.resnets[0](hidden_states, temb)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            hidden_states = attn(hidden_states, encoder_hidden_states).sample
            hidden_states = resnet(hidden_states, temb)

        return hidden_states


class CrossAttnDownBlock2D(nn.Module):
    def __init__(self,
            in_channels: int,
            out_channels: int,
            temb_channels: int,
            dropout: float = 0.0,
            num_layers: int = 1,
            resnet_eps: float = 1e-6,
            resnet_time_scale_shift: str = "default",
            resnet_act_fn: str = "swish",
            resnet_groups: int = 32,
            resnet_pre_norm: bool = True,
            attn_num_head_channels: int = 1,
            cross_attention_dim: int = 1280,
            attention_type: str = "default",
            output_scale_factor: float = 1.0,
            downsample_padding: int = 1,
            add_downsample: bool = True
    ):
        super().__init__()
        resnets = []
        attentions = []

        self.attention_type = attention_type
        self.attn_num_head_channels = attn_num_head_channels

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlockPseudo3D(
                        in_channels = in_channels,
                        out_channels = out_channels,
                        temb_channels = temb_channels,
                        eps = resnet_eps,
                        groups = resnet_groups,
                        dropout = dropout,
                        time_embedding_norm = resnet_time_scale_shift,
                        #non_linearity = resnet_act_fn,
                        output_scale_factor = output_scale_factor,
                        pre_norm = resnet_pre_norm
                )
            )
            attentions.append(
                TransformerPseudo3DModel(
                        in_channels = out_channels,
                        num_attention_heads = attn_num_head_channels,
                        attention_head_dim = out_channels // attn_num_head_channels,
                        num_layers = 1,
                        cross_attention_dim = cross_attention_dim,
                        norm_num_groups = resnet_groups
                )
            )
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                            out_channels,
                            use_conv = True,
                            out_channels = out_channels,
                            padding = downsample_padding,
                            name = "op"
                    )
                ]
            )
        else:
            self.downsamplers = None

    def forward(self, hidden_states, temb = None, encoder_hidden_states = None):
        output_states = ()

        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states, encoder_hidden_states = encoder_hidden_states).sample

            output_states += (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states += (hidden_states,)

        return hidden_states, output_states


class DownBlock2D(nn.Module):
    def __init__(self,
            in_channels: int,
            out_channels: int,
            temb_channels: int,
            dropout: float = 0.0,
            num_layers: int = 1,
            resnet_eps: float = 1e-6,
            resnet_time_scale_shift: str = "default",
            resnet_act_fn: str = "swish",
            resnet_groups: int = 32,
            resnet_pre_norm: bool = True,
            output_scale_factor: float = 1.0,
            add_downsample: bool = True,
            downsample_padding: int = 1
    ) -> None:
        super().__init__()
        resnets = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlockPseudo3D(
                        in_channels = in_channels,
                        out_channels = out_channels,
                        temb_channels = temb_channels,
                        eps = resnet_eps,
                        groups = resnet_groups,
                        dropout = dropout,
                        time_embedding_norm = resnet_time_scale_shift,
                        #non_linearity = resnet_act_fn,
                        output_scale_factor = output_scale_factor,
                        pre_norm = resnet_pre_norm
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        out_channels,
                        use_conv = True,
                        out_channels = out_channels,
                        padding = downsample_padding,
                        name = "op"
                    )
                ]
            )
        else:
            self.downsamplers = None


    def forward(self, hidden_states, temb = None):
        output_states = ()

        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb)

            output_states += (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states += (hidden_states,)

        return hidden_states, output_states


class CrossAttnUpBlock2D(nn.Module):
    def __init__(self,
            in_channels: int,
            out_channels: int,
            prev_output_channel: int,
            temb_channels: int,
            dropout: float = 0.0,
            num_layers: int = 1,
            resnet_eps: float = 1e-6,
            resnet_time_scale_shift: str = "default",
            resnet_act_fn: str = "swish",
            resnet_groups: int = 32,
            resnet_pre_norm: bool = True,
            attn_num_head_channels: int = 1,
            cross_attention_dim: int = 1280,
            attention_type: str = "default",
            output_scale_factor: float = 1.0,
            add_upsample: bool = True
    ) -> None:
        super().__init__()
        resnets = []
        attentions = []

        self.attention_type = attention_type
        self.attn_num_head_channels = attn_num_head_channels

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                    ResnetBlockPseudo3D(
                            in_channels = resnet_in_channels + res_skip_channels,
                            out_channels = out_channels,
                            temb_channels = temb_channels,
                            eps = resnet_eps,
                            groups = resnet_groups,
                            dropout = dropout,
                            time_embedding_norm = resnet_time_scale_shift,
                            #non_linearity = resnet_act_fn,
                            output_scale_factor = output_scale_factor,
                            pre_norm = resnet_pre_norm
                    )
            )
            attentions.append(
                    TransformerPseudo3DModel(
                            in_channels = out_channels,
                            num_attention_heads = attn_num_head_channels,
                            attention_head_dim = out_channels // attn_num_head_channels,
                            num_layers = 1,
                            cross_attention_dim = cross_attention_dim,
                            norm_num_groups = resnet_groups
                    )
            )
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsamplers = nn.ModuleList([
                    Upsample2D(
                            out_channels,
                            use_conv = True,
                            out_channels = out_channels
                    )
            ])
        else:
            self.upsamplers = None

    def forward(self,
            hidden_states,
            res_hidden_states_tuple,
            temb = None,
            encoder_hidden_states = None,
            upsample_size = None
    ):
        for resnet, attn in zip(self.resnets, self.attentions):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states, encoder_hidden_states = encoder_hidden_states).sample

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)

        return hidden_states


class UpBlock2D(nn.Module):
    def __init__(self,
            in_channels: int,
            prev_output_channel: int,
            out_channels: int,
            temb_channels: int,
            dropout: float = 0.0,
            num_layers: int = 1,
            resnet_eps: float = 1e-6,
            resnet_time_scale_shift: str = "default",
            resnet_act_fn: str = "swish",
            resnet_groups: int = 32,
            resnet_pre_norm: bool = True,
            output_scale_factor: float = 1.0,
            add_upsample: bool = True
    ) -> None:
        super().__init__()
        resnets = []

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                    ResnetBlockPseudo3D(
                            in_channels = resnet_in_channels + res_skip_channels,
                            out_channels = out_channels,
                            temb_channels = temb_channels,
                            eps = resnet_eps,
                            groups = resnet_groups,
                            dropout = dropout,
                            time_embedding_norm = resnet_time_scale_shift,
                            #non_linearity = resnet_act_fn,
                            output_scale_factor = output_scale_factor,
                            pre_norm = resnet_pre_norm
                    )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsamplers = nn.ModuleList([
                    Upsample2D(
                            out_channels,
                            use_conv = True,
                            out_channels = out_channels
                    )
            ])
        else:
            self.upsamplers = None


    def forward(self, hidden_states, res_hidden_states_tuple, temb = None, upsample_size = None):
        for resnet in self.resnets:
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            hidden_states = resnet(hidden_states, temb)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)

        return hidden_states


def get_down_block(
        down_block_type: str,
        num_layers: int,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        add_downsample: bool,
        resnet_eps: float,
        resnet_act_fn: str,
        attn_num_head_channels: int,
        resnet_groups: Optional[int] = None,
        cross_attention_dim: Optional[int] = None,
        downsample_padding: Optional[int] = None,
) -> Union[DownBlock2D, CrossAttnDownBlock2D]:
    down_block_type = down_block_type[7:] if down_block_type.startswith("UNetRes") else down_block_type
    if down_block_type == "DownBlock2D":
        return DownBlock2D(
                num_layers = num_layers,
                in_channels = in_channels,
                out_channels = out_channels,
                temb_channels = temb_channels,
                add_downsample = add_downsample,
                resnet_eps = resnet_eps,
                resnet_act_fn = resnet_act_fn,
                resnet_groups = resnet_groups,
                downsample_padding = downsample_padding
        )
    elif down_block_type == "CrossAttnDownBlock2D":
        if cross_attention_dim is None:
            raise ValueError("cross_attention_dim must be specified for CrossAttnDownBlock2D")
        return CrossAttnDownBlock2D(
                num_layers = num_layers,
                in_channels = in_channels,
                out_channels = out_channels,
                temb_channels = temb_channels,
                add_downsample = add_downsample,
                resnet_eps = resnet_eps,
                resnet_act_fn = resnet_act_fn,
                resnet_groups = resnet_groups,
                downsample_padding = downsample_padding,
                cross_attention_dim = cross_attention_dim,
                attn_num_head_channels = attn_num_head_channels
        )
    raise ValueError(f"{down_block_type} does not exist.")


def get_up_block(
        up_block_type: str,
        num_layers,
        in_channels,
        out_channels,
        prev_output_channel,
        temb_channels,
        add_upsample,
        resnet_eps,
        resnet_act_fn,
        attn_num_head_channels,
        resnet_groups = None,
        cross_attention_dim = None,
) -> Union[UpBlock2D, CrossAttnUpBlock2D]:
    up_block_type = up_block_type[7:] if up_block_type.startswith("UNetRes") else up_block_type
    if up_block_type == "UpBlock2D":
        return UpBlock2D(
                num_layers = num_layers,
                in_channels = in_channels,
                out_channels = out_channels,
                prev_output_channel = prev_output_channel,
                temb_channels = temb_channels,
                add_upsample = add_upsample,
                resnet_eps = resnet_eps,
                resnet_act_fn = resnet_act_fn,
                resnet_groups = resnet_groups
        )
    elif up_block_type == "CrossAttnUpBlock2D":
        if cross_attention_dim is None:
            raise ValueError("cross_attention_dim must be specified for CrossAttnUpBlock2D")
        return CrossAttnUpBlock2D(
                num_layers = num_layers,
                in_channels = in_channels,
                out_channels = out_channels,
                prev_output_channel = prev_output_channel,
                temb_channels = temb_channels,
                add_upsample = add_upsample,
                resnet_eps = resnet_eps,
                resnet_act_fn = resnet_act_fn,
                resnet_groups = resnet_groups,
                cross_attention_dim = cross_attention_dim,
                attn_num_head_channels = attn_num_head_channels
        )
    raise ValueError(f"{up_block_type} does not exist.")

