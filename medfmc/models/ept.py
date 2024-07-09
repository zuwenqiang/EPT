import torch
import torch.nn as nn
from mmcls.models import BACKBONES
from mmcls.models.backbones import VisionTransformer
from mmcls.models.utils import resize_pos_embed
from typing import List



@BACKBONES.register_module()
class EPT(VisionTransformer):

    def __init__(self,
                 prompt_length: int = 100,
                 prompt_layers: List[int] = None,
                 prompt_init: str = 'normal',
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        for param in self.parameters():
            param.requires_grad = False

        # EPT-DEEP
        self.prompt_layers = list(range(len(self.layers))) if prompt_layers is None else prompt_layers
        num_patches = self.patch_resolution[0] * self.patch_resolution[1]
        prompt = torch.empty(
            len(self.prompt_layers), self.num_extra_tokens+num_patches, prompt_length)
        if prompt_init == 'uniform':
            nn.init.uniform_(prompt, -0.08, 0.08)
        elif prompt_init == 'zero':
            nn.init.zeros_(prompt)
        elif prompt_init == 'kaiming':
            nn.init.kaiming_normal_(prompt)
        elif prompt_init == 'token':
            nn.init.zeros_(prompt)
            self.prompt_initialized = False
        else:
            nn.init.normal_(prompt, std=0.02)
        self.prompt = nn.Parameter(prompt, requires_grad=True)
        self.prompt_length = prompt_length



    def feature_prompt_attention(self,i, x, prompt):
        x = self.layers[i].norm1(x)
        B,N,_ = x.shape

        # attention
        layer_attn = self.layers[i].attn
        qkv = layer_attn.qkv(x) 
        qkv = qkv.reshape(B, N, 3, layer_attn.num_heads,layer_attn.head_dims).permute(2, 0, 3, 1, 4) 
        q, k, v = qkv[0], qkv[1], qkv[2] 
        qk = (q @ k.transpose(-2, -1))
        attn = qk * layer_attn.scale
        # prompt attention
        prompt = prompt.unsqueeze(1).expand(-1, attn.shape[1], -1, -1)
        attn = torch.cat([prompt, attn],dim=-1)
        attn = attn.softmax(dim=-1)
        attn = layer_attn.attn_drop(attn)
        attn = attn[:,:,:,self.prompt_length:].clone()

        # attn @ v
        x = (attn @ v).transpose(1, 2).reshape(B, -1, layer_attn.embed_dims)
        x = layer_attn.proj(x)
        x = layer_attn.out_drop(layer_attn.proj_drop(x))

        if layer_attn.v_shortcut:
            x = v.squeeze(1) + x

        return x

    def forward(self, x):
        """Following mmcls implementation."""
        B = x.shape[0]
        x, patch_resolution = self.patch_embed(x)

        # stole cls_tokens impl from Phil Wang, thanks
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + resize_pos_embed(
            self.pos_embed,
            self.patch_resolution,
            patch_resolution,
            mode=self.interpolate_mode,
            num_extra_tokens=self.num_extra_tokens)
        x = self.drop_after_pos(x)

        # Add prompt
        if hasattr(self, 'prompt_initialized') and not self.prompt_initialized:
            with torch.no_grad():
                self.prompt.data += x.mean([0, 1]).detach().clone()
            self.prompt_initialized = True
        prompt = self.prompt.unsqueeze(1).expand(-1, x.shape[0], -1, -1)
        # prompt: [layer, batch, length, dim]


        if not self.with_cls_token:
            # Remove class token for transformer encoder input
            x = x[:, 1:]

        outs = []

        bottom_layer_id = self.prompt_layers[0]

        for i, layer in enumerate(self.layers):
            if i in self.prompt_layers:
                x = x + self.feature_prompt_attention(i, x, prompt[i-bottom_layer_id, :, :, :])
                x = layer.ffn(layer.norm2(x), identity=x)
            else:
                x = layer(x)

            if i == len(self.layers) - 1 and self.final_norm:
                x = self.norm1(x)


            if i in self.out_indices:
                outs.append(x[:, 0])


        return tuple(outs)
    




