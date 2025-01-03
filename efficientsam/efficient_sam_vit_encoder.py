# Author: Matteo Risso (github.com/matteorisso)

from typing import Callable, List, Optional

from tinygrad import nn
from tinygrad.tensor import Tensor


class PatchEmbed:
  """2D Image to Patch Embedding"""

  def __init__(
    self,
    img_size: int,
    patch_size: int,
    in_chans: int,
    embed_dim: int,
  ) -> None:
    self.patch_size = patch_size
    self.embedding = (
      Tensor.uniform(embed_dim, in_chans, patch_size, patch_size),
      Tensor.zeros(embed_dim),
    )

  def __call__(self, x: Tensor) -> Tensor:
    x = x.conv2d(*self.embedding, stride=self.patch_size)
    return x


class Attention:
  def __init__(
    self,
    dim: int,
    num_heads: int,
    qkv_bias: bool,
    qk_scale: Optional[float] = None,
  ):
    self.num_heads = num_heads
    head_dim = dim // num_heads
    self.scale = qk_scale or head_dim**-0.5
    self.qkv = (
      Tensor.scaled_uniform(dim, dim * 3),
      Tensor.zeros(dim * 3) if qkv_bias else None,
    )
    self.proj = (Tensor.scaled_uniform(dim, dim), Tensor.zeros(dim))

  def __call__(self, x: Tensor) -> Tensor:
    B, N, C = x.shape
    qkv = (
      x.linear(*self.qkv)
      .reshape(B, N, 3, self.num_heads, C // self.num_heads)
      .permute(2, 0, 3, 1, 4)
    )
    q, k, v = (
      qkv[0],
      qkv[1],
      qkv[2],
    )
    attn = (q @ k.transpose(-2, -1)) * self.scale
    attn = attn.softmax(axis=-1)
    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    x = x.linear(*self.proj)
    return x


class Mlp:
  def __init__(
    self,
    in_features: int,
    hidden_features: Optional[int] = None,
    out_features: Optional[int] = None,
    act_layer: Callable[[Tensor], Tensor] = lambda x: x.gelu(),
  ):
    out_features = out_features or in_features
    hidden_features = hidden_features or in_features
    self.fc1 = (
      Tensor.scaled_uniform(in_features, hidden_features),
      Tensor.zeros(hidden_features),
    )
    self.act = act_layer
    self.fc2 = (
      Tensor.scaled_uniform(hidden_features, out_features),
      Tensor.zeros(out_features),
    )

  def __call__(self, x: Tensor) -> Tensor:
    return self.act(x.linear(*self.fc1)).linear(*self.fc2)


class Block:
  def __init__(
    self,
    dim: int,
    num_heads: int,
    mlp_ratio: float = 4.0,
    qkv_bias: bool = False,
    qk_scale: Optional[float] = None,
    act_layer: Callable[[Tensor], Tensor] = lambda x: x.gelu(),
  ) -> None:
    super().__init__()
    self.attn = Attention(
      dim,
      num_heads=num_heads,
      qkv_bias=qkv_bias,
      qk_scale=qk_scale,
    )
    mlp_hidden_dim = int(dim * mlp_ratio)
    self.mlp = Mlp(
      in_features=dim,
      hidden_features=mlp_hidden_dim,
      act_layer=act_layer,
    )

  def __call__(self, x: Tensor) -> Tensor:
    x = x.add(self.attn(x.layernorm(eps=1e-6)))
    x = x.add(self.mlp(x.layernorm(eps=1e-6)))
    return x


class ImageEncoderViT:
  def __init__(
    self,
    img_size: int,
    patch_size: int,
    in_chans: int,
    patch_embed_dim: int,
    normalization_type: str,
    depth: int,
    num_heads: int,
    mlp_ratio: float,
    neck_dims: List[int],
  ) -> None:
    self.img_size = img_size
    self.image_embedding_size = img_size // (patch_size if patch_size else 1)
    self.transformer_output_dim = neck_dims[-1]
    self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, patch_embed_dim)
    _pretrain_img_size = 224
    _num_patches = (_pretrain_img_size // patch_size) ** 2
    _num_positions = _num_patches + 1  # Add class token
    self.pos_embed = Tensor.zeros(1, _num_positions, patch_embed_dim)
    self.blocks = []
    for i in range(depth):
      vit_block = Block(patch_embed_dim, num_heads, mlp_ratio, True)
      self.blocks.append(vit_block)
    self.neck = [
      nn.Conv2d(
        patch_embed_dim,
        neck_dims[0],
        kernel_size=1,
        bias=False,
      ),
      nn.LayerNorm2d(neck_dims[0]),
      nn.Conv2d(
        neck_dims[0],
        neck_dims[0],
        kernel_size=3,
        padding=1,
        bias=False,
      ),
      nn.LayerNorm2d(neck_dims[0]),
    ]

    def __call__(self, x: Tensor) -> Tensor:
      x = self.patch_embed(x)
      # B C H W -> B H W C
      x = x.permute(0, 2, 3, 1)
      x = x.add(self.pos_embed[:, 1:])
      num_patches = x.shape[1]
      x = x.reshape(x.shape[0], num_patches * num_patches, x.shape[3])
      x = x.sequential(self.blocks)
      x = x.reshape(x.shape[0], num_patches, num_patches, x.shape[2])
      x = x.permute(0, 3, 1, 2).sequential(self.neck)
      return x
