# Author: Matteo Risso (github.com/matteorisso)

from typing import Tuple

import numpy as np
from tinygrad import nn
from tinygrad.tensor import Tensor


class PromptEncoder:
  def __init__(
    self,
    embed_dim: int,
    image_embedding_size: Tuple[int, int],
    input_image_size: Tuple[int, int],
  ) -> None:
    """
    Encodes prompts for input to SAM's mask decoder.

    Arguments:
      embed_dim (int): The prompts' embedding dimension
      image_embedding_size (tuple(int, int)): The spatial size of the
          image embedding, as (H, W).
      input_image_size (int): The padded size of the image as input
          to the image encoder, as (H, W).
    """
    self.embed_dim = embed_dim
    self.input_image_size = input_image_size
    self.image_embedding_size = image_embedding_size
    self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)
    self.invalid_points = nn.Embedding(
      1, embed_dim
    )  # This are points that we declare as not part of what we are looking for
    self.point_embeddings = nn.Embedding(1, embed_dim)
    self.bbox_top_left_embeddings = nn.Embedding(1, embed_dim)
    self.bbox_bottom_right_embeddings = nn.Embedding(1, embed_dim)

  def _embed_points(
    self,
    points: Tensor,
    labels: Tensor,
  ) -> Tensor:
    """
    Embeds point prompts.

    Note: The labels are used to determine the type of prompt for each point.
      If the label is -1, the point is invalid.
      If the label is 1, the point is a point prompt.
      If the label is 2, the point is the top-left corner of a bounding box.
      If the label is 3, the point is the bottom-right corner of a bounding box.
    """
    points = points.add(0.5)  # Shift to center of pixel
    point_embedding = self.pe_layer.forward_with_coords(points, self.input_image_size)

    # Extract different types of prompts ids
    invalid_label_ids = labels.eq(-1)[:, :, None]
    point_label_ids = labels.eq(1)[:, :, None]
    topleft_label_ids = labels.eq(2)[:, :, None]
    bottomright_label_ids = labels.eq(3)[:, :, None]

    point_embedding = (
      point_embedding + self.invalid_points.weight[:, None, :] * invalid_label_ids
    )
    point_embedding = (
      point_embedding + self.point_embeddings.weight[:, None, :] * point_label_ids
    )
    point_embedding = (
      point_embedding
      + self.bbox_top_left_embeddings.weight[:, None, :] * topleft_label_ids
    )
    point_embedding = (
      point_embedding
      + self.bbox_bottom_right_embeddings.weight[:, None, :] * bottomright_label_ids
    )
    return point_embedding

  def __call__(
    self,
    coords: Tensor,
    labels: Tensor,
  ) -> Tensor:
    """
    Embeds different types of prompts, returning both sparse and dense
    embeddings.

    Arguments:
      points: A tensor of shape [B, 2]
      labels: An integer tensor of shape [B] where each element is 1,2 or 3.

    Returns:
      Tensor: sparse embeddings for the points and boxes, with shape
        BxNx(embed_dim), where N is determined by the number of input points
        and boxes.
    """
    return self._embed_points(coords, labels)


class PositionEmbeddingRandom:
  """
  Positional encoding using random spatial frequencies.
  """

  def __init__(
    self,
    num_pos_feats: int,
  ) -> None:
    self.positional_encoding_gaussian_matrix = Tensor.uniform(2, num_pos_feats)

  def _pe_encoding(self, coords: Tensor) -> Tensor:
    """Positionally encode points that are normalized to [0,1]."""
    # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
    coords = coords.mul(2).add(-1)
    coords = coords @ self.positional_encoding_gaussian_matrix
    coords = 2 * np.pi * coords
    # outputs d_1 x ... x d_n x C shape
    sin_coords = coords.sin()
    cos_coords = coords.cos()
    return Tensor.cat(*[sin_coords, cos_coords], dim=-1)

  def forward_with_coords(
    self,
    coords_input: Tensor,
    image_size: Tuple[int, int],
  ) -> Tensor:
    """Positionally encode points that are not normalized to [0,1]."""
    coords = coords_input.clone()
    coords[:, :, 0] = coords[:, :, 0] / image_size[1]
    coords[:, :, 1] = coords[:, :, 1] / image_size[0]
    return self._pe_encoding(coords)  # B x N x C
