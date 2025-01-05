# Author: Matteo Risso (github.com/matteorisso)

from typing import List, Tuple, Optional, cast

from tinygrad.tensor import Tensor

from efficientsam.efficient_sam_decoder import MaskDecoder
from efficientsam.efficient_sam_prompt_encoder import PromptEncoder
from efficientsam.efficient_sam_vit_encoder import ImageEncoderViT
from efficientsam.two_way_transformer import TwoWayTransformer


class EfficientSam:
  mask_threshold: float = 0.0
  image_format: str = "RGB"

  def __init__(
    self,
    image_encoder: ImageEncoderViT,
    prompt_encoder: PromptEncoder,
    decoder_max_num_input_points: int,
    mask_decoder: MaskDecoder,
    pixel_mean: List[float] = [0.485, 0.456, 0.406],
    pixel_std: List[float] = [0.229, 0.224, 0.225],
  ) -> None:
    """
    SAM predicts object masks from an image and input prompts.

    Arguments:
      image_encoder (ImageEncoderViT): The backbone used to encode the
        image into image embeddings that allow for efficient mask prediction.
      prompt_encoder (PromptEncoder): Encodes various types of input prompts.
      mask_decoder (MaskDecoder): Predicts masks from the image embeddings
        and encoded prompts.
      pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
      pixel_std (list(float)): Std values for normalizing pixels in the input image.
    """
    super().__init__()
    self.image_encoder = image_encoder
    self.prompt_encoder = prompt_encoder
    self.decoder_max_num_input_points = decoder_max_num_input_points
    self.mask_decoder = mask_decoder
    self.pixel_mean = Tensor(pixel_mean).view(1, 3, 1, 1)
    self.pixel_std = Tensor(pixel_std).view(1, 3, 1, 1)

  def predict_masks(
    self,
    image_embeddings: Tensor,
    batched_points: Tensor,
    batched_point_labels: Tensor,
    multimask_output: bool,
    input_h: int,
    input_w: int,
    output_h: int = -1,
    output_w: int = -1,
  ) -> Tuple[Tensor, Tensor]:
    """
    Predicts masks given image embeddings and prompts. This only runs the decoder.

    Arguments:
      image_embeddings: A tensor of shape [B, C, H, W] or [B*max_num_queries, C, H, W]
      batched_points: A tensor of shape [B, max_num_queries, num_pts, 2]
      batched_point_labels: A tensor of shape [B, max_num_queries, num_pts]
    Returns:
      A tuple of two tensors:
        low_res_mask: A tensor of shape [B, max_num_queries, 256, 256] of predicted masks
        iou_predictions: A tensor of shape [B, max_num_queries] of estimated IOU scores
    """

    batch_size, max_num_queries, num_pts, _ = batched_points.shape
    num_pts = cast(int, batched_points.shape[2])
    rescaled_batched_points = self.get_rescaled_pts(batched_points, input_h, input_w)

    if num_pts > self.decoder_max_num_input_points:
      rescaled_batched_points = rescaled_batched_points[
        :, :, : self.decoder_max_num_input_points, :
      ]
      batched_point_labels = batched_point_labels[
        :, :, : self.decoder_max_num_input_points
      ]
    elif num_pts < self.decoder_max_num_input_points:
      rescaled_batched_points = rescaled_batched_points.pad(
        (0, 0, 0, self.decoder_max_num_input_points - num_pts),
        value=-1.0,
      )
      batched_point_labels = batched_point_labels.pad(
        (0, self.decoder_max_num_input_points - num_pts),
        value=-1.0,
      )

    sparse_embeddings = self.prompt_encoder(
      rescaled_batched_points.reshape(
        batch_size * max_num_queries, self.decoder_max_num_input_points, 2
      ),
      batched_point_labels.reshape(
        batch_size * max_num_queries, self.decoder_max_num_input_points
      ),
    )
    sparse_embeddings = sparse_embeddings.view(
      batch_size,
      max_num_queries,
      sparse_embeddings.shape[1],
      sparse_embeddings.shape[2],
    )
    low_res_masks, iou_predictions = self.mask_decoder(
      image_embeddings,
      self.prompt_encoder.get_dense_pe(),
      sparse_prompt_embeddings=sparse_embeddings,
      multimask_output=multimask_output,
    )
    _, num_predictions, low_res_size, _ = low_res_masks.shape

    if output_w > 0 and output_h > 0:
      # TODO: Original implementation use "bicubic" mode which is not available in tinygrad
      output_masks = low_res_masks.interpolate(size=(output_h, output_w), mode="linear")
      output_masks = output_masks.reshape(
        batch_size, max_num_queries, num_predictions, output_h, output_w
      )
    else:
      output_masks = low_res_masks.reshape(
        batch_size,
        max_num_queries,
        num_predictions,
        low_res_size,
        low_res_size,
      )
    iou_predictions = iou_predictions.reshape(
      batch_size, max_num_queries, num_predictions
    )
    return output_masks, iou_predictions

  def get_rescaled_pts(
    self, batched_points: Tensor, input_h: int, input_w: int
  ) -> Tensor:
    return (
      (batched_points[..., 0] >= 0)
      .where(
        batched_points[..., 0] * self.image_encoder.img_size / input_w,
        -1.0,
      )
      .stack(
        (batched_points[..., 1] >= 0).where(
          batched_points[..., 1] * self.image_encoder.img_size / input_h,
          -1.0,
        ),
        dim=-1,
      )
    )

  def get_image_embeddings(self, batched_images: Tensor) -> Tensor:
    """
    Predicts masks end-to-end from provided images and prompts.
    If prompts are not known in advance, using SamPredictor is
    recommended over calling the model directly.

    Arguments:
      batched_images: A tensor of shape [B, 3, H, W]
    Returns:
      List of image embeddings each of of shape [B, C(i), H(i), W(i)].
      The last embedding corresponds to the final layer.
    """
    batched_images = self.preprocess(batched_images)
    return self.image_encoder(batched_images)

  def __call__(
    self,
    batched_images: Tensor,
    batched_points: Tensor,
    batched_point_labels: Tensor,
    scale_to_original_image_size: bool = True,
  ) -> Tuple[Tensor, Tensor]:
    """
    Predicts masks end-to-end from provided images and prompts.
    If prompts are not known in advance, using SamPredictor is
    recommended over calling the model directly.

    Arguments:
      batched_images: A tensor of shape [B, 3, H, W]
      batched_points: A tensor of shape [B, num_queries, max_num_pts, 2]
      batched_point_labels: A tensor of shape [B, num_queries, max_num_pts]

    Returns:
      A list tuples of two tensors where the ith element is by considering the first i+1 points.
        low_res_mask: A tensor of shape [B, 256, 256] of predicted masks
        iou_predictions: A tensor of shape [B, max_num_queries] of estimated IOU scores
    """
    _, _, input_h, input_w = batched_images.shape
    input_h = cast(int, input_h)
    input_w = cast(int, input_w)
    image_embeddings = self.get_image_embeddings(batched_images)
    return self.predict_masks(
      image_embeddings,
      batched_points,
      batched_point_labels,
      multimask_output=True,
      input_h=input_h,
      input_w=input_w,
      output_h=input_h if scale_to_original_image_size else -1,
      output_w=input_w if scale_to_original_image_size else -1,
    )

  def preprocess(self, x: Tensor) -> Tensor:
    """Normalize pixel values and pad to a square input."""
    if (
      x.shape[2] != self.image_encoder.img_size
      or x.shape[3] != self.image_encoder.img_size
    ):
      x = x.interpolate(
        (self.image_encoder.img_size, self.image_encoder.img_size),
        mode="linear",
      )
    return x.sub(self.pixel_mean).div(self.pixel_std)


def build_efficient_sam(
  encoder_patch_embed_dim: int,
  encoder_num_heads: int,
  checkpoint: Optional[str] = None,
):
  img_size = 1024
  encoder_patch_size = 16
  encoder_depth = 12
  encoder_mlp_ratio = 4.0
  encoder_neck_dims = [256, 256]
  decoder_max_num_input_points = 6
  decoder_transformer_depth = 2
  decoder_transformer_mlp_dim = 2048
  decoder_num_heads = 8
  decoder_upscaling_layer_dims = [64, 32]
  num_multimask_outputs = 3
  iou_head_depth = 3
  iou_head_hidden_dim = 256
  normalization_type = "layer_norm"
  normalize_before_activation = False

  def activation_fn(x: Tensor) -> Tensor:
    return x.gelu()

  image_encoder = ImageEncoderViT(
    img_size=img_size,
    patch_size=encoder_patch_size,
    in_chans=3,
    patch_embed_dim=encoder_patch_embed_dim,
    normalization_type=normalization_type,
    depth=encoder_depth,
    num_heads=encoder_num_heads,
    mlp_ratio=encoder_mlp_ratio,
    neck_dims=encoder_neck_dims,
  )

  image_embedding_size = image_encoder.image_embedding_size  # How many tokens
  encoder_transformer_output_dim = image_encoder.transformer_output_dim

  sam = EfficientSam(
    image_encoder=image_encoder,
    prompt_encoder=PromptEncoder(
      embed_dim=encoder_transformer_output_dim,
      image_embedding_size=(image_embedding_size, image_embedding_size),
      input_image_size=(img_size, img_size),
    ),
    decoder_max_num_input_points=decoder_max_num_input_points,
    mask_decoder=MaskDecoder(
      transformer_dim=encoder_transformer_output_dim,
      transformer=TwoWayTransformer(
        depth=decoder_transformer_depth,
        embedding_dim=encoder_transformer_output_dim,
        num_heads=decoder_num_heads,
        mlp_dim=decoder_transformer_mlp_dim,
        activation=activation_fn,
        normalize_before_activation=normalize_before_activation,
      ),
      num_multimask_outputs=num_multimask_outputs,
      activation=activation_fn,
      normalization_type=normalization_type,
      normalize_before_activation=normalize_before_activation,
      iou_head_depth=iou_head_depth - 1,
      iou_head_hidden_dim=iou_head_hidden_dim,
      upscaling_layer_dims=decoder_upscaling_layer_dims,
    ),
    pixel_mean=[0.485, 0.456, 0.406],
    pixel_std=[0.229, 0.224, 0.225],
  )
  if checkpoint is not None:
    raise NotImplementedError("Checkpoint loading is not supported yet.")
    # with open(checkpoint, "rb") as f:
    #   state_dict = torch.load(f, map_location="cpu")
    # sam.load_state_dict(state_dict["model"])
  return sam
