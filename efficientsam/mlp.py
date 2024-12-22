# Author: Matteo Risso (github.com/matteorisso)

from typing import Callable

from tinygrad.tensor import Tensor


class MLPBlock:
  def __init__(
    self,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int,
    act: Callable[[Tensor], Tensor],
  ) -> None:
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.output_dim = output_dim
    self.num_layers = num_layers
    self.act = act

    h = [hidden_dim] * (num_layers - 1)
    self.layers = [MLP]
