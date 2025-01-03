# Author: Matteo Risso (github.com/matteorisso)

from typing import Callable

from tinygrad.tensor import Tensor


class MLP:
  def __init__(
    self,
    input_dim: int,
    output_dim: int,
    act: Callable[[Tensor], Tensor],
  ) -> None:
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.act = act

    self.fc = (Tensor.scaled_uniform(input_dim, output_dim), Tensor.zeros(output_dim))

  def __call__(self, x: Tensor) -> Tensor:
    return self.act(x.linear(*self.fc))


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
    self.layers = [MLP(n, k, act) for n, k in zip([input_dim] + h, h + [hidden_dim])]

    # final layer
    self.fc = (Tensor.scaled_uniform(hidden_dim, output_dim), Tensor.zeros(output_dim))

  def __call__(self, x: Tensor) -> Tensor:
    for layer in self.layers:
      x = layer(x)
    return x.linear(*self.fc)
