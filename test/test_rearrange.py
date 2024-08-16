import unittest
from tinygrad.tensor import Tensor
from einops import rearrange
import numpy as np

# tests from https://github.com/arogozhnikov/einops/blob/master/tests/test_ops.py#L555 and more
class TestRearrange(unittest.TestCase):
  def test_identity_patterns(self):
    t = Tensor.zeros(2,3,4,5,6)
    identity_patterns = [
      "...->...",
      "a b c d e-> a b c d e",
      "a b c d e ...-> ... a b c d e",
      "a b c d e ...-> a ... b c d e",
      "... a b c d e -> ... a b c d e",
      "a ... e-> a ... e",
      "a ... -> a ... ",
      "a ... c d e -> a (...) c d e",
    ]
    for p in identity_patterns:
      assert t.shape == t.rearrange(p).shape
  
  def test_equivalent_rearrange_patterns(self):
    t = Tensor.rand((2,3,4,5,6))
    t_npy = t.numpy()
    equivalent_rearrange_patterns = [
      ("a b c d e -> (a b) c d e", "a b ... -> (a b) ... "),
      ("a b c d e -> a b (c d) e", "... c d e -> ... (c d) e"),
      ("a b c d e -> a b c d e", "... -> ... "),
      ("a b c d e -> (a b c d e)", "... ->  (...)"),
      ("a b c d e -> b (c d e) a", "a b ... -> b (...) a"),
      ("a b c d e -> b (a c d) e", "a b ... e -> b (a ...) e"),
    ]
    for p1, p2 in equivalent_rearrange_patterns:
      np.testing.assert_allclose(t.rearrange(p1).numpy(), rearrange(t_npy, p1))
      np.testing.assert_allclose(t.rearrange(p2).numpy(), rearrange(t_npy, p2))
      assert t.rearrange(p1).shape == rearrange(t_npy, p1).shape
      assert t.rearrange(p2).shape == rearrange(t_npy, p2).shape

  def test_rearrange_permutations(self):
    # tests random permutation of axes
    for n_axes in range(1, 5): # reduced from 10 to 5 for performance
      t = Tensor.arange(2**n_axes).reshape([2] * n_axes)
      t_npy = t.numpy()
      permutation = np.random.permutation(n_axes)
      left_expression = " ".join("i" + str(axis) for axis in range(n_axes))
      right_expression = " ".join("i" + str(axis) for axis in permutation)
      expression = left_expression + " -> " + right_expression
      result = rearrange(t_npy, expression)

      np.testing.assert_allclose(t.rearrange(expression).numpy(), rearrange(t_npy, expression))

  def test_mad_patterns(self):
    # additional test for mad designers https://einops.rocks/1-einops-basics/#fancy-examples-in-random-order
    t = Tensor.rand((6, 96, 96, 3))
    t_npy = t.numpy()
    mad_patterns = [
      ("(b1 b2) h w c -> (h b1) (w b2) c ", {"b1": 2}),
      ('(b1 b2) h w c -> (h b1) (b2 w) c', {"b1": 2}),
      ('b (h1 h2) (w1 w2) c -> (h1 w2) (b w1 h2) c', {"h2": 8, "w2": 8}),
      ('b (h1 h2 h3) (w1 w2 w3) c -> (h1 w2 h3) (b w1 h2 w3) c', {"h2": 2, "w2": 2, "w3": 2, "h3": 2}),
      ('(b1 b2) (h1 h2) (w1 w2) c -> (h1 b1 h2) (w1 b2 w2) c', {"h1": 3, "w1": 3, "b2": 3}),
      ('b h w c -> w (b h) c', {})
    ]
    for p, d in mad_patterns:
      np.testing.assert_allclose(t.rearrange(p, **d).numpy(), rearrange(t_npy, p, **d))

  def test_anonymous_dims(self):
    t = Tensor.rand((2,6))
    t_npy = t.numpy()
    anonymous_dims_patterns = [
      "a b -> a () b",
      "a b -> 1 1 a () b 1",
      "a (b 1) -> 1 1 b () a 1",
    ]
    for p in anonymous_dims_patterns:
      np.testing.assert_allclose(t.rearrange(p).numpy(), rearrange(t_npy, p))

  def test_miscellaneous(self):
    t = Tensor.rand((6, 4, 12))
    t_npy = t.numpy()
    misc_patterns = [
      ("a ... (b b1) -> (a b) ... b1", {"b1": 2}),
      ("a ... (c1 c2 1) -> 1 1 1 a ... c1 1 c2", {"c1": 4, "c2": 3}),
      ("... b (c1 c2 1) -> 1 1 1 ... b c1 1 c2", {"c1": 4, "c2": 3})
    ]
    for p, d in misc_patterns:
      np.testing.assert_allclose(t.rearrange(p, **d).numpy(), rearrange(t_npy, p, **d))

if __name__ == "__main__":
  unittest.main()