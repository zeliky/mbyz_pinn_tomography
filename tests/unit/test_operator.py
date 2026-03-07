"""Unit tests for GAT/FMM operator."""

import torch

from tomo.operators.gat_fmm_operator import GATFMMOperator
from tomo.operators.propagation import FMMPropagation
from tomo.state.sos_state import SoSState
from torch_geometric.data import Data


def test_gat_fmm_operator_shape() -> None:
    """Operator returns tof_pred of shape (num_nodes,)."""
    op = GATFMMOperator(num_fmm_iterations=2)
    num_nodes = 5
    state = SoSState(c_values=torch.ones(num_nodes) * 0.5)  # sigmoid ~0.62
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    pos = torch.rand(num_nodes, 2)
    x = torch.full((num_nodes, 1), float("inf"))
    x[0, 0] = 0.0  # source at node 0
    data = Data(x=x, edge_index=edge_index, pos=pos, num_nodes=num_nodes)
    tof = op(state, data)
    assert tof.shape == (num_nodes,)
    assert tof[0] <= tof[1:].min()  # source has smallest T


def test_fmm_propagation_deterministic() -> None:
    """FMM propagation is deterministic for fixed inputs."""
    prop = FMMPropagation()
    T = torch.tensor([0.0, float("inf"), float("inf")])
    c = torch.ones(3) * 1.5
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    pos = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    out1 = prop(T, c, edge_index, pos, None)
    out2 = prop(T, c, edge_index, pos, None)
    torch.testing.assert_close(out1, out2)


def test_fmm_propagation_min_style() -> None:
    """Min-style: T at node 1 is at least T[0] + dist/c."""
    prop = FMMPropagation()
    T = torch.tensor([0.0, float("inf"), float("inf")])
    c = torch.ones(3) * 1.0
    # 0->1 dist 1, 1->2 dist 1
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    pos = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    out = prop(T, c, edge_index, pos, None)
    # T[1] should be 0+1/1=1, T[2] from 1 is 1+1=2
    assert out[0] == 0.0
    assert 0.9 <= out[1].item() <= 1.1
    assert 1.9 <= out[2].item() <= 2.1
