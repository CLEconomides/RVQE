import pytest

from .quantum import *


# QUANTUM


def approx_equal(a: tensor, b: tensor) -> bool:
    return (a.reshape(-1) - b.reshape(-1)).norm() < 1e-4


def equal(a: tensor, b: tensor) -> bool:
    return torch.all(a.reshape(-1) == b.reshape(-1)).item()


def test_ket():
    assert equal(ket("00"), tensor([1.0, 0.0, 0.0, 0.0]))
    assert equal(ket("01"), tensor([0.0, 1.0, 0.0, 0.0]))
    assert equal(ket("10"), tensor([0.0, 0.0, 1.0, 0.0]))
    assert equal(ket("11"), tensor([0.0, 0.0, 0.0, 1.0]))
    assert approx_equal(ket("0+1"), tensor([0.0000, 0.7071, 0.0000, 0.7071, 0.0000, 0.0000, 0.0000, 0.0000]),)
    assert equal(normalize(torch.ones(4).reshape(2, 2)), tensor([0.5, 0.5, 0.5, 0.5]))


def test_probabilities():
    assert equal(probabilities(ket("00"), [0, 1]), tensor([1.0, 0.0, 0.0, 0.0]))
    assert equal(probabilities(ket("11"), [0]), tensor([0.0, 1.0]))
    assert equal(probabilities(ket0(5), [1]), tensor([1.0, 0.0]))

    # EPR pair
    psiminus = normalize(ket("00") - ket("11"))
    assert approx_equal(probabilities(psiminus, [0, 1]), tensor([0.5, 0.0, 0.0, 0.5]))
    assert approx_equal(probabilities(psiminus, [1]), tensor([0.5, 0.5]))


def test_apply():
    assert squish_idcs_up("zkxm") == "zwyx"


# GATE LAYERS
from .gate_layers import *


def test_gates():
    assert equal(XLayer(0)(ket("0")), ket("1"))
    with pytest.raises(AssertionError):
        # wrong index
        assert equal(XLayer(1)(ket("0")), ket("1"))

    assert equal(HLayer(0)(ket("1")), normalize(ket("0") - ket("1")))
    assert approx_equal(rYLayer(0, -pi / 2).to_mat(), tensor([[0.7071, -0.7071], [0.7071, 0.7071]]))
    assert approx_equal(
        crYLayer(0, 1, initial_φ=pi / 2).to_mat(),
        tensor(
            [
                [1.0000, 0.0000, 0.0000, 0.0000],
                [0.0000, 1.0000, 0.0000, 0.0000],
                [0.0000, 0.0000, 0.7071, 0.7071],
                [0.0000, 0.0000, -0.7071, 0.7071],
            ]
        ),
    )
    assert approx_equal(
        apply(rYLayer(0).U, ket0(3), [0], verbose=True),
        apply(rYLayer(2).U, ket0(3), [2], verbose=True).transpose(0, 2),
    )
    assert equal(crYLayer(0, 1).forward(ket("00")), ket("00")), "controlled gate has no action if control is 0"
    assert not equal(crYLayer(1, 0).forward(ket("0+")), ket("0+")), "controlled gate acts if control is 1"
    foo = apply(
        crYLayer([0], 1, initial_φ=1.0).U, apply(HLayer(0).U, ket0(2), [0], verbose=True), [1, 0], verbose=True,
    )
    assert approx_equal(foo, tensor([0.7071, 0.0000, 0.7071, 0.0000])), "crYLayer has no action if control is 0"
    assert equal(
        cmiYLayer(0, 1).to_mat(),
        tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, -1.0, 0.0],]),
    ), "controlled-iY"
    foo = tensor([[1, 2], [3, 4]], dtype=torch.float)
    assert equal(cmiYLayer(0, 1)(foo), tensor([1.0, 2.0, -4.0, 3.0]))
    assert equal(cmiYLayer(0, 1).T.forward(foo), tensor([1.0, 2.0, 4.0, -3.0]))


def test_parameter_sharing():
    foo = rYLayer(2)
    assert len(list(nn.Sequential(foo, foo.T).named_parameters())) == 1, "we expect there to be only one parameter"
    assert (
        len(list(nn.Sequential(foo, rYLayer(2)).named_parameters())) == 2
    ), "there have to be two separate parameters here"


def test_backwards_call():
    temp = PostselectLayer(0, 0).forward(crYLayer(1, 2).T.forward(ket0(3)))[0][1][0]
    temp.backward()


def test_compatibility_with_qiskit():
    psi = normalize(tensor([1.0, 0.0, 1, 1.0, 0, 0.0, 0, 1.0]).reshape(2, 2, 2))
    assert equal(probabilities(psi), tensor([0.2500, 0.0000, 0.2500, 0.2500, 0.0000, 0.0000, 0.0000, 0.2500]),)

    layer1 = crYLayer([1], 2, initial_φ=pi / 4)
    layer2 = rYLayer(2, initial_θ=pi / 8)
    layer3 = cmiYLayer(2, 0)

    psi = layer1(psi)
    assert approx_equal(psi, tensor([0.5000, 0.0000, 0.2706, 0.6533, 0.0000, 0.0000, -0.1913, 0.4619]))
    psi = layer2(psi)
    assert approx_equal(psi, tensor([0.4904, 0.0975, 0.1379, 0.6935, 0.0000, 0.0000, -0.2778, 0.4157]))
    psi = layer3(psi)
    assert approx_equal(psi, tensor([0.4904, 0.0000, 0.1379, -0.4157, 0.0000, 0.0975, -0.2778, 0.6935]))
    psi = layer2.T(psi)
    assert approx_equal(psi, tensor([0.4810, -0.0957, 0.0542, -0.4347, 0.0190, 0.0957, -0.1371, 0.7344]))
    psi = layer1.T(psi)
    assert approx_equal(psi, tensor([0.4810, -0.0957, -0.1163, -0.4223, 0.0190, 0.0957, 0.1543, 0.7310]))
    psi = PostselectLayer(2, on=0).forward(psi)
    assert approx_equal(probabilities(psi), tensor([0.8599, 0.0000, 0.0502, 0.0000, 0.0013, 0.0000, 0.0885, 0.0000]),)


# RVQE
from .model import *


def test_rvqe_cell():
    assert len(list(RVQECell(6, 3).parameters())) == 6 * 3 + 6 * (6 - 1) * 3 + 6 * 3  # == 6*(6+1)*3
    assert len(list(RVQECell(8, 2).parameters())) == 8 * 9 * 2
    temp = RVQECell(3, 1).forward(ket0(5), [1])[0][0]
    temp.backward()


# DATA
from .data import *


def test_data():
    assert equal(bitword_to_onehot(tensor([0, 0, 1]), 3), tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),)
    assert bitword_to_int(tensor([0, 1, 1])) == 3
    assert int_to_bitword(12, 10) == [0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
    assert bitword_to_str([1, 1, 1, 0]) == "1110"
    assert bitword_to_str(char_to_bitword("c", "abc", 3)) == "010"
    assert bitword_to_str(char_to_bitword("c", "abc", 1)) == "0"
