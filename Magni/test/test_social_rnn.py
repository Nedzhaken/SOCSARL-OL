import torch
import pytest
from Magni.src.social_rnn import SocialRNN

def test_forward_output_shape():
    batch_size = 4
    seq_len = 10
    input_size = 2
    hidden_size = 16
    output_size = 1

    model = SocialRNN(input_size, hidden_size, output_size)

    x = torch.randn(batch_size, seq_len, input_size)
    y = model(x)

    assert y.shape == (batch_size, seq_len, output_size)

def test_output_range():
    model = SocialRNN(input_size=2, hidden_size=16, output_size=1)

    x = torch.randn(2, 5, 2)
    y = model(x)

    assert torch.all(y >= 0)
    assert torch.all(y <= 1)

def test_backward_pass():
    model = SocialRNN(input_size=2, hidden_size=16, output_size=1)

    x = torch.randn(3, 7, 2)
    y = model(x)

    loss = y.mean()
    loss.backward()

    # Ensure at least one parameter received gradients
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None for g in grads)

def test_single_sequence():
    model = SocialRNN(input_size=2, hidden_size=16, output_size=1)

    x = torch.randn(1, 1, 2)
    y = model(x)

    assert y.shape == (1, 1, 1)