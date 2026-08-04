"""
Minimal demo: ConvLSTM-based change detection on SYNTHETIC data.

This is a small, independent illustration of the core spatio-temporal
building block described in:

    Zhanabatyrova, A., Xiao, Y., & Souza Leite, C. (2023).
    "Detecting and Classifying Changes in Traffic Rules using
    Induction Loop Data." IEEE BigData 2023.

IMPORTANT - please read before using this in a portfolio/repo:
  * This is NOT the paper's original implementation or dataset.
  * It does NOT reproduce the paper's reported results (F1 > 80%).
  * Data here is entirely synthetic (random noise + injected shapes),
    generated only to demonstrate that a ConvLSTM-based detector can
    be built, trained, and can learn a simple synthetic task end-to-end.
  * The real training code/data belong to Aalto University and are
    not published here.

Run with:
    pip install torch
    python demo_convlstm_synthetic.py

Expected behavior: printed loss should decrease over ~30 epochs
(starting around 0.6-0.7, dropping toward < 0.15) since the task
is deliberately easy (synthetic, low-noise).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------
# 1. A minimal hand-written ConvLSTM cell
#    (PyTorch has no built-in ConvLSTM2D, unlike TensorFlow/Keras,
#     so this is implemented directly - same idea as the paper's
#     stateful ConvLSTM2D bottleneck, simplified to a single cell.)
# ---------------------------------------------------------------------
class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        # One conv produces all 4 gates at once (input, forget, output, candidate)
        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.hidden_channels = hidden_channels

    def forward(self, x, h_prev, c_prev):
        # x: [B, C_in, H, W]   h_prev, c_prev: [B, hidden, H, W]
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c


# ---------------------------------------------------------------------
# 2. Small encoder -> ConvLSTM -> decoder model
#    (a much-simplified stand-in for the paper's 3D U-Net + stateful
#     ConvLSTM2D + residual connections + binary-relevance heads)
# ---------------------------------------------------------------------
class MiniChangeDetector(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.conv_lstm = ConvLSTMCell(hidden_channels, hidden_channels)
        self.decoder = nn.Conv2d(hidden_channels, 1, kernel_size=1)  # 1 class, demo only

    def forward(self, x_seq):
        # x_seq: [B, T, C, H, W]
        B, T, C, H, W = x_seq.shape
        h = torch.zeros(B, self.conv_lstm.hidden_channels, H, W, device=x_seq.device)
        c = torch.zeros_like(h)

        for t in range(T):
            feat = self.encoder(x_seq[:, t])   # [B, hidden, H, W]
            h, c = self.conv_lstm(feat, h, c)  # stateful update across time steps

        logits = self.decoder(h)  # [B, 1, H, W] — change probability map (pre-sigmoid)
        return logits


# ---------------------------------------------------------------------
# 3. Synthetic data generator
#    Mimics the paper's tensor layout: [time, height, width, channels]
#    channels = (occupancy, speed, road_mask) -> here just 3 noisy channels
# ---------------------------------------------------------------------
def generate_batch(batch_size=8, seq_len=3, grid=32, anomaly_rate=0.5):
    x = torch.randn(batch_size, seq_len, 3, grid, grid) * 0.1 + 0.5  # baseline "traffic"
    y = torch.zeros(batch_size, 1, grid, grid)

    for b in range(batch_size):
        if torch.rand(1).item() < anomaly_rate:
            # Inject a rectangular "change" region in the LAST time step only,
            # simulating a sudden shift in occupancy/speed at that location.
            size = torch.randint(4, 10, (1,)).item()
            top = torch.randint(0, grid - size, (1,)).item()
            left = torch.randint(0, grid - size, (1,)).item()

            x[b, -1, :, top:top + size, left:left + size] += 2.0  # sharp value shift
            y[b, 0, top:top + size, left:left + size] = 1.0        # ground-truth label

    return x, y


# ---------------------------------------------------------------------
# 4. Training loop
# ---------------------------------------------------------------------
def train(epochs=30, lr=5e-4):
    model = MiniChangeDetector()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))

    print(f"{'Epoch':>6} | {'Loss':>8}")
    print("-" * 18)

    for epoch in range(1, epochs + 1):
        x, y = generate_batch()
        logits = model(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0 or epoch == 1:
            print(f"{epoch:>6} | {loss.item():>8.4f}")

    return model


if __name__ == "__main__":
    torch.manual_seed(0)
    train()
