# Method — Pseudocode Overview

This document describes, at an algorithmic level, the method proposed in:

> Zhanabatyrova, A., Xiao, Y., & Souza Leite, C. (2023). *Detecting and
> Classifying Changes in Traffic Rules using Induction Loop Data.*
> IEEE International Conference on Big Data (BigData 2023).

It mirrors the structure of Section III of the paper and Figure 1
(Training Pipeline Overview). This is a **structural summary**, not
an implementation — variable names follow the paper's own notation
(`X`, `Y`, `t,i,j,s,c`) so it can be read alongside the manuscript.
The underlying training code is not published here, as it was
developed as part of university-funded research.

---

## 1. Input Representation

The raw induction loop measurements (road occupancy, average vehicle
speed) are organized into a 4D tensor `X = (x_{t,i,j,s})`:

```
t : time step index          (temporal axis)
i, j : spatial coordinates    (height, width — grid over the city)
s : sensing channel           (occupancy, speed, road-network mask)
```

The road-network mask channel is a static binary layer (1 = road,
0 = non-road) that helps the network learn spatial structure.

The target is a 4D tensor of per-class change probabilities
`Y_hat = (y_hat_{t,i,j,c})`, one independent probability per class
`c` (the seven change types), since a region can belong to more
than one change class at once (binary relevance formulation).

```python
def build_input_tensor(occupancy, speed, road_mask):
    """
    occupancy, speed : [T, H, W]  raw sensor readings
    road_mask        : [H, W]     static binary mask
    returns X         : [T, H, W, 3]
    """
    X = stack_channels(occupancy, speed, broadcast(road_mask, T))
    return X
```

---

## 2. Data Preprocessing — Multi-Pooling Down-sampling

Raw resolution is far too sparse and large to learn from directly
(~83% of the grid is non-road). Down-sampling combines **three**
pooling operators to avoid losing extreme-value information the way
a single pooling method would:

```python
def multi_pool_downsample(X, kernel=4, stride=4):
    """
    X : [T, H, W, 3]  channels = (occupancy, speed, road_mask)
    Applies avg / max / min pooling to occupancy & speed,
    and max-only pooling to the static road mask.
    Returns a tensor with 6 channels (was 3) at 1/4 spatial resolution.
    """
    occ, spd, mask = split_channels(X)

    occ_pooled = [avg_pool(occ, kernel, stride),
                  max_pool(occ, kernel, stride),
                  min_pool(occ, kernel, stride)]

    spd_pooled = [avg_pool(spd, kernel, stride),
                  max_pool(spd, kernel, stride),
                  min_pool(spd, kernel, stride)]

    mask_pooled = max_pool(mask, kernel, stride)  # avg/min would erode sparse roads

    return concat_channels(*occ_pooled, *spd_pooled, mask_pooled)
    # spatial size: 2048x2048 -> 512x512 (paper's Luxembourg setup)
    # channels:     3 -> 7 total (6 sensor + 1 mask)
```

---

## 3. 3D Overlapping Sliding Windows + Augmentation

The down-sampled tensor is still too large to process as a whole
city at once. It is split into overlapping spatio-temporal windows,
which also acts as a form of context propagation between
neighboring regions/time steps.

```python
def sliding_windows(X, Y, window_hw=128, overlap_hw=0.5,
                     window_t=3, overlap_t=1):
    """
    Extracts overlapping sub-volumes along (time, height, width).
    window_hw   : spatial window size (e.g. 128x128)
    overlap_hw  : fractional spatial overlap (e.g. 50%)
    window_t    : number of time steps per window (e.g. 3)
    overlap_t   : number of overlapping time steps between windows
    """
    windows = []
    for t_start in stride_range(X.T, window_t, overlap_t):
        for i_start in stride_range(X.H, window_hw, overlap_hw):
            for j_start in stride_range(X.W, window_hw, overlap_hw):
                x_win = crop(X, t_start, i_start, j_start, window_t, window_hw)
                y_win = crop(Y, t_start, i_start, j_start, window_t, window_hw)
                windows.append((x_win, y_win))
    return windows


def augment(x_win, y_win, prob=0.3):
    """
    Applied BEFORE windowing (on the full unsegmented tensor) so that
    spatial/temporal consistency is preserved across the augmented copy.
    """
    if random() < prob:
        angle = random_choice([90, 180, 270])
        x_win = rotate_spatial(x_win, angle)
        y_win = rotate_spatial(y_win, angle)
    return x_win, y_win
```

---

## 4. Model — Modified 3D U-Net + Stateful ConvLSTM

A U-Net backbone is extended from 2D to 3D (time as an extra axis),
with residual connections in every conv/deconv block and a stateful
`ConvLSTM2D` bottleneck to capture temporal dependencies before
decoding. Output uses a **binary-relevance** head: one independent
sigmoid classifier per change class.

```python
def build_model(input_shape, num_classes):
    x = input_tensor(input_shape)          # [T, H, W, C]
    skips = []

    # --- Encoder: 3D conv blocks w/ residual connections ---
    for filters in encoder_filter_sizes:
        x = residual_conv3d_block(x, filters)
        skips.append(x)
        x = downsample3d(x)

    # --- Bottleneck: stateful ConvLSTM2D for temporal dependencies ---
    x = conv_lstm_2d(x, stateful=True)
    # stateful=True: memory persists across sequential batches;
    # requires non-shuffled, sequence-ordered training data.

    # --- Decoder: transposed-conv upsampling + skip connections ---
    for filters, skip in zip(reversed(decoder_filter_sizes), reversed(skips)):
        x = upsample3d_transposed_conv(x, filters)
        x = concat(x, skip)
        x = residual_conv3d_block(x, filters)

    # --- Binary-relevance output heads (one per change class) ---
    outputs = [sigmoid_conv_head(x, name=f"class_{c}")
               for c in range(num_classes)]

    return Model(inputs=x, outputs=outputs)
```

---

## 5. Loss Function — Class-Imbalance-Aware Weighted BCE

Because "no change" vastly outnumbers "change" instances, a random
subsampling mask is generated per training step so the loss ignores
a random subset of majority-class (no-change, on-road) samples,
inducing pseudo-class-balance without discarding data permanently.

Following the paper's notation (Eqs. 1–2):

```python
def compute_loss_weights(Y, road_mask):
    """
    Y         : [T, H, W, C] ground-truth one-hot change tensor
    road_mask : [H, W]       1 = road, 0 = non-road

    A : random tensor, same shape as Y, entries ~ Uniform(0, 1)
    B : mask isolating "road present AND no change" locations
    D = A * B
    k_c   : number of positive (change) samples for class c
    gamma_c : the k_c-th largest value in D[..., c]
    W = 1 where D > gamma_c, else 0   (Eq. 1)
    """
    A = uniform_random_like(Y)
    B = (1 - Y) * broadcast(road_mask)          # "no change" & "on road"
    D = A * B

    W = zeros_like(Y)
    for c in range(Y.num_classes):
        k_c = count_positive(Y[..., c])
        gamma_c = kth_largest(D[..., c], k_c)
        W[..., c] = (D[..., c] > gamma_c)
    return W


def weighted_bce_loss(Y, Y_hat, W):
    """
    Eq. 2 — binary cross-entropy, masked/weighted per component.
    N = num_time_steps * spatial_size
    """
    per_component = W * (Y * log(Y_hat) + (1 - Y) * log(1 - Y_hat))
    return -mean(per_component)  # summed over t, i, j; normalized by N
```

---

## 6. Prediction Confidence Weighing (Inference-time)

At inference, predictions are aggregated over a rolling time window
(paper uses 4 hours), weighted by traffic occupancy — higher
occupancy = more informative signal = more trustworthy prediction.

```python
def confidence_weighted_prediction(y_hat_sequence, occupancy_sequence,
                                     window_hours=4):
    """
    Eq. 3 — weighted average of predictions over a time window,
    weighted by average road occupancy at each time step.

    y_hat_sequence   : predictions for each t in the window
    occupancy_sequence : average occupancy o_{t,i,j} for each t
    """
    weighted_sum = sum(y_hat * occ for y_hat, occ in
                        zip(y_hat_sequence, occupancy_sequence))
    total_weight = sum(occupancy_sequence)
    return weighted_sum / total_weight
```

---

## 7. Training Loop (High Level)

```python
def train(model, train_windows, val_windows, epochs=25,
          lr=5e-4, beta1=0.9, beta2=0.999):
    optimizer = Adam(lr=lr, beta_1=beta1, beta_2=beta2)

    for epoch in range(epochs):
        for X_batch, Y_batch in ordered_batches(train_windows):
            # NOTE: batches must preserve sequence order —
            # required for the stateful ConvLSTM component.
            W_batch = compute_loss_weights(Y_batch, road_mask)

            with gradient_tape() as tape:
                Y_hat = model(X_batch)
                loss = weighted_bce_loss(Y_batch, Y_hat, W_batch)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply(grads, model.trainable_variables)

        validate(model, val_windows, metric="F1")
```

---

## 8. Evaluation

```python
def evaluate(model, test_windows):
    """
    F1-score (harmonic mean of precision & recall), chosen for its
    robustness to severe class imbalance. For multi-class results,
    a support-weighted average F1 is reported across the 7 change
    types.
    """
    predictions, ground_truth = [], []
    for X_batch, Y_batch in test_windows:
        Y_hat = model(X_batch)
        predictions.append(threshold(Y_hat))
        ground_truth.append(Y_batch)

    return support_weighted_f1(ground_truth, predictions)
```

---

## Reference — Key Hyperparameters (from the paper)

| Parameter | Value |
|---|---|
| Framework | Python 3.8.10, TensorFlow 2.12.0 |
| Optimizer | Adam (lr=5e-4, β₁=0.9, β₂=0.999) |
| Dropout | 0.3 |
| Max epochs | 25 |
| Down-sampling kernel/stride | 4×4 |
| Sliding window (spatial) | 128×128, 50% overlap |
| Sliding window (temporal) | 3 time steps, 1-step overlap |
| Temporal resolution | 2 minutes/step |
| Spatial resolution (input) | ~37.18 m²/component |
| Spatial resolution (ground truth) | 2.5 km² |
| Augmentation probability | 30% (rotation: 90°/180°/270°) |
| Confidence-weighing window | 4 hours |
