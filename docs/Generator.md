# Generator

# Overview

The Generator is the core component of the proposed Brain MRI Anomaly Reconstruction framework. Its objective is to learn the distribution of healthy brain anatomy and reconstruct anatomically plausible healthy MRI images.

Unlike a traditional image classifier, the Generator does not predict tumor categories. Instead, it receives an MRI image as input and attempts to generate a reconstructed image that follows the learned appearance of healthy brain tissue.

In this project, the Generator is implemented as an **Attention U-Net** enhanced with **Residual Blocks** and trained using **adversarial learning** with a **PatchGAN Discriminator**.

---

# Role of the Generator

The Generator learns a mapping

```
Input MRI

↓

Healthy Reconstruction
```

rather than

```
Input MRI

↓

Tumor Class
```

Its purpose is to reconstruct healthy brain anatomy while preserving normal structures.

During inference, abnormal regions cannot be reconstructed accurately because the Generator has only learned healthy anatomy.

Instead, these regions are replaced with anatomically plausible healthy tissue.

---

# Why Is a Generator Needed?

Traditional CNNs solve classification problems.

Example:

```
Brain MRI

↓

CNN

↓

Glioma
```

However, this project aims to answer a different question:

> **What would this brain look like if no tumor were present?**

This requires image generation rather than image classification.

The Generator performs this image-to-image translation.

```
Brain MRI

↓

Generator

↓

Healthy Brain MRI
```

---

# Generator Architecture

The Generator follows an encoder-decoder architecture based on Attention U-Net.

```
                Input MRI

                     │

                     ▼

                 Encoder

                     │

                     ▼

          Residual Bottleneck

                     │

                     ▼

                 Decoder

                     │

                     ▼

         Reconstructed MRI
```

Skip Connections transfer high-resolution anatomical information from the encoder to the decoder.

Attention Gates filter these features before fusion.

---

# Encoder

The encoder extracts hierarchical feature representations.

Each encoder stage performs

```
Convolution

↓

Normalization

↓

Activation

↓

Downsampling
```

As the image passes through the encoder

- spatial resolution decreases,
- feature channels increase,
- semantic information becomes richer.

Example

```
224 × 224

↓

112 × 112

↓

56 × 56

↓

28 × 28

↓

14 × 14
```

---

# Residual Bottleneck

The Bottleneck contains several Residual Blocks.

Its purpose is to learn

- global brain anatomy,
- semantic relationships,
- healthy structural distribution.

Residual learning improves

- gradient propagation,
- optimization stability,
- feature reuse.

More details:

- `docs/Models/Bottleneck.md`
- `docs/Theory/Residual_Block.md`

---

# Attention Gates

Skip Connections are filtered using Attention Gates.

Instead of forwarding every encoder feature

```
Encoder

────────────►

Decoder
```

the Generator performs

```
Encoder

↓

Attention Gate

↓

Filtered Features

↓

Decoder
```

This suppresses irrelevant background information and highlights anatomically meaningful regions.

More information:

- `docs/Models/Attention_Gate.md`

---

# Decoder

The decoder reconstructs the image.

Each decoder block performs

```
Upsampling

↓

Concatenation

↓

Convolution

↓

Activation
```

Spatial resolution gradually increases until the original image size is recovered.

```
14 × 14

↓

28 × 28

↓

56 × 56

↓

112 × 112

↓

224 × 224
```

---

# Skip Connections

The Generator preserves anatomical details using Skip Connections.

```
Encoder Features

──────────────►

Decoder
```

These connections help preserve

- tissue boundaries,
- cortical structures,
- ventricles,
- fine anatomical details.

Without Skip Connections, reconstructed images become significantly blurrier.

More details:

- `docs/Models/Skip_Connection.md`

---

# Generator During Training

Only healthy MRI images are used for training.

```
Healthy MRI

↓

Generator

↓

Healthy MRI
```

The target image is identical to the input image.

The Generator therefore learns the appearance of normal brain anatomy.

It never observes

- Glioma
- Meningioma
- Pituitary tumors

during training.

---

# Generator During Inference

During inference, an abnormal MRI is provided.

```
Tumor MRI

↓

Generator

↓

Healthy Reconstruction
```

Since the Generator has only learned healthy anatomy, abnormal structures cannot be faithfully reconstructed.

Instead, they are replaced by anatomically plausible healthy structures.

This behavior forms the foundation of reconstruction-based anomaly detection.

---

# Difference Map

After reconstruction,

the reconstructed MRI is compared with the original MRI.

```
Difference Map

=

| Input MRI − Reconstructed MRI |
```

Large reconstruction errors indicate regions that deviate from the learned healthy distribution.

Pipeline

```
Tumor MRI

↓

Generator

↓

Healthy MRI

↓

Difference Map

↓

Anomaly Localization
```

---

# Generator and PatchGAN

The Generator is trained together with a PatchGAN Discriminator.

```
Healthy MRI

↓

Generator

↓

Generated MRI

↓

PatchGAN

↓

Real / Fake
```

The Discriminator encourages the Generator to produce

- realistic textures,
- sharp anatomical boundaries,
- visually plausible reconstructions.

---

# Loss Functions

The Generator is optimized using multiple objectives.

## Reconstruction Loss

Encourages similarity between the generated MRI and the target healthy MRI.

This preserves anatomical structures.

---

## Adversarial Loss

Encourages the Generator to fool the PatchGAN Discriminator.

This improves image realism.

---

## Total Objective

The Generator minimizes a weighted combination of

- reconstruction loss,
- adversarial loss.

This balances structural accuracy with visual realism.

---

# Why Attention U-Net?

Several architectures could be used as the Generator.

Examples include

- Autoencoder
- Standard U-Net
- Variational Autoencoder
- Diffusion Models

Attention U-Net was selected because it

- preserves spatial information,
- focuses on informative regions,
- suppresses irrelevant features,
- produces higher-quality reconstructions.

---

# Advantages

- Learns healthy brain anatomy.
- Preserves anatomical structures.
- Produces high-quality reconstructions.
- Supports reconstruction-based anomaly detection.
- Integrates naturally with PatchGAN.
- Uses Attention Gates to improve feature selection.
- Uses Residual Blocks for stable optimization.

---

# Limitations

The Generator also has several limitations.

- Requires substantial GPU memory.
- Adversarial training is computationally expensive.
- Reconstruction quality depends on healthy training data.
- Small anomalies may produce subtle reconstruction errors.

---

# Role in This Project

The Generator is the central component of the proposed reconstruction framework.

Training

```
Healthy MRI

↓

Attention U-Net Generator

↓

Healthy MRI
```

Inference

```
Tumor MRI

↓

Attention U-Net Generator

↓

Healthy Reconstruction

↓

Difference Map

↓

Anomaly Localization
```

Rather than detecting tumors directly, the Generator reconstructs what the brain would likely look like if it followed the learned healthy anatomical distribution.




