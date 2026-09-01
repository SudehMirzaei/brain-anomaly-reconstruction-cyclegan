# Adversarial Training in Medical Image Analysis: A Comprehensive Guide

## Table of Contents

1. [Introduction](#introduction)
2. [Fundamentals of Adversarial Training](#fundamentals-of-adversarial-training)
3. [Generative Adversarial Networks (GANs)](#generative-adversarial-networks-gans)
4. [Adversarial Attacks and Defenses](#adversarial-attacks-and-defenses)
5. [Adversarial Training in Medical Imaging](#adversarial-training-in-medical-imaging)
6. [Implementation in Brain Anomaly Reconstruction](#implementation-in-brain-anomaly-reconstruction)
7. [Training Strategies and Best Practices](#training-strategies-and-best-practices)
8. [Evaluation Metrics](#evaluation-metrics)
9. [Common Challenges and Solutions](#common-challenges-and-solutions)
10. [Code Examples](#code-examples)
11. [References](#references)

---

## Introduction

Adversarial training represents a paradigm shift in machine learning where models are trained not just on clean data, but also on adversarial examples designed to challenge and improve model robustness. In medical imaging, adversarial training serves dual purposes: generating realistic synthetic images and defending against potential attacks on diagnostic systems.

### Why Adversarial Training Matters in Medical Imaging

- **Robustness:** Medical AI systems must be reliable under various conditions
- **Data scarcity:** Adversarial generation can create realistic synthetic medical images
- **Domain adaptation:** Transfer knowledge across different scanners and protocols
- **Anomaly detection:** Learn to distinguish normal from abnormal patterns
- **Privacy preservation:** Generate synthetic data without patient information

## Fundamentals of Adversarial Training

### The Adversarial Game

Adversarial training can be understood as a two-player minimax game:

```
min_G max_D V(D, G) = E[log D(x)] + E[log(1 - D(G(z)))]
```

Where:

- **Generator (G):** Creates synthetic data to fool the discriminator
- **Discriminator (D):** Distinguishes real from synthetic data
- **Value Function (V):** Represents the game's objective

### Types of Adversarial Training

```python
# Different adversarial training paradigms

# 1. Standard GAN Training
def standard_gan_training():
    for epoch in range(epochs):
        # Train discriminator
        real_loss = discriminator_loss(real_images, real_labels=1)
        fake_loss = discriminator_loss(generated_images, real_labels=0)
        d_loss = real_loss + fake_loss
        
        # Train generator
        g_loss = generator_loss(discriminator(generated_images), real_labels=1)

# 2. Adversarial Robustness Training
def robustness_training():
    for epoch in range(epochs):
        # Generate adversarial examples
        adv_examples = generate_adversarial_examples(model, data, epsilon=0.1)
        
        # Train on both clean and adversarial examples
        loss = model_loss(data, labels) + lambda * model_loss(adv_examples, labels)

# 3. Domain Adversarial Training
def domain_adversarial_training():
    for epoch in range(epochs):
        # Feature extraction
        features = feature_extractor(source_data)
        
        # Domain classification (adversarial)
        domain_loss = domain_classifier(features, domain_labels)
        
        # Task-specific loss
        task_loss = task_classifier(features, task_labels)
        
        # Gradient reversal for domain adaptation
        total_loss = task_loss - lambda * domain_loss
```

## Generative Adversarial Networks (GANs)

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    GAN Architecture                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Random Noise ──→ Generator ──→ Synthetic Image ──┐       │
│                                                     │       │
│   Real Image ──────────────────────────────┐       │       │
│                                            ↓       ↓       │
│                                    Discriminator           │
│                                            │               │
│                                     Real / Fake            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### GAN Variants for Medical Imaging

| Variant   | Key Features                    | Medical Application          |
|-----------|----------------------------------|------------------------------|
| DCGAN     | Deep convolutional architecture  | Basic image generation       |
| WGAN      | Wasserstein loss for stability   | Improved training stability   |
| cGAN      | Conditional generation           | Specific pathology generation  |
| CycleGAN  | Unpaired translation             | Modality conversion          |
| StyleGAN  | Style-based generation           | High-quality synthesis       |
| Progressive GAN | Progressive growing       | High-resolution generation   |

### Implementation Components

```python
import torch
import torch.nn as nn

class MedicalImageGenerator(nn.Module):
    def __init__(self, latent_dim=100, img_channels=1, img_size=256):
        super().__init__()
        self.latent_dim = latent_dim
        
        self.main = nn.Sequential(
            # Input: latent_dim x 1 x 1
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 512 x 4 x 4
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 256 x 8 x 8
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 128 x 16 x 16
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 64 x 32 x 32
            nn.ConvTranspose2d(64, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # img_channels x 64 x 64
        )
    
    def forward(self, z):
        return self.main(z)

class MedicalImageDiscriminator(nn.Module):
    def __init__(self, img_channels=1, img_size=256):
        super().__init__()
        
        self.main = nn.Sequential(
            # Input: img_channels x 64 x 64
            nn.Conv2d(img_channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 64 x 32 x 32
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # 128 x 16 x 16
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # 256 x 8 x 8
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # 512 x 4 x 4
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # 1 x 1 x 1
        )
    
    def forward(self, img):
        return self.main(img)
```

## Adversarial Attacks and Defenses

### Types of Adversarial Attacks

```python
# Common adversarial attack methods

# 1. Fast Gradient Sign Method (FGSM)
def fgsm_attack(model, image, label, epsilon=0.1):
    image.requires_grad = True
    output = model(image)
    loss = nn.CrossEntropyLoss()(output, label)
    model.zero_grad()
    loss.backward()
    
    perturbed_image = image + epsilon * image.grad.sign()
    return torch.clamp(perturbed_image, 0, 1)

# 2. Projected Gradient Descent (PGD)
def pgd_attack(model, image, label, epsilon=0.1, alpha=0.01, iterations=40):
    perturbed_image = image.clone().detach()
    
    for _ in range(iterations):
        perturbed_image.requires_grad = True
        output = model(perturbed_image)
        loss = nn.CrossEntropyLoss()(output, label)
        model.zero_grad()
        loss.backward()
        
        with torch.no_grad():
            perturbed_image = perturbed_image + alpha * perturbed_image.grad.sign()
            perturbed_image = torch.clamp(perturbed_image, 
                                           image - epsilon, 
                                           image + epsilon)
            perturbed_image = torch.clamp(perturbed_image, 0, 1)
    
    return perturbed_image

# 3. Carlini & Wagner (C&W) Attack
def cw_attack(model, image, label, c=1, iterations=1000):
    w = torch.zeros_like(image, requires_grad=True)
    optimizer = torch.optim.Adam([w], lr=0.01)
    
    for _ in range(iterations):
        perturbed_image = 0.5 * (torch.tanh(w) + 1)
        output = model(perturbed_image)
        
        # Loss formulation
        f_loss = max(output[label] - output.max(), 0)
        l2_loss = torch.norm(perturbed_image - image) ** 2
        total_loss = l2_loss + c * f_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    
    return perturbed_image
```

### Defense Mechanisms

```python
# Adversarial defense strategies

# 1. Adversarial Training
def adversarial_training(model, train_loader, epsilon=0.1):
    for images, labels in train_loader:
        # Generate adversarial examples
        adv_images = fgsm_attack(model, images, labels, epsilon)
        
        # Train on mixed batch
        mixed_images = torch.cat([images, adv_images])
        mixed_labels = torch.cat([labels, labels])
        
        output = model(mixed_images)
        loss = criterion(output, mixed_labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# 2. Input Transformation
def input_transformation_defense(image):
    # Random resizing and padding
    transformed = random_resize_pad(image)
    # JPEG compression
    transformed = jpeg_compress(transformed)
    # Random noise addition
    transformed = add_noise(transformed, sigma=0.01)
    
    return transformed

# 3. Feature Squeezing
def feature_squeezing(image, bit_depth=4):
    # Reduce color bit depth
    squeezed = quantize_bits(image, bit_depth)
    # Median filtering
    squeezed = median_filter(squeezed, kernel_size=3)
    
    return squeezed
```

## Adversarial Training in Medical Imaging

### Applications

1. **Image Reconstruction**

```python
class ReconstructionAdversarialNetwork:
    def __init__(self):
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.discriminator = Discriminator()
    
    def train_step(self, incomplete_scan, complete_scan):
        # Generate reconstruction
        reconstruction = self.decoder(self.encoder(incomplete_scan))
        
        # Adversarial loss
        real_pred = self.discriminator(complete_scan)
        fake_pred = self.discriminator(reconstruction)
        
        # Combined loss
        reconstruction_loss = mse_loss(reconstruction, complete_scan)
        adversarial_loss = bce_loss(fake_pred, torch.ones_like(fake_pred))
        
        total_loss = reconstruction_loss + 0.1 * adversarial_loss
        
        return total_loss
```

2. **Lesion Detection**

```python
class LesionDetectionGAN:
    def __init__(self):
        self.segmenter = Segmenter()
        self.discriminator = Discriminator()
    
    def train_step(self, image, ground_truth_mask):
        # Generate segmentation
        pred_mask = self.segmenter(image)
        
        # Adversarial training for realistic segmentations
        real_score = self.discriminator(ground_truth_mask)
        fake_score = self.discriminator(pred_mask)
        
        # Dice loss + adversarial loss
        dice_loss = 1 - dice_coefficient(pred_mask, ground_truth_mask)
        adv_loss = bce_loss(fake_score, torch.ones_like(fake_score))
        
        return dice_loss + 0.01 * adv_loss
```

3. **Domain Adaptation**

```python
class DomainAdversarialNetwork:
    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.task_classifier = TaskClassifier()
        self.domain_classifier = DomainClassifier()
    
    def train_step(self, source_data, target_data):
        # Extract features
        source_features = self.feature_extractor(source_data)
        target_features = self.feature_extractor(target_data)
        
        # Task loss (source domain)
        task_output = self.task_classifier(source_features)
        task_loss = cross_entropy(task_output, source_labels)
        
        # Domain adversarial loss
        domain_output = self.domain_classifier(
            torch.cat([source_features, target_features])
        )
        domain_labels = torch.cat([
            torch.zeros(source_features.size(0)),
            torch.ones(target_features.size(0))
        ])
        domain_loss = cross_entropy(domain_output, domain_labels)
        
        # Gradient reversal for feature extractor
        total_loss = task_loss - lambda * domain_loss
        
        return total_loss
```

## Implementation in Brain Anomaly Reconstruction

### Architecture Design

```python
class BrainAnomalyGAN:
    def __init__(self):
        # Generator: Anomaly → Healthy
        self.G_anomaly_to_healthy = ResNetGenerator()
        
        # Generator: Healthy → Anomaly
        self.G_healthy_to_anomaly = ResNetGenerator()
        
        # Discriminators
        self.D_healthy = PatchGANDiscriminator()
        self.D_anomaly = PatchGANDiscriminator()
        
        # Anomaly detector
        self.anomaly_detector = AnomalyDetector()
    
    def forward(self, anomalous_scan):
        # Generate healthy reconstruction
        healthy_reconstruction = self.G_anomaly_to_healthy(anomalous_scan)
        
        # Detect anomaly via residual
        residual = torch.abs(anomalous_scan - healthy_reconstruction)
        anomaly_map = self.anomaly_detector(residual)
        
        return healthy_reconstruction, anomaly_map
    
    def adversarial_loss(self, real_images, fake_images, discriminator):
        # Standard GAN loss
        real_pred = discriminator(real_images)
        fake_pred = discriminator(fake_images)
        
        real_loss = bce_loss(real_pred, torch.ones_like(real_pred))
        fake_loss = bce_loss(fake_pred, torch.zeros_like(fake_pred))
        
        return (real_loss + fake_loss) / 2
```

### Training Strategy

```python
def train_brain_anomaly_gan(model, healthy_loader, anomaly_loader, epochs=200):
    for epoch in range(epochs):
        for (healthy, anomaly) in zip(healthy_loader, anomaly_loader):
            # Phase 1: Train discriminators
            # Healthy discriminator
            fake_healthy = model.G_anomaly_to_healthy(anomaly)
            d_healthy_loss = model.adversarial_loss(
                healthy, fake_healthy, model.D_healthy
            )
            
            # Anomaly discriminator
            fake_anomaly = model.G_healthy_to_anomaly(healthy)
            d_anomaly_loss = model.adversarial_loss(
                anomaly, fake_anomaly, model.D_anomaly
            )
            
            # Phase 2: Train generators
            # Cycle consistency
            cycle_healthy = model.G_anomaly_to_healthy(fake_anomaly)
            cycle_anomaly = model.G_healthy_to_anomaly(fake_healthy)
            
            cycle_loss = l1_loss(cycle_healthy, healthy) + \
                        l1_loss(cycle_anomaly, anomaly)
            
            # Adversarial loss for generators
            g_healthy_loss = bce_loss(
                model.D_healthy(fake_healthy),
                torch.ones_like(model.D_healthy(fake_healthy))
            )
            g_anomaly_loss = bce_loss(
                model.D_anomaly(fake_anomaly),
                torch.ones_like(model.D_anomaly(fake_anomaly))
            )
            
            # Identity loss
            identity_loss = l1_loss(
                model.G_anomaly_to_healthy(healthy), healthy
            ) + l1_loss(
                model.G_healthy_to_anomaly(anomaly), anomaly
            )
            
            # Total generator loss
            g_loss = g_healthy_loss + g_anomaly_loss + \
                    10 * cycle_loss + 5 * identity_loss
            
            # Update weights
            update_discriminators(d_healthy_loss, d_anomaly_loss)
            update_generators(g_loss)
```

## Training Strategies and Best Practices

### Hyperparameter Optimization

```python
# Recommended hyperparameters for medical imaging GANs

config = {
    'learning_rate': 0.0002,
    'beta1': 0.5,
    'beta2': 0.999,
    'batch_size': 1,  # Small batches for high-resolution images
    'lambda_cycle': 10.0,  # Cycle consistency weight
    'lambda_identity': 5.0,  # Identity loss weight
    'lambda_adversarial': 1.0,  # Adversarial loss weight
    'discriminator_update_freq': 1,  # Update frequency
    'generator_update_freq': 1,
    'gradient_clip_value': 0.01,  # Gradient clipping
    'spectral_norm': True,  # Use spectral normalization
}
```

### Loss Function Selection

| Loss Type         | Formula                                           | Use Case                     |
|-------------------|---------------------------------------------------|------------------------------|
| Minimax Loss      | min_G max_D E[log D(x)] + E[log(1-D(G(z)))]     | Original GAN                 |
| LSGAN Loss        | min_D E[(D(x)-1)²] + E[D(G(z))²]                 | More stable training         |
| WGAN Loss         | min_G max_D E[D(x)] - E[D(G(z))]                 | Improved convergence         |
| Hinge Loss        | min_D E[max(0, 1-D(x))] + E[max(0, 1+D(G(z)))]   | SOTA results                 |
| Dice Loss         | `1 - 2* X∩Y`                                     | Segmentation tasks           |

### Training Stability Techniques

```python
def stabilize_training(model, optimizer):
    # 1. Gradient penalty (WGAN-GP)
    def gradient_penalty(discriminator, real, fake):
        alpha = torch.rand(real.size(0), 1, 1, 1).to(real.device)
        interpolated = alpha * real + (1 - alpha) * fake
        interpolated.requires_grad = True
        
        d_interpolated = discriminator(interpolated)
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True
        )[0]
        
        gradient_norm = gradients.view(gradients.size(0), -1).norm(2, dim=1)
        return ((gradient_norm - 1) ** 2).mean()
    
    # 2. Spectral normalization
    def apply_spectral_norm(layer):
        return nn.utils.spectral_norm(layer)
    
    # 3. Learning rate scheduling
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: 1.0 - max(0, epoch - 100) / 100
    )
    
    return gradient_penalty, scheduler
```

## Evaluation Metrics

### Quantitative Metrics

```python
def evaluate_gan(generator, test_loader):
    metrics = {}
    
    # 1. Fréchet Inception Distance (FID)
    def calculate_fid(real_images, fake_images):
        # Extract features using InceptionV3
        real_features = inception_model(real_images)
        fake_features = inception_model(fake_images)
        
        # Calculate statistics
        mu_real, sigma_real = real_features.mean(0), real_features.cov()
        mu_fake, sigma_fake = fake_features.mean(0), fake_features.cov()
        
        # FID calculation
        diff = mu_real - mu_fake
        fid = diff @ diff + trace(sigma_real + sigma_fake - 
                                  2 * sqrt(sigma_real @ sigma_fake))
        return fid
    
    # 2. Structural Similarity Index (SSIM)
    def calculate_ssim(real, fake):
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        
        mu_x = gaussian_filter(real, 1.5)
        mu_y = gaussian_filter(fake, 1.5)
        
        sigma_x = gaussian_filter(real ** 2, 1.5) - mu_x ** 2
        sigma_y = gaussian_filter(fake ** 2, 1.5) - mu_y ** 2
        sigma_xy = gaussian_filter(real * fake, 1.5) - mu_x * mu_y
        
        ssim = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x ** 2 + mu_y ** 2 + C1) * 
                (sigma_x + sigma_y + C2))
        
        return ssim.mean()
    
    # 3. Peak Signal-to-Noise Ratio (PSNR)
    def calculate_psnr(real, fake):
        mse = torch.mean((real - fake) ** 2)
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
        return psnr
    
    return metrics
```

### Qualitative Assessment

```python
def qualitative_evaluation(model, test_images, save_dir):
    # Visualize reconstructions
    fig, axes = plt.subplots(3, len(test_images), figsize=(15, 10))
    
    for i, image in enumerate(test_images):
        # Original image
        axes[0, i].imshow(image[0], cmap='gray')
        axes[0, i].set_title('Original')
        
        # Reconstructed image
        reconstruction = model(image.unsqueeze(0))
        axes[1, i].imshow(reconstruction[0, 0], cmap='gray')
        axes[1, i].set_title('Reconstruction')
        
        # Anomaly map
        anomaly_map = torch.abs(image - reconstruction[0])
        axes[2, i].imshow(anomaly_map[0], cmap='hot')
        axes[2, i].set_title('Anomaly Map')
    
    plt.savefig(f'{save_dir}/qualitative_results.png')
```

## Common Challenges and Solutions

### Challenge 1: Mode Collapse

**Problem:** Generator produces limited variety of outputs

**Solutions:**

```python
# 1. Minibatch discrimination
class MinibatchDiscrimination(nn.Module):
    def __init__(self, in_features, out_features, num_kernels=100):
        super().__init__()
        self.T = nn.Parameter(torch.randn(in_features, out_features, num_kernels))
    
    def forward(self, x):
        matrices = x.mm(self.T.view(self.T.size(0), -1))
        matrices = matrices.view(-1, self.T.size(1), self.T.size(2))
        
        # Calculate L1 distance between samples
        diffs = matrices.unsqueeze(0) - matrices.unsqueeze(1)
        abs_diffs = torch.abs(diffs).sum(2)
        
        # Apply exponential
        minibatch_features = torch.exp(-abs_diffs)
        return torch.cat([x, minibatch_features], 1)

# 2. Unrolled GANs
def unrolled_generator_loss(generator, discriminator, real_data, num_unroll=5):
    # Unroll discriminator updates
    for _ in range(num_unroll):
        discriminator.zero_grad()
        d_loss = discriminator_loss(discriminator, generator(noise), real_data)
        d_loss.backward()
        discriminator_optimizer.step()
    
    # Calculate generator loss with unrolled discriminator
    g_loss = generator_loss(discriminator, generator(noise))
    return g_loss
```

### Challenge 2: Training Instability

**Solutions:**

```python
# 1. Feature matching loss
def feature_matching_loss(real_features, fake_features):
    loss = 0
    for real_f, fake_f in zip(real_features, fake_features):
        loss += torch.mean(torch.abs(real_f - fake_f))
    return loss

# 2. Label smoothing
def smooth_labels(labels, smoothing=0.1):
    return labels * (1 - smoothing) + 0.5 * smoothing

# 3. Two time-scale update rule (TTUR)
generator_optimizer = Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
discriminator_optimizer = Adam(discriminator.parameters(), lr=0.0004, betas=(0.5, 0.999))
```

### Challenge 3: Evaluation Difficulties

**Solutions:**

```python
# Multi-metric evaluation framework
def comprehensive_evaluation(generator, test_loader):
    results = {}
    
    # Image quality metrics
    results['fid'] = calculate_fid(generator, test_loader)
    results['is'] = calculate_inception_score(generator, test_loader)
    results['lpips'] = calculate_lpips(generator, test_loader)
    
    # Task-specific metrics
    results['dice'] = calculate_dice_coefficient(generator, test_loader)
    results['sensitivity'] = calculate_sensitivity(generator, test_loader)
    results['specificity'] = calculate_specificity(generator, test_loader)
    
    # Human evaluation (if available)
    if human_evaluators_available:
        results['human_score'] = collect_human_evaluations(generator)
    
    return results
```

## Code Examples

### Complete Training Pipeline

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class MedicalGANPipeline:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize models
        self.generator = MedicalImageGenerator().to(self.device)
        self.discriminator = MedicalImageDiscriminator().to(self.device)
        
        # Initialize optimizers
        self.g_optimizer = optim.Adam(
            self.generator.parameters(),
            lr=config['learning_rate'],
            betas=(config['beta1'], config['beta2'])
        )
        self.d_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=config['learning_rate'],
            betas=(config['beta1'], config['beta2'])
        )
        
        # Loss functions
        self.criterion = nn.BCELoss()
        self.l1_loss = nn.L1Loss()
        
        # Fixed noise for visualization
        self.fixed_noise = torch.randn(64, config['latent_dim'], 1, 1).to(self.device)
    
    def train_epoch(self, dataloader):
        self.generator.train()
        self.discriminator.train()
        
        g_losses = []
        d_losses = []
        
        for batch_idx, (real_images, _) in enumerate(dataloader):
            batch_size = real_images.size(0)
            real_images = real_images.to(self.device)
            
            # Labels
            real_labels = torch.ones(batch_size, 1).to(self.device)
            fake_labels = torch.zeros(batch_size, 1).to(self.device)
            
            # Train Discriminator
            self.d_optimizer.zero_grad()
            
            # Real images
            real_output = self.discriminator(real_images)
            d_real_loss = self.criterion(real_output, real_labels)
            
            # Fake images
            noise = torch.randn(batch_size, self.config['latent_dim'], 1, 1).to(self.device)
            fake_images = self.generator(noise)
            fake_output = self.discriminator(fake_images.detach())
            d_fake_loss = self.criterion(fake_output, fake_labels)
            
            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            self.d_optimizer.step()
            
            # Train Generator
            self.g_optimizer.zero_grad()
            
            fake_output = self.discriminator(fake_images)
            g_loss = self.criterion(fake_output, real_labels)
            
            # Optional: Add reconstruction loss if applicable
            if self.config.get('use_reconstruction_loss', False):
                g_loss += self.config['reconstruction_weight'] * \
                         self.l1_loss(fake_images, real_images)
            
            g_loss.backward()
            self.g_optimizer.step()
            
            # Record losses
            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())
            
            # Logging
            if batch_idx % self.config['log_interval'] == 0:
                print(f'Train Epoch: {self.current_epoch} '
                      f'[{batch_idx * len(real_images)}/{len(dataloader.dataset)}] '
                      f'Loss_D: {d_loss.item():.4f} '
                      f'Loss_G: {g_loss.item():.4f}')
        
        return np.mean(g_losses), np.mean(d_losses)
    
    def generate_samples(self, num_samples=16):
        self.generator.eval()
        with torch.no_grad():
            noise = torch.randn(num_samples, self.config['latent_dim'], 1, 1).to(self.device)
            samples = self.generator(noise)
        return samples.cpu()
```

### Inference and Deployment

```python
class MedicalGANInference:
    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.model = self.load_model(model_path)
        self.model.eval()
    
    def load_model(self, model_path):
        checkpoint = torch.load(model_path, map_location=self.device)
        model = MedicalImageGenerator()
        model.load_state_dict(checkpoint['generator_state_dict'])
        return model.to(self.device)
    
    def generate_medical_image(self, condition=None, noise=None):
        with torch.no_grad():
            if noise is None:
                noise = torch.randn(1, 100, 1, 1).to(self.device)
            
            if condition is not None:
                # Conditional generation
                condition = condition.to(self.device)
                generated = self.model(noise, condition)
            else:
                generated = self.model(noise)
        
        return generated.cpu().numpy()
    
    def batch_generate(self, batch_size=32):
        noise = torch.randn(batch_size, 100, 1, 1).to(self.device)
        with torch.no_grad():
            generated = self.model(noise)
        return generated.cpu().numpy()
```

## References

1. Goodfellow, I., et al. (2014). Generative Adversarial Networks. NeurIPS 2014.
2. Arjovsky, M., Chintala, S., & Bottou, L. (2017). Wasserstein GAN. ICML 2017.
3. Madry, A., et al. (2018). Towards Deep Learning Models Resistant to Adversarial Attacks. ICLR 2018.
4. Yi, X., Walia, E., & Babyn, P. (2019). Generative adversarial network in medical imaging: A review. Medical Image Analysis.
5. Kazeminia, S., et al. (2020). GANs for medical image analysis. Artificial Intelligence in Medicine.
6. Baur, C., et al. (2021). Deep Autoencoding Models for Unsupervised Anomaly Segmentation in Brain MR Images. Medical Image Analysis.
7. Chen, X., & Konukoglu, E. (2022). Unsupervised Detection of Lesions in Brain MRI using constrained adversarial auto-encoders. MIDL 2022.

