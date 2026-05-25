# content.md — Summary of notebooks

This file summarizes the contents of all notebooks in the repository, organized by Week → Day with a short brief for each day.

## Week 1

### Day 1
Tensors and Autograd basics: creating tensors, `requires_grad`, manual gradients vs automatic differentiation, and retain_graph explanation.

### Day 2
Autograd internals: computation graphs, `grad_fn`, `.backward()` behavior, and `detach()` semantics.

### Day 3
`nn.Module` fundamentals: `nn.Parameter` registration, building modules, and freezing parameters (requires_grad control).

### Day 4
Training loop core: batching, forward/backward passes, importance of `zero_grad()`, and debugging gradient accumulation.

### Day 5
Validation and model persistence: overfitting demonstration, validation checks, and saving/loading `state_dict` (linear_model.pth).

## Week 2

### Day 1
Datasets and `DataLoader`: MNIST/FashionMNIST loading, transforms, and train/validation splitting.

### Day 2
MLP training on MNIST: model definitions, loss/optimizer setup, and basic training loop experiments.

### Day 3
Overfitting vs regularization: experiments demonstrating overfitting and techniques to mitigate it.

### Day 4
Optimizer and learning-rate experiments: comparing optimizers, tuning learning rates, and observing training dynamics.

### Day 5
Checkpointing and best-model tracking: saving best model weights (best_mnist_model.pth) and evaluation procedures.

## Week 3

### Day 1
Convolution fundamentals: conv layers, receptive fields, and basic feature-map intuition.

### Day 2
Building CNNs for MNIST: architecture design, convolutional blocks, and training on image data.

### Day 3
Pooling, stride and padding: effects on spatial dimensions and computation of output shapes.

### Day 4
Feature visualization: inspecting activations and intermediate feature maps to understand learned filters.

### Day 5
Training tips for CNNs: regularization, weight initialization considerations, and evaluation practices.

## Week 4

### Day 1
Transfer learning concepts: when and why to use pretrained models and feature extraction vs fine-tuning.

### Day 2
Using ResNet: replacing the classifier head, adapting pretrained `resnet` for new tasks.

### Day 3
Feature-extraction workflow: freezing backbone weights and training only the new classifier layers.

### Day 4
Fine-tuning strategies: unfreezing deeper layers (e.g., `layer4`), differential learning rates, and selective training.

### Day 5
Data augmentation and schedulers: augmentations for robustness and scheduler experiments for improved convergence.

## Week 5

### Day 1
Vanishing and exploding gradients: causes, diagnostics, and high-level mitigation strategies.

### Day 2
Batch Normalization: motivation, how it stabilizes training, and its effect on learning dynamics.

### Day 3
Residual connections: intuition behind skip connections and how they alleviate degradation in deep nets.

### Day 4
Activation functions: comparison of Sigmoid/Tanh vs ReLU variants, and why ReLU transformed deep learning.

### Day 5
Weight initialization: Xavier/Glorot and He initializations, why initialization scale matters for stable training.

## Week 6

### Day 1
Optimization algorithms: SGD vs Adam tradeoffs and practical considerations for choosing optimizers.

### Day 2
Learning-rate scheduling: step, cosine, and warmup schedules and their effects on training stability.

### Day 3
Optimization landscapes: discussion of sharp vs flat minima and implications for generalization.
