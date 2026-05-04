# deep-learning-using-pytorch

### Purpose: 
A PyTorch-focused learning repository of annotated notebooks that teach core deep learning concepts (theory + hands-on code), practical training recipes, and applied CV examples.

### Core contents:
- Fundamentals: tensors, autograd, training loop, model/nn.Module, saving/loading.
- Training & optimization: optimizers, LR schedules, momentum, sharp vs flat minima, initialization, vanishing/exploding gradients.
- CNNs & vision: conv math, pooling, feature hierarchy, building/training CNNs (MNIST/CIFAR), transfer learning (ResNet), differential fine-tuning.
- Best practices: validation, checkpoints, early stopping, debugging, batchnorm, residual connections.

### Quick start

##### Highly Recommended 
- Open the notebooks in Google colab or Kaggle notebook

###### Optional
- Create a Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

- Run a notebook in `week*/` using Jupyter, or run a quick training example:

```bash
python scripts/train.py --dataset mnist --epochs 1
```

Repository layout
- `week1/` … `week6/` — existing notebooks (theory + exercises)
- `data/` — dataset downloads (ignored by default)
- `models/` — saved checkpoints (.pth)
- `scripts/` — small utilities: `train.py`, `data.py`, `eval.py`
- `requirements.txt` — python dependencies
