## 🧠 Variational Autoencoder (VAE) - Fashion MNIST

This project implements a **Variational Autoencoder (VAE)** using **PyTorch** to learn compressed latent representations of clothing images from the **Fashion-MNIST** dataset and reconstruct them.

---

## 📌 Features

- Custom encoder-decoder VAE architecture using `Linear` layers.
- Implements the **reparameterization trick** to allow backpropagation through the stochastic latent space.
- Loss function combines **Reconstruction Loss (MSE)** and **KL Divergence**.
- Reconstruction quality evaluation using custom pixel-wise accuracy.
- Trained on Fashion-MNIST dataset with GPU acceleration via CUDA.

---

## 📊 Results

- **Reconstruction MSE per pixel**: `0.597`
- **Reconstruction accuracy (pixel difference < 0.1)**: `15.32%`

---

## 📁 Dataset

- Fashion-MNIST dataset contains 28x28 grayscale images of fashion items like shirts, shoes, and bags.
- Dataset loaded via `torchvision.datasets.FashionMNIST`.

---

## 🛠️ Technologies Used

- Python
- PyTorch
- CUDA (optional GPU support)
- Fashion-MNIST dataset (via `torchvision`)
- Matplotlib (for visualization)

---

## 🚀 Getting Started

### 1. Clone this repository

```bash
git clone https://github.com/yourusername/vae-fashionmnist.git
cd vae-fashionmnist
````

### 2. Install dependencies

```bash
pip install torch torchvision matplotlib
```

### 3. Run the training script

```bash
python train_vae.py
```

### 4. Training Configuration

* **Epochs**: 10
* **Batch Size**: 64
* **Latent Dimensions**: 20
* **Optimizer**: Adam

---

## 🖼️ Visualization

The model reconstructs Fashion-MNIST images after encoding them into a 20-dimensional latent space.

| Original Image                     | Reconstructed Image                          |
| ---------------------------------- | -------------------------------------------- |
| ![original](examples/original.png) | ![reconstructed](examples/reconstructed.png) |

---

## 📈 Loss Curves

The training tracks both reconstruction loss (MSE) and KL divergence:

```
Epoch 10: Loss (Reconstruction): 0.5974, KL Loss: 3.4567, Accuracy: 15.32%
```

---

## 🧑‍💻 Code Implementation

* **VAE Model Definition**: Defined in `vae_model.py` using PyTorch's `nn.Module`.
* **Training Loop**: Includes the reparameterization trick and both reconstruction and KL divergence loss.
* **Evaluation**: Reconstruction quality is evaluated using pixel-wise accuracy and MSE loss.
* **Visualizations**: Reconstruction results are plotted alongside original images using `matplotlib`.

---

## 📎 License

This project is licensed under the MIT License.

---

## 📝 Acknowledgements

* Fashion-MNIST dataset is provided by Zalando Research.
* Original VAE paper by Kingma & Welling (2014) for introducing Variational Autoencoders.


```
