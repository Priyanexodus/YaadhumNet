# YaadhumNet

**YaadhumNet** is an open-source research implementation of the **MOON (Model-Contrastive Federated Learning)** algorithm built using PyTorch and Flower. The project explores federated training using CNN and Vision Transformer encoders while focusing on practical deployment of federated learning systems across distributed client machines and cloud-based servers.

The repository aims to reproduce and extend the ideas presented in the *MOON: Model-Contrastive Federated Learning* paper and provide a structured environment for experimentation, benchmarking, and deployment of federated learning systems.

---

# Overview

Federated Learning enables multiple clients to collaboratively train machine learning models without sharing their raw data. Instead of centralizing data, each client trains locally and shares model updates with a central aggregation server.

However, real-world federated learning systems often suffer from **data heterogeneity** across clients. The MOON algorithm introduces **model-level contrastive learning** to mitigate this problem by encouraging local models to remain close to the global representation during training.

YaadhumNet implements this approach and explores multiple encoder architectures to evaluate their effectiveness in federated environments.

---

# Implemented Models

The project currently supports two encoder variants.

## CNN Encoder

A convolutional neural network serves as the baseline encoder.
The architecture follows the design described in the MOON paper.

Components include:

* CNN feature encoder
* Projection head for contrastive representation learning
* Classification head for prediction

The projection head maps encoder outputs into a latent space where **contrastive similarity between local and global models** is computed.

---

## Vision Transformer Encoder

In addition to the baseline CNN model, the project also explores a **Vision Transformer (ViT)** encoder.

Vision Transformers are known for strong representation learning capabilities and have demonstrated state-of-the-art performance across many computer vision tasks.

This allows the project to evaluate **transformer-based federated learning with model-contrastive objectives**.

---

# System Architecture

YaadhumNet follows a typical federated learning setup consisting of:

**Central Server**

* Coordinates training rounds
* Aggregates model updates
* Maintains the global model

**Client Nodes**

* Hold private datasets
* Train local models
* Send updated weights to the server

Federated orchestration is handled using the Flower framework, while model training is implemented using PyTorch.

---

# Deployment Vision

One of the long-term goals of this project is to move beyond research experimentation and support **real-world deployment of federated learning systems**.

Planned deployment architecture includes:

* Cloud-hosted federated server
* Multiple distributed client machines
* Secure communication between nodes
* Scalable infrastructure orchestration

Future work includes deployment on cloud platforms such as AWS and integration with containerized environments for reproducible training.

---

# Roadmap

Planned improvements for the project include:

* CNN-based MOON baseline implementation
* Vision Transformer based encoder
* Experiment configuration management
* Automated testing pipeline
* CI/CD integration
* Cloud-based deployment setup
* Multi-client distributed training experiments

---

# References

**MOON: Model-Contrastive Federated Learning**

Li, Qinbin et al.
Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021.

Paper:
https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Model-Contrastive_Federated_Learning_CVPR_2021_paper.pdf

---

# License

This project is released under the MIT License.

---

# Acknowledgements

This project builds upon the tools and research contributions of the federated learning community.

Special thanks to the developers of:

* PyTorch
* Flower
