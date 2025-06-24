# Image Captioning with Visual Attention: Show, Attend and Tell

This project implements an image captioning model using **Show, Attend and Tell** architecture. It combines convolutional neural networks (CNNs) for feature extraction and attention-based recurrent neural networks (RNNs) for generating natural language descriptions of images. The attention mechanism allows the model to focus on relevant regions of the image while generating each word.

---

## Features

#### Encoder: Convolutional Neural Network (CNN)

* Used a pretrained ResNet-101 model to extract image feature maps.
* Converted high-dimensional image data into lower-dimensional spatial features suitable for attention.

#### Attention Mechanism:

* Implemented Bahdanau (soft) attention to dynamically focus on different parts of the image during caption generation.
* Improved model interpretability by visualizing attention weights across image regions.

#### Decoder: LSTM with Attention

* Designed an LSTM-based decoder that generates captions word-by-word.
* Integrated attention context vectors at each time step for accurate captioning.

#### Evaluation Metrics:

* Evaluated model performance using BLEU-4 score for caption quality.
* Visualized attention maps to analyze model focus during caption generation.

---

## Getting Started

#### Prerequisites

* Python 3.x
* PyTorch
* Torchvision
* NumPy, Pandas
* Matplotlib, Seaborn
* nltk
* tqdm
* PIL, OpenCV

#### Install packages:

```bash
pip install -r requirements.txt
```

#### Dataset

* **MS COCO 2014 Dataset**
* The dataset can be downloaded following the instructions in the notebook.

---

## How to Run

1. Clone this repository:

```bash
git clone https://github.com/yourusername/image-captioning-with-attention.git
```

2. Navigate to the project directory:

```bash
cd image-captioning-with-attention
```

3. Launch the Jupyter Notebook:

```bash
jupyter notebook Show_Attend_and_Tell.ipynb
```

4. Follow the notebook cells to:

* Preprocess and tokenize the captions.
* Train the model.
* Generate captions for test images.
* Visualize attention maps during caption generation.

---

## Results

* Achieved BLEU-4 scores indicating strong alignment between generated and reference captions.
* Attention visualizations confirmed the model’s ability to focus on relevant image regions during caption generation.

---

## Analysis

* The attention mechanism successfully enhances the quality of generated captions by dynamically selecting image features relevant to each word.
* The model demonstrates how integrating attention improves performance over standard encoder-decoder models in image captioning tasks.
* Attention visualizations provide valuable interpretability into model decision-making.

---

## Key Learnings

* Attention mechanisms greatly improve image captioning by focusing on meaningful visual regions.
* Combining CNN encoders with LSTM decoders allows effective sequence generation conditioned on visual context.
* Evaluation using BLEU scores and visual analysis of attention maps helps quantify both accuracy and interpretability.

---

## Future Work

* Experiment with Transformer-based image captioning models.
* Explore reinforcement learning-based caption generation.
* Extend the model to work with larger and more diverse datasets.
* Incorporate multimodal inputs like audio or metadata for richer captions.

