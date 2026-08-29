# Image Caption Generator

An end-to-end **Image Captioning** system that generates natural-language descriptions for images.

The project combines a pretrained **ResNet-50** convolutional neural network for image feature extraction with an **LSTM-based language model** for caption generation. The trained model is evaluated on the Flickr8k test set and is also packaged into a Docker container with a simple Streamlit web interface.

---

## Project Overview

The goal of this project is to build an image captioning pipeline that takes an image as input and automatically generates a textual description of its contents.

The complete pipeline is:

```text
                Input Image
                     │
                     ▼
             ┌───────────────┐
             │   ResNet-50   │
             │ Feature       │
             │ Extraction    │
             └───────┬───────┘
                     │
              2048-D Features
                     │
                     ▼
             ┌───────────────┐
             │ Image         │
             │ Projection    │
             └───────┬───────┘
                     │
                     ▼
             ┌───────────────┐
             │     LSTM      │
             │ Caption       │
             │ Generation    │
             └───────┬───────┘
                     │
                     ▼
             Generated Caption
```

For example:

```text
Input:
1000268201_693b08cb0e.jpg

Output:
"a girl in a pink dress is sitting on a wooden bench"
```

---

## Dataset

The project uses the **Flickr8k dataset**.

Flickr8k contains **8,000 images**, with multiple human-written captions associated with each image.

The dataset is organized into:

```text
data/
└── raw/
    └── Flickr8k/
        ├── Images/
        └── captions.txt
```

The images are used as visual input, while the associated captions are used as the target text during training and evaluation.

The dataset is divided into training, validation, and test sets.

The final evaluation was performed on **810 test images**.

---

## Model Architecture

The system consists of two main components:

### 1. ResNet-50 Feature Extractor

A pretrained ResNet-50 network is used to convert each image into a numerical feature representation.

The final classification layer of ResNet-50 is removed:

```python
model = nn.Sequential(
    *list(model.children())[:-1]
)
```

The resulting representation has:

```text
2048 dimensions
```

The feature extractor uses the preprocessing transformations associated with the pretrained ResNet-50 weights.

The extracted feature is saved as a PyTorch tensor so that image features do not have to be recomputed during every training epoch.

---

### 2. LSTM Caption Generator

The caption generator is an LSTM-based neural network.

The architecture contains:

```text
Image Features
      │
      ▼
Image Projection
      │
      ▼
LSTM
      │
      ▼
Word Embedding
      │
      ▼
LSTM
      │
      ▼
Output Layer
      │
      ▼
Next Word
```

The image feature is projected into the model's embedding space and used to initialize the LSTM's hidden state.

During training, caption tokens are provided sequentially to the LSTM.

During inference, the model starts with the `<start>` token and predicts one word at a time until the `<end>` token is generated or the maximum caption length is reached.

---

## Preprocessing

### Image preprocessing

Images are loaded using PIL and converted to RGB.

The preprocessing transformations are provided by the pretrained ResNet-50 weights:

```python
weights = ResNet50_Weights.DEFAULT
self.transform = weights.transforms()
```

The resulting image tensor is passed through ResNet-50.

The output feature has the shape:

```text
[2048]
```

and is saved as a `.pt` file.

---

### Caption preprocessing

Captions are converted into token sequences.

Special tokens are used to indicate the beginning and end of a caption:

```text
<start>
<end>
<pad>
<unk>
```

A vocabulary maps words to integer IDs:

```text
word → token ID
```

and can also convert token IDs back into words during inference.

The vocabulary is stored in:

```text
artifacts/vocabulary.pkl
```

---

## Feature Extraction

The project precomputes image features using ResNet-50.

The extracted features are stored in:

```text
data/processed/features/
```

Example:

```text
data/processed/features/518789868_8895ef8792.pt
data/processed/features/519059913_4906fe4050.pt
...
```

This allows the captioning model to train using precomputed image representations rather than running ResNet-50 for every training example.

---

## Training

The captioning model is trained using the processed image features and corresponding caption sequences.

The training pipeline consists of:

```text
Flickr8k Images
       │
       ▼
Image Preprocessing
       │
       ▼
ResNet-50
       │
       ▼
2048-D Image Features
       │
       ▼
Caption Dataset
       │
       ▼
Vocabulary / Tokenization
       │
       ▼
LSTM Captioning Model
       │
       ▼
Training
       │
       ▼
Best Model Checkpoint
```

The best trained model is saved as:

```text
artifacts/checkpoints/best_model.pt
```

The checkpoint contains the trained model parameters and the configuration required to reconstruct the model.

---

## Evaluation

The trained model was evaluated on the Flickr8k test set.

A total of:

```text
810 images
```

were evaluated.

The following metrics were used:

- BLEU-1
- BLEU-2
- BLEU-3
- BLEU-4
- ROUGE-L
- METEOR

### Results

| Metric | Score |
|---|---:|
| BLEU-1 | 0.5604 |
| BLEU-2 | 0.3761 |
| BLEU-3 | 0.2387 |
| BLEU-4 | 0.1592 |
| ROUGE-L | 0.4499 |
| METEOR | 0.3667 |

The complete test results are stored in:

```text
artifacts/test_results.csv
```

---

## Project Structure

```text
image-caption-generator/
│
├── artifacts/
│   ├── checkpoints/
│   │   └── best_model.pt
│   ├── vocabulary.pkl
│   └── test_results.csv
│
├── data/
│   ├── raw/
│   │   └── Flickr8k/
│   │       └── Images/
│   │
│   └── processed/
│       └── features/
│
├── src/
│   ├── config/
│   │   ├── __init__.py
│   │   └── config.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── build_vocabulary.py
│   │   ├── dataset.py
│   │   ├── preprocessing.py
│   │   └── vocabulary.py
│   │
│   ├── evaluation/
│   │   ├── evaluate_test_set.py
│   │   └── metrics.py
│   │
│   ├── models/
│   │   └── caption_model.py
│   │
│   ├── vision/
│   │   ├── __init__.py
│   │   ├── extract_features.py
│   │   └── feature_extractor.py
│   │
│   ├── inference/
│   │   └── predict.py
│   │
│   ├── inference.py
│   ├── train.py
│   └── app.py
│
├── tests/
│   ├── test_dataset.py
│   ├── test_evaluation.py
│   ├── test_inference.py
│   ├── test_model.py
│   ├── test_preprocessing.py
│   ├── test_training.py
│   └── test_vocabulary.py
│
├── Dockerfile
├── pytest.ini
├── requirements.txt
└── README.md
```

---

## Installation

### 1. Clone the repository

```bash
git clone -b task6_2 git@github.com:rahmaokda/task6.git
cd task6
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
```

Activate it:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running Inference

The model can generate a caption for an individual image using:

```bash
python3 -m src.inference.predict \
    --image data/raw/Flickr8k/Images/1000268201_693b08cb0e.jpg
```

Example output:

```text
Image: data/raw/Flickr8k/Images/1000268201_693b08cb0e.jpg
Caption: a girl in a pink dress is sitting on a wooden bench
```

A maximum caption length can also be specified:

```bash
python3 -m src.inference.predict \
    --image path/to/image.jpg \
    --max-length 30
```

---

# Web Application

The project includes a simple **Streamlit interface** that allows a user to upload an image and receive a generated caption.

Start the application with:

```bash
streamlit run src/app.py
```

Streamlit will provide a local URL, normally:

```text
http://localhost:8501
```

Open this URL in a web browser.

### Application workflow

```text
        User
         │
         ▼
   Upload an image
         │
         ▼
    Streamlit App
         │
         ▼
    ResNet-50
         │
         ▼
   Image Features
         │
         ▼
   LSTM Caption Model
         │
         ▼
  Generated Caption
```

---

## Example Application

### Input

The user uploads an image through the Streamlit interface.

Example:

```text
1000268201_693b08cb0e.jpg
```

### Output

```text
a girl in a pink dress is sitting on a wooden bench
```

A short screen recording demonstrating the application can be added here:

```text
[Insert application demo video/GIF here]
```

> **Demo:** Upload an image → the model processes it → the generated caption is displayed.

---

# Docker

The project is Dockerized so that the application and its dependencies can be run in a reproducible environment.

## Build the Docker image

From the project root:

```bash
docker build -t image-caption-generator .
```

## Test the image

The container can be started with:

```bash
docker run --rm image-caption-generator
```

The Docker image contains:

```text
Python environment
        +
Project source code
        +
Trained model
        +
Vocabulary
        +
Required dependencies
```

### Running inference inside Docker

An image directory can be mounted into the container:

```bash
docker run --rm \
    -v "$(pwd)/data/raw/Flickr8k/Images:/app/images" \
    image-caption-generator \
    --image /app/images/1000268201_693b08cb0e.jpg
```

The container generates a caption using the trained model.

---

# Trained Model

The trained model and required vocabulary are hosted separately on Hugging Face.

### Hugging Face Model Repository

**[https://huggingface.co/rahmamosaad/image-caption-generator](https://huggingface.co/rahmamosaad/image-caption-generator)**

The repository contains the model artifacts required for inference, including:

```text
best_model.pt
vocabulary.pkl
```

The Hugging Face repository provides a public and shareable location for the trained model files rather than requiring users to obtain them from the source code repository.

---

# Testing

The project includes unit tests covering important parts of the pipeline.

Run the test suite with:

```bash
pytest
```

The tests cover areas including:

- Dataset handling
- Preprocessing
- Vocabulary
- Model
- Training
- Inference
- Evaluation

---

# Technologies Used

- **Python**
- **PyTorch**
- **Torchvision**
- **ResNet-50**
- **LSTM**
- **PIL**
- **NLTK**
- **NumPy**
- **Pandas**
- **scikit-learn**
- **Streamlit**
- **Docker**
- **Hugging Face**

---

# End-to-End Pipeline

The complete system can be summarized as:

```text
                    Flickr8k
                       │
             ┌─────────┴─────────┐
             │                   │
          Images              Captions
             │                   │
             ▼                   ▼
        ResNet-50          Preprocessing
             │                   │
             ▼                   ▼
       Image Features        Vocabulary
             │                   │
             └─────────┬─────────┘
                       │
                       ▼
                LSTM Caption Model
                       │
                       ▼
                  Training
                       │
                       ▼
                Best Checkpoint
                       │
             ┌─────────┴─────────┐
             │                   │
             ▼                   ▼
        Evaluation           Inference
             │                   │
             ▼                   ▼
       Test Metrics        Generated Caption
                                 ▲
                                 │
                          User Uploaded Image
                                 │
                                 ▼
                            Streamlit UI
```

---

# Example

Given an image such as:

```text
data/raw/Flickr8k/Images/1000268201_693b08cb0e.jpg
```

the system produces:

```text
a girl in a pink dress is sitting on a wooden bench
```

This demonstrates the complete process from image input to natural-language caption generation.

---

# Model Artifacts

The main trained artifacts are:

```text
artifacts/
├── checkpoints/
│   └── best_model.pt
├── vocabulary.pkl
└── test_results.csv
```

The trained model and vocabulary are also available from the public Hugging Face repository:

**https://huggingface.co/rahmamosaad/image-caption-generator**

---

# Conclusion

This project implements a complete image captioning pipeline, starting from Flickr8k image and caption data and ending with a deployable image captioning application.

The system combines:

```text
Computer Vision
      +
Natural Language Processing
      +
Deep Learning
      +
Model Evaluation
      +
Web Interface
      +
Dockerization
```

The resulting application allows users to upload an image and receive an automatically generated textual description.