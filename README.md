# whisper-mandarin-pitch
[LoRA Fine-Tuning with Pitch Augmented Features]

## Description
This project trains and evaluates an OpenAI Whisper model with additional pitch (f0) features to improve speech recognition accuracy.

## Repository Structure

- **train.py**: Main script for training the model.
- **eval.py**: Main script for evaluating the model.
- **config.yaml**: Configuration file containing parameters for training and evaluation.
- **model_utils.py**: Contains functions and classes related to model setup, modification, and custom layers.
- **data_utils.py**: Contains functions for data preparation, data collators, and dataset loading.
- **requirements.txt**: Lists all required Python packages.
- **README.md**: Provides an overview of the project and instructions on how to use it.

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU with the appropriate CUDA toolkit installed (e.g., CUDA 11.8)
- Git

### Install Dependencies

First, clone the repository:

```bash
git clone https://github.com/suminparkl/whisper-mandarin-pitch.git
cd whisper-mandarin-pitch
```

### References

This project was inspired by and references [Vaibhavs10's fast-whisper-finetuning](https://github.com/Vaibhavs10/fast-whisper-finetuning).
We have incorporated some of the ideas and methodologies from their repository to enhance this implementation.
