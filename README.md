# SJTU B.Eng Degree Thesis Project

This repository contains the code and materials for the thesis project of the Bachelor of Engineering (B.Eng) degree at Shanghai Jiao Tong University (SJTU).

## Project Overview

Large Language Models (LLMs) have been widely adopted in tasks such as natural language processing and question answering, with their reasoning capabilities drawing growing attention. This thesis investigates the relationship between model size and reasoning ability in LLMs. Through systematic experiments across multiple datasets and benchmarks, a positive correlation between model size and reasoning performance is observed, particularly on more challenging problems.

Key contributions of this work include:
- Proposing a Layerwise Padding approach based on the Patchscopes framework.
- Conducting mechanistic interpretability analysis to gain deeper insight into how reasoning abilities emerge across LLMs of different scales.
- Introducing a classification-based method to efficiently trade off between accuracy and computational efficiency, addressing out-of-distribution real world tasks.

These findings contribute theoretical insight and experimental evidence toward understanding the reasoning mechanisms of LLMs and provide practical guidance for future model design and optimization.

---

## Main Functions of Each File

- `thesis_main.py`: Main entry point for running thesis experiments. Evaluates LLMs of different sizes on reasoning tasks, compares their performance, and applies the classifier-based method for efficient inference.
- `classifier.py`: Loads a trained BERT-based classifier to predict the difficulty of math questions (easy, hard, very hard). Used for routing questions to the appropriate model in the classifier-based method.
- `data_loader.py`: Handles loading and preprocessing of datasets for training, evaluation, and labeling. Includes utilities for sampling, shuffling, and preparing data splits.
- `eval.py`: Runs the Patchscope mechanistic interpretability analysis, evaluating how reasoning ability emerges across model layers. Produces accuracy and surprisal curves.
- `evaluate.py`: Provides evaluation utilities for Patchscope outputs, including accuracy metrics and label evaluation.
- `label_generation.py`: Generates difficulty labels for questions by running LLMs and saving results for classifier training.
- `model.py`: Contains model loading utilities, including functions to load LLMs and tokenizers, and to enumerate model layers.
- `parser.py`: Provides functions to parse questions and answers from various datasets, and to standardize answer formats for evaluation.
- `patchscope.py`: Implements the Patchscope framework, including hidden state extraction, patching, and evaluation across layers.
- `train_classifier.py`: Script for training the BERT-based difficulty classifier using labeled data.
- `utils.py`: Utility functions for seeding, file I/O, math comparison, prompt construction, and plotting.

---

## How to Use This Repository

1. **Environment Setup**
   - Install dependencies using the provided `environment.yml` file:
     ```
     conda env create -f environment.yml
     conda activate <env_name>
     ```

2. **Data Preparation**
   - Place or link your datasets in the `data/` directory, following the expected structure (see `data_loader.py` for details).

3. **Running Main Experiments**
   - To reproduce the main thesis results (model evaluation, classifier-based routing, etc.), run:
     ```
     python thesis_main.py --base_model <base_model_name> --advanced_model <advanced_model_name> --data_names <comma_separated_datasets>
     ```
   - See `thesis_main.py` for all configurable arguments.

4. **Patchscope Analysis**
   - To run mechanistic interpretability experiments and generate layerwise accuracy/surprisal curves:
     ```
     python eval.py --data_names <dataset> --source_model_name <model_name>
     ```

5. **Difficulty Label Generation**
   - To generate difficulty labels for questions (for classifier training):
     ```
     python label_generation.py --data_names <dataset>
     ```

6. **Train the Classifier**
   - To train the BERT-based classifier on labeled data:
     ```
     python train_classifier.py
     ```

7. **Classify Question Difficulty**
   - Use `classifier.py` to predict the difficulty of new questions.
