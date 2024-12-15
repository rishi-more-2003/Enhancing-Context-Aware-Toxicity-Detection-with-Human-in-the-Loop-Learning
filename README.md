# Context-Aware Toxicity Detection in Online Communities using BiLSTM, BC, and DPO

## Table of Contents
- [Introduction](#introduction)
- [Background](#background)
- [Methodology](#methodology)
  - [Bidirectional LSTM Model](#bidirectional-lstm-model)
  - [Behavior Cloning (BC)](#behavior-cloning-bc)
  - [Direct Preference Optimization (DPO)](#direct-preference-optimization-dpo)
- [Experiments](#experiments)
  - [Datasets](#datasets)
  - [Training Procedure](#training-procedure)
- [Results](#results)
- [Conclusions](#conclusions)
- [Acknowledgments](#acknowledgments)

## Introduction
The rapid growth of online communities has exposed significant limitations in conventional toxicity detection systems. While traditional models are adept at identifying overtly toxic content, they often struggle to detect nuanced, context-dependent harmful language. This project aims to improve toxicity detection by integrating human feedback into machine learning models using BC and DPO methods.

## Background
Traditional toxicity detection models often fail to capture subtle, context-dependent toxic content. By incorporating human-in-the-loop methods, models can adapt to evolving standards and better capture language nuances within specific contexts. This project compares BC and DPO approaches in fine-tuning a BiLSTM model for enhanced toxicity detection.

## Methodology

### Bidirectional LSTM Model
A BiLSTM network processes input sequences in both forward and backward directions, capturing contextual information effectively. The model uses word embeddings and a fully connected layer with a sigmoid activation function to output toxicity probabilities.

### Behavior Cloning (BC)
BC involves training the model using labeled data derived from community feedback. Reddit comments are labeled as toxic or non-toxic based on their upvote-to-downvote ratios. The model learns to imitate this behavior by minimizing the binary cross-entropy loss.

### Direct Preference Optimization (DPO)
DPO is an approach where the model learns directly from preference comparisons between data points, rather than from explicit labels. Preference pairs are constructed from Reddit comments, and the model is trained to assign higher scores to preferred comments using a pairwise preference loss.

## Experiments

### Datasets
- **Kaggle Toxic Comment Classification Challenge Dataset**: Used to establish a baseline for explicit toxicity detection.
- **Reddit Comments Dataset**: Scraped from Reddit, labeled based on upvote/downvote ratios for BC and used to create preference pairs for DPO.
- **Golden Standard Dataset**: A curated collection of 974 comments from r/politics on the Reveddit platform, reflecting moderator decisions on toxicity.

### Training Procedure
- **Baseline Training**: The BiLSTM model is initially trained on the Kaggle dataset.
- **Behavior Cloning Fine-Tuning**: The model is fine-tuned using BC on the Reddit Comments Dataset.
- **Direct Preference Optimization Fine-Tuning**: The model is fine-tuned using DPO, learning from preference pairs based on community feedback.

## Results
- **Behavior Cloning**: Achieved high training accuracy but performed poorly on the golden dataset, indicating limited ability to detect context-dependent toxicity.
- **Direct Preference Optimization**: Demonstrated exceptional performance, achieving perfect accuracy and F1 score on the golden dataset, effectively capturing nuanced community standards.

## Conclusions
The DPO approach significantly outperforms BC in aligning the model with community standards for toxicity detection. By directly learning from preference comparisons, the DPO model better understands subtle cues distinguishing acceptable content from toxic content.

## Acknowledgments
Dataset:
[Jigsaw Unintended Bias in Toxicity Classification](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data)
