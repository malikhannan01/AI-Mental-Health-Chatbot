# Mental Health Counseling Chatbot

## AI nPowered Emotion Detection and Therapeutic Response System

---

A dual model AI chatbot that detects user emotions using BiLSTM and generates empathetic counseling responses using BlenderBot. The system first classifies the user's emotional state from their text input, then generates a supportive therapeutic response tailored to that emotion.

---

## Overview

This project is an AI-based mental health counseling chatbot built using two deep learning models working together. The first model is a BiLSTM trained on the GoEmotions dataset to detect 28 different emotions from user text input. The second model is a fine-tuned BlenderBot-400M that generates empathetic and therapeutic responses based on the detected emotion. The complete pipeline provides an end-to-end conversational AI system for mental health support.

---

## Dataset

| Dataset | Purpose | Samples |
|---------|---------|---------|
| GoEmotions | Emotion Detection | 58,000 labeled Reddit comments |
| MentalChat16K | Response Generation | 16,000 counselor-patient dialogues |

---

## Model Architecture

| Component | Model | Parameters | Task |
|-----------|-------|------------|------|
| Emotion Detection | BiLSTM | 2.6M | Classifies text into 28 emotions |
| Response Generation | BlenderBot-400M | 400M | Generates therapeutic responses |

---

## Results

| Metric | Score |
|--------|-------|
| Emotion Detection Accuracy | 45.46% across 28 classes |
| Training Dataset | Balanced GoEmotions with class weights |
| Response Style | Empathetic, structured, multi-step advice |

---

## Features

- Detects 28 emotions including sadness, anxiety, anger, joy, grief, nervousness
- Generates empathetic therapeutic responses using fine-tuned BlenderBot
- End-to-end pipeline from emotion detection to response generation
- Interactive chat interface for real-time conversation
- Trained on balanced GoEmotions dataset with class weighting
- BlenderBot fine-tuned on mental health counseling conversations
- Works offline without API dependencies

---

## Setup Instructions

### Step 1: Download Model Files

The complete model folder is available on Google Drive. Due to large file sizes (7-8 GB), download the entire folder.

Download link: https://drive.google.com/drive/folders/1ndOcz9W2VsxOE2BG2eXdlRoaltXSBhJD?usp=sharing

### Step 2: Upload to Your Google Drive

Upload the entire `ai_chat_bot` folder to your own Google Drive. Do not rename the folder or any files inside it.

Folder structure:

ai_chat_bot/
├── BiLSTM/
│ ├── goemotion_tokenizer.pkl
│ ├── goemotion_label_mapping.pkl
│ ├── goemotion_max_length.pkl
│ └── bilstm_goemotion_balanced.h5
└── MentalHealth/
├── config.json
├── model.safetensors
├── tokenizer_config.json
├── vocab.json
├── merges.txt
└── generation_config.json
