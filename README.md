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

## Project Notebooks

| Notebook | Purpose |
|----------|---------|
| `mental_health_chatbot.ipynb` | Model training code for BiLSTM and BlenderBot fine-tuning |
| `chatbot_pipeline.ipynb` | Complete pipeline with loading and inference code |

  ---

## How It Works

1. User types a message about their feelings
2. BiLSTM model analyzes the text and detects the dominant emotion
3. Emotion label is prepended to the user input as context
4. BlenderBot generates an empathetic therapeutic response
5. Response is displayed with the detected emotion label
6. Conversation continues until user types quit

## Setup Instructions

### Step 1: Download Model Files

The complete model folder is available on Google Drive. Due to large file sizes (7-8 GB), download the entire folder.

Download link: https://drive.google.com/drive/folders/1ndOcz9W2VsxOE2BG2eXdlRoaltXSBhJD?usp=sharing

### Step 2: Upload to Your Google Drive

Upload the entire `ai_chat_bot` folder to your own Google Drive. Do not rename the folder or any files inside it.


### Step 3: Run the Pipeline

1. Download the `chatbot_pipeline.ipynb` notebook from this repository.  
2. After Uploading the `ai_chat_bot` folder to your Google Drive.  
3. Open the `chatbot_pipeline.ipynb` notebook in Google Colab.  
4. Run First cell to Mount your Google Drive.  
5. Than Run all cells in the notebook.  
6. The chatbot will start working automatically.  
7. If needed, update file paths to match your Google Drive folder location.

   ---

## Limitations

- Emotion detection accuracy is 45% across 28 fine-grained emotions
- Some similar emotions may be confused (sadness vs disappointment)
- BlenderBot responses may occasionally be generic
- Requires both models loaded, approximately 2GB RAM
- Not a substitute for professional mental health care
- Limited to English language only

## Future Improvements

- Improve emotion detection with transformer-based models
- Add conversation history for contextual responses
- Support multilingual conversations
- Deploy as web application with user interface
- Add voice input and output capabilities
- Integrate crisis detection and hotline referrals





