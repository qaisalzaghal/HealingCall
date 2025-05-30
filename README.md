# HealingCall: A Voice-Enabled Medical Assistant with NLP and LLaMA LLM
This project presents a machine learning model that predicts the likely disease based on a person's described symptoms

# 🧠 Disease Prediction Model Based on Symptoms

## 📝 Text + 🎤 Voice Input | 🌐 Local API | 🤖 Small LLM Integration
This project is a smart disease prediction system that uses Natural Language Processing (NLP) and Machine Learning (ML) to identify the most likely disease based on user-described symptoms. It accepts both text and voice input, runs on a local API, and includes an optional small LLM module to enhance understanding of more complex or vague symptom descriptions.

##  📊 Dataset Information
The dataset was downloaded from the [Kaggle](https://www.kaggle.com/datasets/niyarrbarman/symptom2disease) website. it consists of 1200 datapoints and has two columns: "label" and "text".

label : contains the disease labels

text : contains the natural language symptom descriptions.

The dataset comprises 24 different diseases, and each disease has 50 symptom descriptions, resulting in a total of 1200 datapoints.

The following 24 diseases have been covered in the dataset:

Psoriasis, Varicose Veins, Typhoid, Chicken pox, Impetigo, Dengue, Fungal infection, Common Cold, Pneumonia, Dimorphic Hemorrhoids, Arthritis, Acne, Bronchial Asthma, Hypertension, Migraine, Cervical spondylosis, Jaundice, Malaria, urinary tract infection, allergy, gastroesophageal reflux disease, drug reaction, peptic ulcer disease, diabetes

## 🔍 Key Features
📝 Text Input: Users can type symptoms in natural language.

🎤 Voice Input: Supports voice input using speech_recognition.

🧠 ML Classifier: Predicts disease using trained data.

🌐 Local API: Easily testable and integrable via Flask or FastAPI.

🤖 Small LLM Module: Enhances language understanding, handles more complex symptom descriptions, and improves overall accuracy.


# 🔧 Work Steps
This section outlines the key steps followed throughout the project, from data acquisition to model testing:

##  1. 📥 Data Collection
- Collected the dataset from the [Kaggle](https://www.kaggle.com/datasets/niyarrbarman/symptom2disease) website.

##  2. 🧹 Data Preprocessing
* Removing stop words and punctuations
* Calculating the word count before and after cleaning
* Visualized the most frequent words using WordCloud

##  3.⚙️ Data Preparation for Machine Learning
- Transformed text data using Sentence Transformer embeddings

- Trained a Logistic Regression model (89% Accuracy)

 - Trained a Random Forest classifier (94% Accuracy)

 - Applied Grid Search Cross-Validation to optimize the Logistic Regression model (96% Accuracy)

## 4. 🧪 Model Testing via Voice Input through Flask API
- A local API is created using the Flask framework.

- The speech_recognition library is used to convert voice input into text.

- The converted text is then processed and passed to the disease prediction model.

- Input: Voice (spoken symptoms)

- Output: Predicted Disease Name

## 5. 🤖 Building and Fine-Tuning an LLM (LLaMA) for Medical Descriptions
   
   - Loaded the fine-tuned LLaMA 2 model from [aboonaji](https://huggingface.co/aboonaji/llama2finetune-v2) 
   - Performed supervised fine-tuning using the dataset [aboonaji/wiki_medical_terms](https://huggingface.co/datasets/aboonaji/wiki_medical_terms_llam2_format)
   - Developed a lightweight chatbot powered by the fine-tuned model to provide detailed explanations about diseases and respond to symptom-related queries in natural language.

