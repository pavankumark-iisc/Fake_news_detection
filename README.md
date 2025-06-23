# 🤝 Contributors
Rahul Kumar, Rahul Rai, Vishakha Kumari, Manoj Kumar, Mukunda, Pavan Kumar, Rajat Chaudhary

# Description
This project implements an intelligent chatbot capable of verifying whether a news claim — provided as text or audio — is REAL, FAKE, or UNSURE. It uses a hybrid architecture that combines:
    -> Retrieval-Augmented Generation (RAG) for grounding the model's reasoning in factual web-based evidence.
    -> SarvamAI APIs for real-time speech-to-text (STT) conversion and translation.
    -> ChatGroq-powered LLMs (Qwen2.5, Mistral, LLaMA) for fact-checking, explanation, and natural language reasoning.

# Key Features:
🔤 Input Flexibility: Accepts user input in multiple languages and formats (typed text, spoken audio).
🌐 Multilingual Processing: Automatically translates non-English claims to English using SarvamAI and processes them through the LLM pipeline.
🧠 Dual-Language Output: Explanations are returned in both English and the original input language for clarity and accessibility.
🔎 Grounded Fact Checking: Verifies claims using real-time search results(trusted sources (e.g., news18.com, ptinews.com)) and generates evidence-supported explanations via a LangChain RAG workflow.
🤖 Powered by Open Source LLMs: Supports Qwen2.5, Mistral, and Phi-3-mini through the ChatGroq LLM API for fast and scalable responses.

# 🔍 How It Works
1) Accepts a news claim as text or audio(in any Indian language).
2) Translates to English using SarvamAI (if needed).
3) Performs intelligent web search using Serper.dev.
4) Applies multi-query RAG to gather and summarize evidence.
5) Uses an LLM to classify the claim as REAL / FAKE / UNSURE with explanation.
6) Optionally translates the verdict back to the original language.

# 📦 Setup
Step1: Install dependencies: pip install -r requirements.txt
Step2: Create a .env file with:
    SARVAM_API_KEY, GROK_API_KEY, SERP_DEV_API_KEY, model_multi_query, model_summarizer, model_judge.
Step 3: execute app.py file

# 🚀 Hosted Demo
This project is also hosted on Hugging Face Spaces. You can try it live by clicking the link below:
🔗 Try it here: [Fake News Detection LLM on Hugging Face](https://huggingface.co/spaces/rahul8459875/Fake_News_Detection_LLM)
No installation needed — just paste or speak a claim to get started!

NOTE: This file(code/fake_news_detection_llm.py) contains the full pipeline for news verification using RAG, LangChain, SarvamAI, and Groq.

=======================================================================================================================================================

# 📊 Evaluation Metrics

The following tables summarize performance across languages and input types using different LLMs and strategies.

## 🗣️ LLM Evaluation — English Claims (Strategy 3)

| Model                                           | Total Coverage | F1 Score (Real) | F1 Score (Fake) |
|------------------------------------------------|----------------|------------------|------------------|
| llama3-8b-8192                                 | 64%            | 0.88             | 0.71             |
| qwen/qwen3-32b                                  | 72%            | 0.90             | 0.74             |
| mistral-saba-24b                                | 67%            | 0.90             | 0.68             |
| deepseek-r1-distill-llama-70b                   | 69%            | 0.90             | 0.69             |
| meta-llama/llama-4-scout-17b-16e-instruct       | 70%            | 0.91             | 0.71             |
| meta-llama/llama-4-maverick-17b-128e-instruct   | 52%            | 0.95             | 0.65             |
| qwen-qwq-32b                                     | 67%            | 0.91             | 0.73             |

## 🌐 LLM Evaluation — Regional Languages (Hindi/Kannada)

| Model                                           | Coverage | F1 Score (Real) | F1 Score (Fake) |
|------------------------------------------------|----------|------------------|------------------|
| llama3-8b-8192                                 | 0.68     | 0.92             | 0.69             |
| qwen/qwen3-32b                                  | 0.75     | 0.91             | 0.76             |
| mistral-saba-24b                                | 0.68     | 0.90             | 0.60             |
| deepseek-r1-distill-llama-70b                   | 0.70     | 0.88             | 0.58             |
| meta-llama/llama-4-scout-17b-16e-instruct       | 0.70     | 0.89             | 0.55             |
| meta-llama/llama-4-maverick-17b-128e-instruct   | 0.66     | 0.89             | 0.55             |
| qwen-qwq-32b                                     | 0.66     | 0.89             | 0.57             |

## 🔊 LLM Evaluation — Audio Inputs (Multilingual)

| Model            | Coverage | F1 Score (Real) | F1 Score (Fake) |
|------------------|----------|------------------|------------------|
| llama3-8b-8192   | 0.22     | 0.57             | 0.28             |
| mistral-saba-24b | 0.22     | 0.95             | 0.52             |
| qwen-qwq-32b     | 0.51     | 0.66             | 0.64             |

## 🧠 SarvamAI Speech & Translation Performance

| Metric        | Score   |
|---------------|---------|
| WER           | 0.2887  |
| CER           | 0.0887  |
| BLEU Score    | 0.2027  |
| METEOR Score  | 0.4949  |
| BERTScore     | 0.9149  |
