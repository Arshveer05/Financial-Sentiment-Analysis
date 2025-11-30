# Financial-Sentiment-Analysis
A multi model analysis for news sentiment classification on the Polygon.io financial news dataset.
The system processes raw financial articles, performs transformer-based encoding, trains multiple deep learning architectures (including FinBERT, RoBERTa variants, multi-transformer fusion, keyword-focused masking, and a custom continued-pretrained RoBERTa), and evaluates them systematically.

# Project Structure
 ## Data Pipeline and Preprocessing
 
   ### Inputs
     1. title
     2. description
   Concatenated into one field:
   Text = Title + " " + Description
   ### Label Processing
   LabelEncoder converts raw labels to integer IDs

   The project implements 5 different models as mentioned below:
 ## FinBERT
   The model used was created by ProsusAI. 
   FinBERT is based on BERT-base-uncased, but is pretrained on general-domain English further finetuned on financial text, including:
     1. 10-K filings
     2. Analyst reports
     3. Earnings call transcripts
     4. Financial news
  This makes it ideal for financial sentiment tasks.
  Model Details: - 12 Transformer Encoder Layers
                 - 12 Attention Heads per Layer
                 - Hidden Size = 768
                 - Total Parameters around 110 Million
  We use FinBERT as a frozen encoder, which means there are no weight updates to the FinBERT layers and we are not using FinBERTs classification head, only its embeddings.
    ###

   
 ## RoBERTa
 ## RoBERTa With Keyword Masking (In House Innovation - Provides Novelty)
 ## RoBERTa further Pre-trained (In House pretraining)
 ## Multi Transformer(BERT + RoBERTa + DistilBERT) Embeddings + LSTM 
