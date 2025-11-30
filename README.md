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
  Model Details: 
  
                 - 12 Transformer Encoder Layers

                 - 12 Attention Heads per Layer
    
                 - Hidden Size = 768
    
                 - Total Parameters around 110 Million
  We use FinBERT as a frozen encoder, which means there are no weight updates to the FinBERT layers and we are not using FinBERTs classification head, only its embeddings.
### Tokenization
  The Auto Tokenizer from the pretrained ProsusAI FinBERT model is used and this gives us:

    1. Input IDs
    2. Attention Mask
  Max Token Length is 128 Tokens
### Embedding Extraction and Trainable Classifier Heads
    last_hidden_state:  [batch_size, seq_len, 768]
   Apply Mean Pooling to create a single embedding per input with the embedding size of 768.

   We use a 2 layer feed forward neural network
   Input: 768-d FinBERT vector
----------------------------------------------------------
Dropout(p=0.3)

Linear(768 → 256)

ReLU

Dropout(p=0.3)

Linear(256 → 3 classes)

Output: logits (raw scores)

----------------------------------------------------------
### End to End Forward Pipeline
Raw Text
   ↓
Tokenizer (FinBERT)
   ↓
input_ids, attention_mask
   ↓
Frozen FinBERT Encoder
   ↓
last_hidden_state (768-d embeddings per token)
   ↓
Mean Pooling
   ↓
Document Embedding (768-d)
   ↓
Feed-Forward Classifier
   ↓
Logits
   ↓
CrossEntropyLoss
   ↓
Backpropagation (only classifier head gets gradients)
### Training Architecture
  We require only about 200k trainable parameters - the FinBERT classifier parameters.
  Adam Optimizer is used with the loss function being Cross Entropy Loss
 
### Architecture Overview: 
<img width="768" height="542" alt="image" src="https://github.com/user-attachments/assets/8e3970a2-a410-493d-809e-cc37aa90c3d3" />
   
 ## RoBERTa
 With RoBERTa we use a 2 stage hypbrid model:
 
        Stage 1 uses a Frozen RoBERTa Encoder which acts as a feature extractor
        Stage 2 uses a trainable Lightweight Feed Forward Classifier
        
 ### Tokenization
 roberta-base autotokenizer tokenizes the text and pads and truncates it to 128 tokens, this returns

      1. Input IDs
      2. Attention Mask
 All RoBERTa weights are frozen i.e. it is in inference mode only, this returns the output: last_hidden_state → shape [batch, seq_len, 768]
 For Sentence Embeddings we use mean pooling that produces a 768 dimensional embedding per document
 ### Trainable Classifier
 ------------------------------------------------------

Input: 768-d embedding

Dropout(p=0.3)

Linear(768 → 256)

ReLU

Dropout(p=0.3)

Linear(256 → 3_CLASSES)

Output: unnormalized logits

------------------------------------------------------
### Training
Since only the classifier head is trainable over here as well , we have about 200k parameters
Training Loop:

    1. Mini-batch training

    2. No gradient flow into RoBERTa

    3. Validation at each epoch

### Architecture Overview:
<img width="870" height="538" alt="image" src="https://github.com/user-attachments/assets/0ad30352-8a41-43bc-bc45-fb62596a60bd" />


 ## RoBERTa With Keyword Masking (In House Innovation - Provides Novelty)
 This model introduces a custom keyword-guided attention mask, which forces RoBERTa to focus on financially significant terms.
 The architecture consists of:
 
     1. Phase 1 — Smart Attribute Discovery (novel preprocessing)

     2. Phase 2 — Keyword-Focused RoBERTa Encoder (novel masking)

     3. Phase 3 — Classification Head

     4. Phase 4 — Best-epoch weight selection
 This is a hybrid system combining statistical feature selection, keyword-based masking, and transformer pooling.

 ### Smart Attribute Discovery
 Inputs
 
      1. Title
      2. Description
      3. Keywords list (50 manually selected keywords)

  We convert the text to a bag of words using the CountVectorizer then apply a chi square test to extract the top 200 most label predictive words and then merge it with the manually created keywords list.
  Thus we get a hybrid keyword dictionary.

  ### Keyword focused dataset
  Dataset returns:

     1. input_ids

     2. attention_mask

     3. keyword_mask

     4. labels

 Keyword Mask Logic:
      
 For every token in the input sequence:

     1. Convert token ID → token text

     2. Strip RoBERTa’s “Ġ” marker - to be able to match keywords

     3. Lowercase

     4. Check if token ∈ smart_attribute_list
     
  This produces a tensor of shape: keyword_mask: [seq_len]

### Keyword Focused RoBERTa Encoder
Base Model: roberta-base
Standard outputs: last_hidden_state: [batch(16), seq_len(max 160), 768]


Then apply keyword masking to embeddings and multiply the embeddings:
masked_embeddings = last_hidden_state * expanded_mask


We use a different kind of pooling here compared to previous models:

     pooled_output = sum(masked_embeddings) / sum(keyword_mask)

So final embedding is:

     1. Average only across financially relevant tokens

     2. Completely discards irrelevant context (stopwords, filler, etc.)

pooled_output: [batch(16) , 768]

### Classification Head
Dropout(0.3)
Linear(768 → 3)

### Training Architecture
  AdamW optimizer (weight decay 0.02)
  Warmup + linear decay
  Gradient clipping applied
### Architecture Overview:
<img width="1600" height="971" alt="image" src="https://github.com/user-attachments/assets/83cdd21b-d2a0-4cbb-a771-b260db05558a" />

  

 ## RoBERTa further Pre-trained (In House pretraining)
 ## Multi Transformer(BERT + RoBERTa + DistilBERT) Embeddings + LSTM 
