import pandas as pd
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer
import numpy as np

def process_intent_data():
    # Load the dataset
    df_intent = pd.read_csv('intent_classifier_dataset.csv')
    
    # Encode the intents
    label_encoder = LabelEncoder()
    df_intent['intent_encoded'] = label_encoder.fit_transform(df_intent['intent'])
    
    # Create intent mapping
    intent_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    print("Intent mapping:", intent_mapping)
    
    # Save the label encoder for later use
    np.save('intent_encoder.npy', label_encoder.classes_)
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    
    # Tokenize the text
    def tokenize_text(text):
        return tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='tf'
        )
    
    # Apply tokenization
    tokenized_data = df_intent['text'].apply(tokenize_text)
    
    # Create input_ids and attention_mask arrays
    input_ids = np.array([item['input_ids'].numpy()[0] for item in tokenized_data])
    attention_mask = np.array([item['attention_mask'].numpy()[0] for item in tokenized_data])
    
    # Save processed data
    np.save('input_ids.npy', input_ids)
    np.save('attention_mask.npy', attention_mask)
    np.save('labels.npy', df_intent['intent_encoded'].values)
    
    # Print dataset statistics
    print("\nDataset Statistics:")
    print(f"Total samples: {len(df_intent)}")
    print(f"Number of unique intents: {len(intent_mapping)}")
    print("\nIntent distribution:")
    print(df_intent['intent'].value_counts())
    
    # Save processed dataset
    df_intent.to_csv('processed_intent_dataset.csv', index=False)
    print("\nProcessed dataset saved to 'processed_intent_dataset.csv'")
    
    # Create a sample of the processed data
    print("\nSample of processed data:")
    print(df_intent[['text', 'intent', 'intent_encoded']].head())

if __name__ == "__main__":
    process_intent_data() 