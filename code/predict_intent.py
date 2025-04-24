import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

def load_model(model_path="./intent_classifier"):
    """Load the trained model and tokenizer"""
    model = TFAutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load labels
    label_classes = np.load(f'{model_path}/label_classes.npy', allow_pickle=True)
    
    return model, tokenizer, label_classes

def predict_intent(text, model, tokenizer, label_classes):
    """Predict the intent of a given text"""
    # Tokenize the input text
    inputs = tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors="tf")
    
    # Get model prediction
    outputs = model(inputs)
    predictions = tf.nn.softmax(outputs.logits, axis=-1)
    
    # Get the predicted class and confidence
    predicted_class_idx = tf.argmax(predictions, axis=1).numpy()[0]
    confidence = float(predictions[0, predicted_class_idx].numpy())
    
    # Get the label
    predicted_intent = label_classes[predicted_class_idx]
    
    # Get top 3 predictions with their confidences
    top_idxs = tf.argsort(predictions[0], direction='DESCENDING').numpy()[:3]
    top_predictions = [
        {"intent": label_classes[idx], "confidence": float(predictions[0, idx].numpy())}
        for idx in top_idxs
    ]
    
    return {
        "predicted_intent": predicted_intent,
        "confidence": confidence,
        "top_predictions": top_predictions
    }

def main():
    # Load model, tokenizer, and labels
    print("Loading model...")
    model, tokenizer, label_classes = load_model()
    
    print(f"Model loaded. Available intents: {label_classes}")
    
    # Interactive prediction loop
    print("\nEnter text to predict intent (type 'exit' to quit):")
    
    while True:
        text = input("\nText: ")
        
        if text.lower() == "exit":
            break
        
        # Make prediction
        result = predict_intent(text, model, tokenizer, label_classes)
        
        # Print results
        print(f"\nPredicted intent: {result['predicted_intent']} (confidence: {result['confidence']:.4f})")
        print("\nTop predictions:")
        for idx, pred in enumerate(result['top_predictions']):
            print(f"{idx+1}. {pred['intent']} - {pred['confidence']:.4f}")

if __name__ == "__main__":
    main() 