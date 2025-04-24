import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, AutoConfig
from transformers.models.distilbert import TFDistilBertForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm.auto import tqdm
import time

# Custom callback with enhanced progress bar
class TqdmProgressCallback(tf.keras.callbacks.Callback):
    def __init__(self, epochs, metrics=None, overall_bar=True):
        super(TqdmProgressCallback, self).__init__()
        self.epochs = epochs
        self.metrics = metrics or []
        self.overall_bar = overall_bar
        self.epoch_start_time = None
        
    def on_train_begin(self, logs=None):
        if self.overall_bar:
            self.overall_progress = tqdm(total=self.epochs, desc='Training Progress', position=0)
        
    def on_train_end(self, logs=None):
        if self.overall_bar:
            self.overall_progress.close()
    
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_progress = tqdm(
            desc=f'Epoch {epoch+1}/{self.epochs}',
            position=1,
            leave=True
        )
        self.current_step = 0
        self.steps = self.params['steps']
        self.epoch_start_time = time.time()
        
    def on_train_batch_end(self, batch, logs=None):
        self.current_step += 1
        self.epoch_progress.update(1)
        self.epoch_progress.total = self.steps
        
        # Update metrics in description
        metrics_str = ' - '.join(f'{m}: {logs.get(m, 0):.4f}' for m in self.metrics if m in logs)
        self.epoch_progress.set_description(
            f'Epoch {self.epoch + 1}/{self.epochs} - {metrics_str}'
        )
        
    def on_epoch_end(self, epoch, logs=None):
        self.epoch_progress.close()
        
        # Collect all available metrics
        metrics_str = ' - '.join(f'{k}: {v:.4f}' for k, v in logs.items())
        epoch_time = time.time() - self.epoch_start_time
        
        # Print a summary for the epoch including time taken
        print(f"Epoch {epoch+1}/{self.epochs} completed in {epoch_time:.2f}s - {metrics_str}")
        
        if self.overall_bar:
            self.overall_progress.update(1)
            # Update overall progress bar with key metrics (loss and accuracy)
            val_acc = logs.get('val_accuracy', 0)
            train_acc = logs.get('accuracy', 0)
            self.overall_progress.set_description(
                f'Training Progress - Loss: {logs.get("loss", 0):.4f} - Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f}'
            )

# Learning Rate Finder class
class LRFinder:
    def __init__(self, model, min_lr=1e-7, max_lr=1e-2, steps=100):
        self.model = model
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.steps = steps
        self.history = {"lr": [], "loss": []}
        
    def find(self, dataset, batch_size=16, beta=0.98):
        print("Starting Learning Rate Finder...")
        # Save original weights
        original_weights = self.model.get_weights()
        
        # Calculate step factor
        step_factor = (self.max_lr / self.min_lr) ** (1 / self.steps)
        lr = self.min_lr
        
        # Prepare dataset
        batched_dataset = dataset.batch(batch_size)
        
        # Initialize optimizer with minimum learning rate
        self.model.optimizer.lr.assign(lr)
        
        # Training loop with progress bar
        avg_loss = 0
        progress_bar = tqdm(total=self.steps, desc=f"Finding optimal learning rate", position=0)
        
        for step, (x, y) in enumerate(batched_dataset):
            if step >= self.steps:
                break
                
            # Update learning rate for this batch
            lr = self.min_lr * (step_factor ** step)
            self.model.optimizer.lr.assign(lr)
            
            # Compute loss
            with tf.GradientTape() as tape:
                logits = self.model(x, training=True)
                loss = self.model.compiled_loss(y, logits)
                
            # Apply gradients
            grads = tape.gradient(loss, self.model.trainable_variables)
            self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            
            # Track loss and lr
            loss_value = float(loss)
            avg_loss = beta * avg_loss + (1 - beta) * loss_value
            smoothed_loss = avg_loss / (1 - beta ** (step + 1))
            
            self.history["lr"].append(lr)
            self.history["loss"].append(smoothed_loss)
            
            # Update progress bar with current learning rate and loss
            progress_bar.set_description(
                f"Finding optimal learning rate - LR: {lr:.8f} - Loss: {smoothed_loss:.4f}"
            )
            progress_bar.update(1)
            
            # Stop if loss explodes
            if step > 0 and smoothed_loss > 4 * self.history["loss"][0]:
                progress_bar.set_description(
                    f"Stopping search - Loss exploded at LR: {lr:.8f}"
                )
                break
        
        progress_bar.close()
                
        # Restore original weights
        self.model.set_weights(original_weights)
        
    def plot(self, skip_start=10, skip_end=5):
        plt.figure(figsize=(12, 6))
        plt.plot(self.history["lr"][skip_start:-skip_end], 
                 self.history["loss"][skip_start:-skip_end])
        plt.xscale('log')
        plt.xlabel('Learning Rate')
        plt.ylabel('Loss')
        plt.title('Learning Rate Finder')
        plt.savefig('lr_finder_plot.png')
        plt.close()
        
        # Find the learning rate with the steepest negative gradient
        losses = self.history["loss"][skip_start:-skip_end]
        lrs = self.history["lr"][skip_start:-skip_end]
        min_grad_idx = np.argmin(np.gradient(losses))
        suggested_lr = lrs[min_grad_idx]
        
        # Find the point with minimum loss
        min_loss_idx = np.argmin(losses)
        min_loss_lr = lrs[min_loss_idx]
        
        print(f"Suggested learning rate (steepest slope): {suggested_lr:.6f}")
        print(f"Learning rate with minimum loss: {min_loss_lr:.6f}")
        return suggested_lr

# Set environment variable to enable progress bar
os.environ["TQDM_NOTEBOOK"] = "true"

# Load preprocessed data
input_ids = np.load('input_ids.npy')
attention_mask = np.load('attention_mask.npy')
labels = np.load('labels.npy')
label_classes = np.load('intent_encoder.npy', allow_pickle=True)

# Create mappings
id2label = {idx: label for idx, label in enumerate(label_classes)}
label2id = {label: idx for idx, label in id2label.items()}


# Split the data into train and validation sets
indices = np.arange(len(labels))
np.random.seed(42)
np.random.shuffle(indices)

train_idx = indices[:int(0.8 * len(indices))]
val_idx = indices[int(0.8 * len(indices)):]

train_input_ids = input_ids[train_idx]
train_attention_mask = attention_mask[train_idx]
train_labels = labels[train_idx]

val_input_ids = input_ids[val_idx]
val_attention_mask = attention_mask[val_idx]
val_labels = labels[val_idx]

# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices(({
    "input_ids": train_input_ids,
    "attention_mask": train_attention_mask
}, train_labels)).shuffle(1000).batch(16)

val_dataset = tf.data.Dataset.from_tensor_slices(({
    "input_ids": val_input_ids,
    "attention_mask": val_attention_mask
}, val_labels)).batch(16)

# Initialize the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Initialize config and model
config = AutoConfig.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(label_classes),
    id2label=id2label,
    label2id=label2id
)

model = TFDistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    config=config
)

# Use Keras native training instead of TFTrainer (which is deprecated)
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metrics = [
    tf.keras.metrics.SparseCategoricalAccuracy('accuracy'),
    tf.keras.metrics.SparseCategoricalCrossentropy(name='cross_entropy', from_logits=True)
]

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# Create unbatched dataset for learning rate finder
unbatched_train_dataset = tf.data.Dataset.from_tensor_slices(({
    "input_ids": train_input_ids,
    "attention_mask": train_attention_mask
}, train_labels)).shuffle(1000)

# Run learning rate finder
print("Running Learning Rate Finder...")
lr_finder = LRFinder(model, min_lr=1e-7, max_lr=1, steps=100)
lr_finder.find(unbatched_train_dataset)
suggested_lr = lr_finder.plot()

# You can use the suggested learning rate or keep the default
print(f"Using learning rate: {suggested_lr}")
model.optimizer.lr.assign(suggested_lr)

# Define number of epochs
epochs = 5

# Train the model with enhanced progress bar
print("Starting training...")
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs,
    callbacks=[
        TqdmProgressCallback(epochs=epochs, metrics=['loss', 'accuracy', 'val_loss', 'val_accuracy']),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True),
        tf.keras.callbacks.TensorBoard(log_dir='./logs'),
        tf.keras.callbacks.ModelCheckpoint(
            filepath='./checkpoints/model_{epoch}',
            save_best_only=True,
            monitor='val_loss'
        )
    ],
    verbose=0  # Turn off default progress bar since we're using our custom one
)

# Plot training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.savefig('training_history.png')
print("Training history plot saved as 'training_history.png'")

# Save the model
model.save_pretrained("./intent_classifier")
tokenizer.save_pretrained("./intent_classifier")

# Evaluate the model
print("Evaluating final model...")
progress_bar = tqdm(total=len(val_dataset), desc="Evaluation")
results = model.evaluate(val_dataset, verbose=0)
for batch in val_dataset:
    # Just update the progress bar
    progress_bar.update(1)
progress_bar.close()

print(f"Validation loss: {results[0]:.4f}")
print(f"Validation accuracy: {results[1]:.4f}")

# Generate predictions for confusion matrix with progress bar
y_pred = []
y_true = []

progress_bar = tqdm(total=len(val_dataset), desc="Generating predictions")
for batch in val_dataset:
    x, y = batch
    logits = model(x, training=False).logits
    predictions = tf.argmax(logits, axis=-1)
    
    y_pred.extend(predictions.numpy())
    y_true.extend(y.numpy())
    progress_bar.update(1)
progress_bar.close()

# Calculate class-wise accuracy
class_correct = {}
class_total = {}
for true, pred in zip(y_true, y_pred):
    label = id2label[true]
    if label not in class_total:
        class_total[label] = 0
        class_correct[label] = 0
    class_total[label] += 1
    if true == pred:
        class_correct[label] += 1

print("\nClass-wise Accuracy:")
for label in class_total:
    accuracy = class_correct[label] / class_total[label]
    print(f"{label}: {accuracy:.4f} ({class_correct[label]}/{class_total[label]})")

# Create confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm, 
    annot=True, 
    fmt='d', 
    cmap='Blues',
    xticklabels=label_classes,
    yticklabels=label_classes
)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()

# Save the confusion matrix plot
plt.savefig('confusion_matrix.png')
print("Confusion matrix saved as 'confusion_matrix.png'")

# Print classification report
report = classification_report(
    y_true, 
    y_pred, 
    target_names=label_classes,
    digits=3
)
print("\nClassification Report:")
print(report)

# Save the classification report
with open('classification_report.txt', 'w') as f:
    f.write(report)

# Save the label mappings
np.save('./intent_classifier/label_classes.npy', label_classes)
with open('./intent_classifier/label_mapping.txt', 'w') as f:
    for label, idx in label2id.items():
        f.write(f"{label}: {idx}\n")
