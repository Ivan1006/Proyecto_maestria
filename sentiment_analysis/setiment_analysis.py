import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def analyze_sentiment(text, tokenizer, model, temperature=1.0):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    logits = outputs.logits.detach().cpu().numpy()[0]
    probabilities = torch.softmax(torch.tensor(logits) / temperature, dim=0).numpy()
    predicted_class = np.argmax(probabilities)
    return predicted_class, probabilities

def plot_sentiment_distribution(sentiments):
    sentiment_labels = ["Negativo", "Neutro", "Positivo"]
    counts = Counter(sentiments)
    x_ticks = range(len(sentiment_labels))
    fig, ax = plt.subplots()
    ax.bar(x_ticks, [counts[i] for i in range(3)], tick_label=sentiment_labels, color=['red', 'grey', 'green'])
    ax.set_title("Distribución de Sentimientos")
    ax.set_xlabel("Sentimiento")
    ax.set_ylabel("Frecuencia")
    plt.show()
