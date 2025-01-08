from flask import Flask, request, jsonify, render_template
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn.functional as F
import time  # For artificial delay
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

app = Flask(__name__)

# Load BERT model and tokenizer
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained("../bert-incident-classifier")
tokenizer = BertTokenizer.from_pretrained("../bert-incident-classifier")
model.to(device)
model.eval()

# Inverse mapping for label decoding (replace with your own categories)
label_encoder = {
    0: "Access",
    1: "Administrative rights",
    2: "HR Support",
    3: "Hardware",
    4: "Internal Project",
    5: "Miscellaneous",
    6: "Purchase",
    7: "Storage",
}

def classify_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=1)
    top_probs, top_indices = torch.topk(probs, 5)
    top_probs = top_probs.squeeze().tolist()
    top_indices = top_indices.squeeze().tolist()

    recommendations = [
        f"{label_encoder.get(idx, 'Unknown Category')} ({round(prob * 100, 2)}%)"
        for idx, prob in zip(top_indices, top_probs)
    ]
    return recommendations

# Route to render the main page
@app.route('/')
def index():
    return render_template('index.html')

# API endpoint for classification logic
@app.route('/api/classify', methods=['POST'])
def classify():
    data = request.json
    incident_description = data.get('description', '')

    # Get real recommendations from BERT model
    recommendations = classify_text(incident_description)

    return jsonify({'recommendations': recommendations})

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    return jsonify({
        'accuracy': 0.95,
        'f1': 0.93,
        'recall': 0.90,
        'precision': 0.94,
        'confusion_matrix': '[[45, 5], [3, 47]]'
    })

# Confusion matrix plot generation
@app.route('/api/confusion-matrix')
def confusion_matrix():
    # Example confusion matrix data
    matrix = np.array([[45, 5], [3, 47]])
    labels = ['Class 0', 'Class 1']

    # Plotting with seaborn
    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')

    # Convert to BytesIO and send as image response
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    return Response(img.getvalue(), mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)