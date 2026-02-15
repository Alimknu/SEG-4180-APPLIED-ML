from flask import Flask, request, jsonify
from transformers import pipeline
import os
import time

app = Flask(__name__)

# Load the pretrained sentiment analysis pipeline
MODEL_NAME = os.environ.get('MODEL_NAME', 'distilbert-base-uncased-finetuned-sst-2-english')
classifier = pipeline('sentiment-analysis', model=MODEL_NAME)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model': MODEL_NAME})

@app.route('/predict', methods=['POST'])
def predict():
    """
    Expected JSON: {"text": "I love chess!"}
    Returns: {"label": "POSITIVE", "score": 0.999, "latency_ms": 123}
    """
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing text field'}), 400
        
        text = data['text']
        
        start = time.time()
        result = classifier(text)[0]  # pipeline returns list
        latency = (time.time() - start) * 1000
        
        response = {
            'text': text,
            'label': result['label'],
            'score': result['score'],
            'latency_ms': round(latency, 2)
        }
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """
    Expected JSON: {"texts": ["text1", "text2"]}
    """
    try:
        data = request.get_json()
        if not data or 'texts' not in data:
            return jsonify({'error': 'Missing texts list'}), 400
        
        texts = data['texts']
        
        start = time.time()
        results = classifier(texts)
        latency = (time.time() - start) * 1000
        
        response = {
            'results': results,
            'count': len(texts),
            'total_latency_ms': round(latency, 2)
        }
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)