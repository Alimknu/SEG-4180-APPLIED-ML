from flask import Flask, request, jsonify, send_file
import numpy as np
from PIL import Image
import io, os, time, logging
from pathlib import Path

from config import AppConfig, setup_logging
import tensorflow as tf

setup_logging()
logger = logging.getLogger(__name__)
keras = tf.keras

app = Flask(__name__)

# Load model
def load_model():
    """Load the trained segmentation model"""
    model_path = AppConfig.MODEL_PATH
    
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}")
        logger.warning("Using random predictions for testing")
        return None
    
    try:
        model = keras.models.load_model(model_path)
        logger.info(f"Loaded model from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None


model = load_model()


def preprocess_image(image_bytes, target_size=(256, 256)):
    """Load image, resize to 256x256, normalize to [0,1]"""
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)


def postprocess_mask(mask_pred, threshold=0.5):
    """Binarize prediction at threshold, remove batch dim, return PIL Image"""
    mask_binary = (mask_pred > threshold).astype(np.uint8) * 255
    mask_binary = mask_binary[0, :, :, 0]  # Remove batch dim
    return Image.fromarray(mask_binary, mode='L')


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    model_loaded = model is not None
    return jsonify({
        'status': 'healthy',
        'service': 'house-segmentation',
        'model_loaded': model_loaded,
        'model_path': AppConfig.MODEL_PATH
    }), 200 if model_loaded else 503


@app.route('/predict', methods=['POST'])
def predict():
    """Segment buildings in satellite image"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'Missing image file'}), 400
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 503
        
        image_file = request.files['image']
        
        start = time.time()
        image_bytes = image_file.read()
        image_array = preprocess_image(image_bytes)
        preprocess_time = (time.time() - start) * 1000
        
        start = time.time()
        mask_pred = model.predict(image_array, verbose=0)
        mask_image = postprocess_mask(mask_pred)
        inference_time = (time.time() - start) * 1000
        
        mask_array = np.array(mask_image) / 255.0
        building_percentage = float(np.mean(mask_array) * 100)
        
        response = {
            'filename': image_file.filename,
            'building_percentage': round(building_percentage, 2),
            'preprocess_time_ms': round(preprocess_time, 2),
            'inference_time_ms': round(inference_time, 2),
            'total_latency_ms': round(preprocess_time + inference_time, 2),
            'status': 'success'
        }
        
        logger.info(f"Prediction for {image_file.filename}: "
                   f"{building_percentage:.2f}% buildings")
        
        # Return mask as file with metadata in JSON
        return jsonify({
            **response,
            'message': 'Mask generated successfully. Download from /download_mask'
        }), 200
    
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return jsonify({'error': str(e), 'status': 'error'}), 500


@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """Process multiple images in batch"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 503
        
        data = request.get_json() or {}
        num_images = data.get('count', 0)
        
        if 'images' in request.files:
            # Handle multiple file uploads
            images = request.files.getlist('images')
            num_images = len(images)
        else:
            images = []
        
        start_total = time.time()
        results = []
        
        for idx, image_file in enumerate(images):
            try:
                image_bytes = image_file.read()
                image_array = preprocess_image(image_bytes)
                mask_pred = model.predict(image_array, verbose=0)
                mask_image = postprocess_mask(mask_pred)
                
                mask_array = np.array(mask_image) / 255.0
                building_percentage = float(np.mean(mask_array) * 100)
                
                results.append({
                    'index': idx,
                    'filename': image_file.filename,
                    'building_percentage': round(building_percentage, 2),
                    'status': 'success'
                })
            except Exception as e:
                logger.error(f"Error processing image {idx}: {e}")
                results.append({
                    'index': idx,
                    'filename': image_file.filename if image_file else 'unknown',
                    'error': str(e),
                    'status': 'failed'
                })
        
        total_latency = (time.time() - start_total) * 1000
        
        response = {
            'num_images': len(images),
            'results': results,
            'total_latency_ms': round(total_latency, 2),
            'avg_latency_ms': round(total_latency / max(len(images), 1), 2),
            'status': 'completed'
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/model-info', methods=['GET'])
def model_info():
    """Get model information and configuration"""
    if model is None:
        return jsonify({
            'error': 'Model not loaded',
            'status': 'unavailable'
        }), 503
    
    return jsonify({
        'model_name': AppConfig.MODEL_NAME,
        'model_path': AppConfig.MODEL_PATH,
        'task': 'semantic_segmentation',
        'input_shape': model.input_shape,
        'output_shape': model.output_shape,
        'num_parameters': int(model.count_params()),
        'status': 'ready'
    }), 200


if __name__ == '__main__':
    port = AppConfig.PORT
    debug = AppConfig.DEBUG
    logger.info(f"Starting Flask app on port {port}")
    logger.info(f"Debug mode: {debug}")
    app.run(host='0.0.0.0', port=port, debug=debug)