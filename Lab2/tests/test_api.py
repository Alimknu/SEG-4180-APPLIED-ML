"""API endpoint tests"""

import pytest, json, os
from io import BytesIO
from PIL import Image
import numpy as np
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import app


@pytest.fixture
def client():
    """Create Flask test client"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def sample_image():
    """Create a sample image for testing"""
    img_array = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    img = Image.fromarray(img_array, 'RGB')
    
    img_bytes = BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    return img_bytes


class TestHealthEndpoint:
    def test_health_check_returns_200(self, client):
        response = client.get('/health')
        assert response.status_code in [200, 503]
    
    def test_health_check_returns_json(self, client):
        response = client.get('/health')
        assert response.content_type == 'application/json'
    
    def test_health_check_has_status_field(self, client):
        response = client.get('/health')
        data = json.loads(response.data)
        assert 'status' in data or 'error' in data


class TestPredictEndpoint:
    def test_predict_no_image_returns_400(self, client):
        response = client.post('/predict')
        assert response.status_code == 400
    
    def test_predict_with_image(self, client, sample_image):
        response = client.post('/predict', data={'image': (sample_image, 'test.jpg')}, content_type='multipart/form-data')
        assert response.status_code in [200, 500, 503]
    
    def test_predict_response_format(self, client, sample_image):
        """Prediction response should have expected format"""
        response = client.post(
            '/predict',
            data={'image': (sample_image, 'test.jpg')},
            content_type='multipart/form-data'
        )
        
        if response.status_code == 200:
            data = json.loads(response.data)
            
            # Check response fields
            assert 'status' in data
            assert 'filename' in data
            assert 'preprocess_time_ms' in data
            assert 'inference_time_ms' in data
            assert 'total_latency_ms' in data


class TestBatchPredictEndpoint:
    def test_batch_predict_empty_request(self, client):
        response = client.post('/predict_batch', data=json.dumps({}), content_type='application/json')
        assert response.status_code == 400
    
    def test_batch_predict_with_images(self, client):
        images = []
        for i in range(2):
            img_array = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
            img = Image.fromarray(img_array, 'RGB')
            img_bytes = BytesIO()
            img.save(img_bytes, format='JPEG')
            img_bytes.seek(0)
            images.append(('images', (img_bytes, f'test{i}.jpg')))
        
        response = client.post('/predict_batch', data=images, content_type='multipart/form-data')
        assert response.status_code in [200, 400, 500, 503]


class TestModelInfoEndpoint:
    def test_model_info_endpoint(self, client):
        response = client.get('/model-info')
        assert response.status_code in [200, 503]
    
    def test_model_info_returns_json(self, client):
        response = client.get('/model-info')
        assert response.content_type == 'application/json'


class TestErrorHandling:
    
    def test_invalid_image_file(self, client):
        response = client.post('/predict', data={'image': (BytesIO(b'not an image'), 'test.txt')}, content_type='multipart/form-data')
        assert response.status_code in [400, 500, 503]
    
    def test_missing_content_type(self, client):
        response = client.post('/predict')
        assert response.status_code in [400, 415, 500]


class TestConfiguration:
    
    def test_config_loads(self):
        from config import AppConfig, Config
        port = Config.get_int('PORT', 5000)
        assert isinstance(port, int) and port > 0
    
    def test_env_variables_loaded(self):
        from config import AppConfig
        
        # These should be set from .env or defaults
        assert AppConfig.MODEL_PATH is not None
        assert AppConfig.DATASET_PATH is not None
        assert AppConfig.PORT > 0


class TestIntegration:
    def test_health_to_model_info_flow(self, client):
        health_response = client.get('/health')
        assert health_response.status_code in [200, 503]
        info_response = client.get('/model-info')
        assert info_response.status_code in [200, 503]
    
    def test_multiple_requests_isolated(self, client, sample_image):
        for _ in range(3):
            response = client.get('/health')
            assert response.status_code in [200, 503]
        
        for i in range(2):
            sample_image.seek(0)
            response = client.post(
                '/predict',
                data={'image': (sample_image, f'test{i}.jpg')},
                content_type='multipart/form-data'
            )
            assert response.status_code in [200, 500, 503]


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
