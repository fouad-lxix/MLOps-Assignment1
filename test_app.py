import pytest
from app import app


# Fixture to create a test client
@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


# Test for a successful prediction with valid input data
def test_predict_success(client):
    input_data = {
        'longitude': 122.23,
        'latitude': 37.88,
        'housing_median_age': 41.0,
        'total_rooms': 880.0,
        'total_bedrooms': 129.0,
        'population': 322.0,
        'households': 126.0,
        'median_income': 8.3252
    }
    response = client.post('/predict', json=input_data)
    # Assert that the prediction was successful
    assert response.status_code == 200
    json_data = response.get_json()
    assert 'prediction' in json_data
    assert isinstance(json_data['prediction'], float)


# Test for a successful prediction with different valid input data
def test_predict_success_different_data(client):
    input_data = {
        'longitude': 118.25,
        'latitude': 34.05,
        'housing_median_age': 30.0,
        'total_rooms': 1500.0,
        'total_bedrooms': 300.0,
        'population': 800.0,
        'households': 280.0,
        'median_income': 5.0000
    }
    response = client.post('/predict', json=input_data)
    # Assert that the prediction was successful
    assert response.status_code == 200
    json_data = response.get_json()
    assert 'prediction' in json_data
    assert isinstance(json_data['prediction'], float)


# Test for a successful prediction with edge case input data
def test_predict_success_edge_case(client):
    input_data = {
        'longitude': 120.0,
        'latitude': 35.0,
        'housing_median_age': 50.0,
        'total_rooms': 500.0,
        'total_bedrooms': 100.0,
        'population': 200.0,
        'households': 80.0,
        'median_income': 2.5000
    }
    response = client.post('/predict', json=input_data)
    # Assert that the prediction was successful
    assert response.status_code == 200
    json_data = response.get_json()
    assert 'prediction' in json_data
    assert isinstance(json_data['prediction'], float)
