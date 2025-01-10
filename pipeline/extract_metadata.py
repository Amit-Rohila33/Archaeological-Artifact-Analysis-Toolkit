from models.artifact_predictor import ArtifactPredictor

def extract_and_predict_metadata(metadata_text):
    predictor = ArtifactPredictor()
    prediction = predictor.predict(metadata_text)
    return prediction  # Return class (cultural or historical context)
