import pandas as pd
from src.pipeline import TravelTimePipeline

def run_complete_pipeline(filepath, target_column='travel_time', model=None, test_size=0.2, random_state=23):
    pipeline = TravelTimePipeline(model=model, test_size=test_size, random_state=random_state)
    pipeline.load_data(filepath)
    pipeline.prepare_data(target_column)
    pipeline.fit()
    pipeline.evaluate(pipeline.X_train, pipeline.y_train, "Training")
    pipeline.evaluate()
    pipeline.get_feature_importance()
    return pipeline
