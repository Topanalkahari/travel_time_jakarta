from src.utils import run_complete_pipeline
import pandas as pd

if __name__ == "__main__":
    pipeline = run_complete_pipeline(filepath="data/train_sample.csv")
    pipeline.save_pipeline("models/travel_time_model.pkl")

    # contoh prediksi
    #sample_preds = pipeline.predict(pipeline.X_test.iloc[:5])
    #print("Sample prediction:", sample_preds)
    
    # Contoh prediksi pada data baru
    print("\n🔮 Contoh prediksi pada sample data:")
    sample_predictions = pipeline.predict(pipeline.X_test.iloc[:5])
    actual_values = pipeline.y_test.iloc[:5].values
    
    comparison_df = pd.DataFrame({
        'Actual': actual_values,
        'Predicted': sample_predictions,
        'Difference': actual_values - sample_predictions
    })
    print(comparison_df)