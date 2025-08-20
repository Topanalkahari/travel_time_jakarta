import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Import pipeline classes dari struktur Anda
try:
    from src.pipeline import TravelTimePipeline
    from src.preprocessor import TravelTimePreprocessor
    PIPELINE_AVAILABLE = True
except ImportError as e:
    st.error(f"❌ Error importing pipeline: {str(e)}")
    st.error("Pastikan file src/pipeline.py dan src/preprocessor.py tersedia")
    PIPELINE_AVAILABLE = False

# Konfigurasi halaman Streamlit
st.set_page_config(
    page_title="Travel Time Prediction",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_validate_data(uploaded_file):
    """Load dan validasi data dari uploaded file"""
    try:
        data = pd.read_csv(uploaded_file)
        
        # Validasi kolom yang diperlukan
        required_columns = [
            'start_point', 'end_point', 'time_of_day', 'day_of_week',
            'traffic_condition', 'event_count', 'is_holiday', 'vehicle_density', 
            'population_density', 'weather', 'public_transport_availability', 
            'historical_delay_factor', 'travel_time'
        ]
        
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            st.error(f"❌ Missing required columns: {missing_columns}")
            return None
        
        return data
    
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return None

def create_prediction_form():
    """Membuat form untuk input prediksi"""
    st.markdown("<h2 class='sub-header'>🔮 Prediksi Travel Time</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        start_point = st.text_input("🏁 Start Point", value="West Jakarta")
        end_point = st.text_input("🏁 End Point", value="South Jakarta")
        day_of_week = st.selectbox(
            "📅 Day of Week",
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        )
        time_of_day = st.selectbox(
            "🕐 Time of Day",
            ["morning", "day", "evening", "night"]
        )
        is_holiday = st.selectbox(
            "🚦 Is Holiday",
            [1, 0],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )
        # Historical delay factor input
        historical_delay_factor = st.number_input(
            "⏰ Historical Delay Factor",
            min_value=0.0,
            max_value=3.0,
            value=1.0,
            step=0.1,
            help="Factor representing historical delays (0.0 = no delay, 3.0 = 3x normal time)"
        )
    
    with col2:
        vehicle_density = st.selectbox(
            "🚗 Vehicle Density",
            ["low", "medium", "high"]
        )
        population_density = st.selectbox(
            "👥 Population Density",
            ["low", "medium", "high"]
        )
        weather = st.selectbox(
            "🌤️ Weather",
            ["storm", "fog", "rain", "clear"]
        )
        traffic_condition = st.selectbox(
            "🚦 Traffic Condition",
            list(range(11)),  # 0 to 10
            format_func=lambda x: f"Level {x}"
        )
        public_transport_availability = st.selectbox(
            "🚦 Public Transport Availability",
            list(range(3)),  # 0 to 2
            format_func=lambda x: f"Level {x}"
        )
    
    event_count = st.slider("🎯 Event Count", min_value=0, max_value=20, value=2)
    
    return {
        'start_point': start_point,
        'end_point': end_point,
        'day_of_week': day_of_week,
        'time_of_day': time_of_day,
        'is_holiday': is_holiday,
        'vehicle_density': vehicle_density,
        'population_density': population_density,
        'weather': weather,
        'traffic_condition': traffic_condition,
        'public_transport_availability': public_transport_availability,
        'event_count': event_count,
        'historical_delay_factor': historical_delay_factor
    }

def plot_model_comparison(results_dict):
    """Plot perbandingan performa model"""
    models = list(results_dict.keys())
    metrics = ['MAE', 'RMSE', 'R²']
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=metrics,
        specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
    )
    
    mae_values = [results_dict[model]['mae'] for model in models]
    rmse_values = [results_dict[model]['rmse'] for model in models]
    r2_values = [results_dict[model]['r2'] for model in models]
    
    fig.add_trace(
        go.Bar(x=models, y=mae_values, name="MAE", marker_color='lightblue'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=models, y=rmse_values, name="RMSE", marker_color='lightcoral'),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Bar(x=models, y=r2_values, name="R²", marker_color='lightgreen'),
        row=1, col=3
    )
    
    fig.update_layout(
        title_text="Model Performance Comparison",
        showlegend=False,
        height=400
    )
    
    return fig

def plot_predictions_vs_actual(y_true, y_pred, title="Predictions vs Actual"):
    """Plot prediksi vs actual values"""
    fig = go.Figure()
    
    # Scatter plot
    fig.add_trace(go.Scatter(
        x=y_true,
        y=y_pred,
        mode='markers',
        name='Predictions',
        marker=dict(color='blue', size=8, opacity=0.6)
    ))
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Perfect Prediction',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Actual Travel Time (minutes)",
        yaxis_title="Predicted Travel Time (minutes)",
        height=500
    )
    
    return fig

def plot_residuals(y_true, y_pred):
    """Plot residuals"""
    residuals = y_true - y_pred
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=y_pred,
        y=residuals,
        mode='markers',
        name='Residuals',
        marker=dict(color='green', size=8, opacity=0.6)
    ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    
    fig.update_layout(
        title="Residuals Plot",
        xaxis_title="Predicted Travel Time (minutes)",
        yaxis_title="Residuals (Actual - Predicted)",
        height=400
    )
    
    return fig

def plot_feature_importance(feature_importance_df, top_n=10):
    """Plot feature importance"""
    top_features = feature_importance_df.head(top_n)
    
    fig = px.bar(
        x=top_features['importance'],
        y=top_features['feature'],
        orientation='h',
        title=f"Top {top_n} Most Important Features",
        labels={'x': 'Importance', 'y': 'Features'}
    )
    fig.update_layout(height=400)
    
    return fig

def main():
    """Main Streamlit application"""
    
    if not PIPELINE_AVAILABLE:
        st.stop()
    
    # Header
    st.markdown("<h1 class='main-header'>🚗 Travel Time Prediction App</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Aplikasi prediksi waktu perjalanan menggunakan Machine Learning</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    st.sidebar.markdown("## ⚙️ Configuration")
    
    # Mode selection
    mode = st.sidebar.selectbox(
        "Select Mode:",
        ["🔮 Single Prediction", "📊 Model Training & Evaluation", "📈 Batch Prediction"]
    )
    
    if mode == "🔮 Single Prediction":
        st.markdown("<h2 class='sub-header'>Single Travel Time Prediction</h2>", unsafe_allow_html=True)
        
        # Check if trained model exists
        model_path = "travel_time_model.pkl"
        
        if os.path.exists(model_path):
            try:
                # Load trained pipeline
                pipeline = TravelTimePipeline()
                pipeline.load_pipeline(model_path)
                st.success("✅ Pre-trained model loaded successfully!")
                
                # Prediction form
                input_data = create_prediction_form()
                
                if st.button("🚀 Predict Travel Time", type="primary"):
                    try:
                        # Create DataFrame from input
                        input_df = pd.DataFrame([input_data])
                        
                        # Make prediction
                        prediction = pipeline.predict(input_df)[0]
                        
                        # Display result
                        st.markdown("<div class='success-box'>", unsafe_allow_html=True)
                        st.markdown(f"### 🎯 Predicted Travel Time: **{prediction:.1f} minutes**")
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Show input summary
                        st.markdown("### 📋 Input Summary:")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**Route:** {input_data['start_point']} → {input_data['end_point']}")
                            st.write(f"**Day:** {input_data['day_of_week']}")
                            st.write(f"**Time:** {input_data['time_of_day']}")
                            st.write(f"**Is Holiday:** {input_data['is_holiday']}")
                            st.write(f"**Events:** {input_data['event_count']}")
                        
                        with col2:
                            st.write(f"**Vehicle Density:** {input_data['vehicle_density']}")
                            st.write(f"**Population Density:** {input_data['population_density']}")
                            st.write(f"**Weather:** {input_data['weather']}")
                            st.write(f"**Traffic:** {input_data['traffic_condition']}")
                            st.write(f"**Public Transport:** {input_data['public_transport_availability']}")
                            st.write(f"**Historical Delay Factor:** {input_data['historical_delay_factor']}")
                        
                        # Confidence indicator
                        if hasattr(pipeline.model, 'predict_proba') or hasattr(pipeline.model, 'decision_function'):
                            st.info("💡 Tip: Prediksi lebih akurat pada kondisi yang mirip dengan data training")
                        
                    except Exception as e:
                        st.error(f"❌ Error in prediction: {str(e)}")
                        st.info("Pastikan input data sesuai dengan format yang diharapkan model.")
                        
            except Exception as e:
                st.error(f"❌ Error loading model: {str(e)}")
                st.info("Model file mungkin corrupt atau tidak kompatibel.")
        else:
            st.warning("⚠️ No pre-trained model found!")
            st.info("Silakan train model terlebih dahulu menggunakan mode 'Model Training & Evaluation'.")
    
    elif mode == "📊 Model Training & Evaluation":
        st.markdown("<h2 class='sub-header'>Model Training & Evaluation</h2>", unsafe_allow_html=True)
        
        # File upload
        uploaded_file = st.file_uploader(
            "📁 Upload Training Data (CSV)",
            type=['csv'],
            help="Upload your training dataset dengan kolom: start_point, end_point, time_of_day, day_of_week, traffic_condition, event_count, is_holiday, vehicle_density, population_density, weather, public_transport_availability, historical_delay_factor, travel_time"
        )
        
        if uploaded_file is not None:
            # Load dan validasi data
            data = load_and_validate_data(uploaded_file)
            
            if data is not None:
                st.success(f"✅ Data loaded successfully! Shape: {data.shape}")
                
                # Show data preview
                with st.expander("👀 Data Preview & Statistics"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**📊 Data Preview (10 rows):**")
                        st.dataframe(data.head(10))
                    
                    with col2:
                        st.markdown("**📈 Data Statistics:**")
                        st.write(f"- Total Rows: {data.shape[0]:,}")
                        st.write(f"- Total Columns: {data.shape[1]}")
                        st.write(f"- Missing Values: {data.isnull().sum().sum()}")
                        
                        if 'travel_time' in data.columns:
                            st.markdown("**🎯 Target Variable (travel_time):**")
                            st.write(f"- Mean: {data['travel_time'].mean():.2f} min")
                            st.write(f"- Std: {data['travel_time'].std():.2f} min")
                            st.write(f"- Min: {data['travel_time'].min():.2f} min")
                            st.write(f"- Max: {data['travel_time'].max():.2f} min")
                
                # Data distribution visualization
                with st.expander("📊 Data Distribution"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Travel time distribution
                        fig_dist = px.histogram(
                            data, x='travel_time', 
                            title="Distribution of Travel Time",
                            nbins=30
                        )
                        st.plotly_chart(fig_dist, use_container_width=True)
                    
                    with col2:
                        # Route frequency
                        route_counts = data.groupby(['start_point', 'end_point']).size().reset_index(name='count')
                        route_counts['route'] = route_counts['start_point'] + ' → ' + route_counts['end_point']
                        
                        fig_routes = px.bar(
                            route_counts.head(10), 
                            x='count', y='route',
                            orientation='h',
                            title="Top 10 Most Frequent Routes"
                        )
                        st.plotly_chart(fig_routes, use_container_width=True)
                
                # Model configuration
                st.markdown("### 🤖 Model Configuration")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    model_choice = st.selectbox(
                        "Choose Model:",
                        ["Linear Regression", "Ridge Regression", "Random Forest"]
                    )
                    
                    # Model parameters
                    if model_choice == "Ridge Regression":
                        alpha = st.slider("Alpha (Regularization)", 0.1, 10.0, 1.0, 0.1)
                        model = Ridge(alpha=alpha)
                    elif model_choice == "Random Forest":
                        n_estimators = st.slider("Number of Trees", 10, 200, 100, 10)
                        max_depth = st.slider("Max Depth", 3, 20, 10)
                        model = RandomForestRegressor(
                            n_estimators=n_estimators, 
                            max_depth=max_depth, 
                            random_state=23
                        )
                    else:
                        model = LinearRegression()
                
                with col2:
                    test_size = st.slider("Test Size Ratio", 0.1, 0.4, 0.2, 0.05)
                    random_state = st.number_input("Random State", value=23, min_value=1)
                
                # Train model button
                if st.button("🚀 Train Model", type="primary"):
                    with st.spinner("Training model... This may take a few minutes."):
                        try:
                            # Initialize pipeline
                            pipeline = TravelTimePipeline(
                                model=model,
                                test_size=test_size,
                                random_state=int(random_state)
                            )
                            
                            # Set data
                            pipeline.data = data
                            
                            # Prepare data
                            pipeline.prepare_data('travel_time')
                            
                            # Progress bar
                            progress_bar = st.progress(0)
                            progress_bar.progress(25)
                            
                            # Fit pipeline
                            pipeline.fit()
                            progress_bar.progress(75)
                            
                            # Evaluate on training data
                            train_results = pipeline.evaluate(
                                pipeline.X_train, pipeline.y_train, "Training"
                            )
                            
                            # Evaluate on test data
                            test_results = pipeline.evaluate()
                            progress_bar.progress(100)
                            
                            # Store pipeline in session state
                            st.session_state['trained_pipeline'] = pipeline
                            st.session_state['train_results'] = train_results
                            st.session_state['test_results'] = test_results
                            
                            st.success("✅ Model trained successfully!")
                            
                            # Display results
                            st.markdown("### 📊 Model Performance")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("#### 🎯 Training Results")
                                st.metric("MAE", f"{train_results['mae']:.4f} min")
                                st.metric("RMSE", f"{train_results['rmse']:.4f} min")
                                st.metric("R²", f"{train_results['r2']:.4f}")
                            
                            with col2:
                                st.markdown("#### 🧪 Test Results")
                                st.metric("MAE", f"{test_results['mae']:.4f} min", 
                                         delta=f"{test_results['mae'] - train_results['mae']:.4f}")
                                st.metric("RMSE", f"{test_results['rmse']:.4f} min",
                                         delta=f"{test_results['rmse'] - train_results['rmse']:.4f}")
                                st.metric("R²", f"{test_results['r2']:.4f}",
                                         delta=f"{test_results['r2'] - train_results['r2']:.4f}")
                            
                            # Visualization
                            st.markdown("### 📈 Model Analysis")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                fig_pred = plot_predictions_vs_actual(
                                    pipeline.y_test.values, 
                                    test_results['predictions']
                                )
                                st.plotly_chart(fig_pred, use_container_width=True)
                            
                            with col2:
                                fig_res = plot_residuals(
                                    pipeline.y_test.values, 
                                    test_results['predictions']
                                )
                                st.plotly_chart(fig_res, use_container_width=True)
                            
                            # Feature importance
                            try:
                                importance_df = pipeline.get_feature_importance(top_n=15)
                                if importance_df is not None:
                                    st.markdown("### 📈 Feature Importance")
                                    fig_importance = plot_feature_importance(importance_df, top_n=10)
                                    st.plotly_chart(fig_importance, use_container_width=True)
                                    
                                    # Show feature importance table
                                    with st.expander("📋 Detailed Feature Importance"):
                                        st.dataframe(importance_df.head(20))
                            except Exception as e:
                                st.warning(f"⚠️ Could not compute feature importance: {str(e)}")
                            
                            # Model interpretation
                            st.markdown("### 🔍 Model Interpretation")
                            
                            overfitting_score = train_results['r2'] - test_results['r2']
                            
                            if overfitting_score > 0.1:
                                st.warning(f"⚠️ Possible overfitting detected (Train R² - Test R² = {overfitting_score:.3f})")
                                st.info("💡 Consider regularization or simpler model")
                            elif overfitting_score < -0.05:
                                st.info(f"📈 Model may have room for improvement (Train R² - Test R² = {overfitting_score:.3f})")
                            else:
                                st.success(f"✅ Good model balance (Train R² - Test R² = {overfitting_score:.3f})")
                            
                        except Exception as e:
                            st.error(f"❌ Error during training: {str(e)}")
                            st.info("Please check your data format and pipeline configuration.")
                
                # Save model section
                if 'trained_pipeline' in st.session_state:
                    st.markdown("### 💾 Save Trained Model")
                    
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        model_name = st.text_input(
                            "Model Name:", 
                            value="travel_time_model.pkl"
                        )
                    
                    with col2:
                        if st.button("💾 Save Model"):
                            try:
                                st.session_state['trained_pipeline'].save_pipeline(model_name)
                                st.success(f"✅ Model saved as '{model_name}'")
                            except Exception as e:
                                st.error(f"❌ Error saving model: {str(e)}")
        
        else:
            # Show sample data format
            st.info("📝 Upload a CSV file to start training. Expected format:")
            
            sample_data = pd.DataFrame({
                'start_point': ['Jakarta', 'Bandung', 'Surabaya'],
                'end_point': ['Bandung', 'Jakarta', 'Malang'],
                'time_of_day': ['morning', 'afternoon', 'evening'],
                'day_of_week': ['Monday', 'Tuesday', 'Wednesday'],
                'traffic_condition': [8, 5, 2],
                'event_count': [5, 2, 1],
                'is_holiday': [0, 0, 0],
                'vehicle_density': ['high', 'medium', 'low'],
                'population_density': ['high', 'medium', 'low'],
                'weather': ['sunny', 'rainy', 'cloudy'],
                'public_transport_availability': [2, 1, 0],
                'historical_delay_factor': [1.2, 1.0, 0.8],
                'travel_time': [120, 95, 85]
            })
            
            st.dataframe(sample_data)
            
            # Download sample template
            csv = sample_data.to_csv(index=False)
            st.download_button(
                label="📥 Download Sample Template",
                data=csv,
                file_name="travel_time_template.csv",
                mime="text/csv"
            )
    
    elif mode == "📈 Batch Prediction":
        st.markdown("<h2 class='sub-header'>Batch Travel Time Prediction</h2>", unsafe_allow_html=True)
        
        # Check for trained model
        model_path = "travel_time_model.pkl"
        
        if os.path.exists(model_path):
            try:
                # Load pipeline
                pipeline = TravelTimePipeline()
                pipeline.load_pipeline(model_path)
                st.success("✅ Pre-trained model loaded!")
                
                # File upload for batch prediction
                uploaded_file = st.file_uploader(
                    "📁 Upload Data for Prediction (CSV)",
                    type=['csv'],
                    help="Upload CSV file dengan format yang sama seperti data training (tanpa kolom travel_time)"
                )
                
                # Optional: Upload data with actual values for R² calculation
                actual_values_file = st.file_uploader(
                    "📁 Upload Data with Actual Values (Optional - for R² calculation)",
                    type=['csv'],
                    help="Upload CSV file with actual travel_time values to calculate R² score"
                )
                
                if uploaded_file is not None:
                    try:
                        # Load data
                        pred_data = pd.read_csv(uploaded_file)
                        st.info(f"📊 Data loaded: {pred_data.shape[0]:,} rows, {pred_data.shape[1]} columns")
                        
                        # Validate columns (exclude travel_time)
                        required_columns = [
                            'start_point', 'end_point', 'time_of_day', 'day_of_week',
                            'traffic_condition', 'event_count', 'is_holiday', 'vehicle_density', 
                            'population_density', 'weather', 'public_transport_availability',
                            'historical_delay_factor'
                        ]
                        
                        missing_columns = [col for col in required_columns if col not in pred_data.columns]
                        
                        if missing_columns:
                            st.error(f"❌ Missing required columns: {missing_columns}")
                        else:
                            # Show preview
                            with st.expander("👀 Data Preview"):
                                st.dataframe(pred_data.head(10))
                            
                            if st.button("🚀 Generate Predictions", type="primary"):
                                with st.spinner("Generating predictions..."):
                                    try:
                                        # Make predictions
                                        predictions = pipeline.predict(pred_data)
                                        
                                        # Add predictions to dataframe
                                        pred_data['predicted_travel_time'] = predictions
                                        
                                        st.success(f"✅ Generated {len(predictions):,} predictions!")
                                        
                                        # Calculate R² if actual values are provided
                                        r2_score_value = None
                                        if actual_values_file is not None:
                                            try:
                                                actual_data = pd.read_csv(actual_values_file)
                                                if 'travel_time' in actual_data.columns and len(actual_data) == len(predictions):
                                                    from sklearn.metrics import r2_score
                                                    r2_score_value = r2_score(actual_data['travel_time'], predictions)
                                                    st.success(f"🎯 R² Score: **{r2_score_value:.4f}**")
                                                else:
                                                    st.warning("⚠️ Actual values file must have 'travel_time' column and same number of rows as prediction data")
                                            except Exception as e:
                                                st.warning(f"⚠️ Could not calculate R²: {str(e)}")
                                        
                                        # Show results preview
                                        st.markdown("### 📊 Prediction Results Preview")
                                        display_columns = ['start_point', 'end_point', 'day_of_week', 'time_of_day', 'predicted_travel_time']
                                        available_columns = [col for col in display_columns if col in pred_data.columns]
                                        st.dataframe(pred_data[available_columns].head(10))
                                        
                                        # Statistics
                                        col1, col2, col3, col4, col5 = st.columns(5)
                                        
                                        with col1:
                                            st.metric("Total Predictions", f"{len(predictions):,}")
                                        with col2:
                                            st.metric("Average Time", f"{predictions.mean():.1f} min")
                                        with col3:
                                            st.metric("Min Time", f"{predictions.min():.1f} min")
                                        with col4:
                                            st.metric("Max Time", f"{predictions.max():.1f} min")
                                        with col5:
                                            if r2_score_value is not None:
                                                st.metric("R² Score", f"{r2_score_value:.4f}")
                                            else:
                                                st.metric("R² Score", "N/A", help="Upload actual values file to see R² score")
                                        
                                        # Visualizations
                                        col1, col2 = st.columns(2)
                                        
                                        with col1:
                                            # Distribution of predictions
                                            fig_dist = px.histogram(
                                                x=predictions,
                                                nbins=30,
                                                title="Distribution of Predicted Travel Times"
                                            )
                                            fig_dist.update_layout(
                                                xaxis_title="Predicted Travel Time (minutes)",
                                                yaxis_title="Frequency"
                                            )
                                            st.plotly_chart(fig_dist, use_container_width=True)
                                        
                                        with col2:
                                            # Predictions by day of week
                                            day_stats = pred_data.groupby('day_of_week')['predicted_travel_time'].agg(['mean', 'count']).reset_index()
                                            
                                            fig_day = px.bar(
                                                day_stats, 
                                                x='day_of_week', 
                                                y='mean',
                                                title="Average Predicted Travel Time by Day"
                                            )
                                            fig_day.update_layout(
                                                xaxis_title="Day of Week",
                                                yaxis_title="Average Travel Time (minutes)"
                                            )
                                            st.plotly_chart(fig_day, use_container_width=True)
                                        
                                        # Additional analytics
                                        with st.expander("📊 Detailed Analytics"):
                                            # Time of day analysis
                                            time_stats = pred_data.groupby('time_of_day')['predicted_travel_time'].agg(['mean', 'std', 'count']).round(2)
                                            st.markdown("**⏰ Predictions by Time of Day:**")
                                            st.dataframe(time_stats)
                                            
                                            # Weather impact
                                            weather_stats = pred_data.groupby('weather')['predicted_travel_time'].agg(['mean', 'std', 'count']).round(2)
                                            st.markdown("**🌤️ Predictions by Weather:**")
                                            st.dataframe(weather_stats)
                                        
                                        # Download results
                                        st.markdown("### 📥 Download Results")
                                        
                                        csv_result = pred_data.to_csv(index=False)
                                        st.download_button(
                                            label="📥 Download All Predictions (CSV)",
                                            data=csv_result,
                                            file_name="travel_time_predictions.csv",
                                            mime="text/csv"
                                        )
                                        
                                        # Download summary
                                        summary_stats = pred_data.groupby(['start_point', 'end_point'])['predicted_travel_time'].agg([
                                            'mean', 'std', 'min', 'max', 'count'
                                        ]).round(2).reset_index()
                                        
                                        summary_csv = summary_stats.to_csv(index=False)
                                        st.download_button(
                                            label="📊 Download Route Summary (CSV)",
                                            data=summary_csv,
                                            file_name="route_summary_predictions.csv",
                                            mime="text/csv"
                                        )
                                        
                                    except Exception as e:
                                        st.error(f"❌ Error generating predictions: {str(e)}")
                                        st.info("Please check your data format matches the training data.")
                    
                    except Exception as e:
                        st.error(f"❌ Error loading prediction data: {str(e)}")
                        
            except Exception as e:
                st.error(f"❌ Error loading model: {str(e)}")
                st.info("Please ensure the model file exists and is not corrupted.")
        
        else:
            st.warning("⚠️ No trained model found!")
            st.info("Silakan train model terlebih dahulu menggunakan mode 'Model Training & Evaluation'.")
            
            # Show expected format for prediction data
            st.markdown("### 📝 Expected Data Format for Batch Prediction:")
            
            sample_pred_data = pd.DataFrame({
                'start_point': ['Jakarta', 'Bandung', 'Surabaya'],
                'end_point': ['Bandung', 'Jakarta', 'Malang'],
                'time_of_day': ['morning', 'afternoon', 'evening'],
                'day_of_week': ['Monday', 'Tuesday', 'Wednesday'],
                'traffic_condition': [8, 5, 2],
                'event_count': [5, 2, 1],
                'is_holiday': [0, 0, 0],
                'vehicle_density': ['high', 'medium', 'low'],
                'population_density': ['high', 'medium', 'low'],
                'weather': ['sunny', 'rainy', 'cloudy'],
                'public_transport_availability': [2, 1, 0],
                'historical_delay_factor': [1.2, 1.0, 0.8]
            })
            
            st.dataframe(sample_pred_data)
            
            csv_template = sample_pred_data.to_csv(index=False)
            st.download_button(
                label="📥 Download Prediction Template",
                data=csv_template,
                file_name="prediction_template.csv",
                mime="text/csv"
            )
    
    # Sidebar - Model Comparison Feature
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🏆 Advanced Features")
    
    if st.sidebar.button("🔍 Compare Multiple Models"):
        if 'trained_pipeline' in st.session_state:
            st.markdown("### 🏆 Multi-Model Comparison")
            
            # Get the current data from session
            pipeline = st.session_state['trained_pipeline']
            
            # Define models to compare
            models_to_compare = {
                'Linear Regression': LinearRegression(),
                'Ridge Regression': Ridge(alpha=1.0),
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=23)
            }
            
            comparison_results = {}
            
            with st.spinner("Comparing models..."):
                progress_bar = st.progress(0)
                
                for i, (model_name, model) in enumerate(models_to_compare.items()):
                    try:
                        # Create new pipeline with different model
                        temp_pipeline = TravelTimePipeline(
                            model=model,
                            test_size=pipeline.test_size,
                            random_state=pipeline.random_state
                        )
                        
                        # Use same data split
                        temp_pipeline.X_train = pipeline.X_train
                        temp_pipeline.X_test = pipeline.X_test
                        temp_pipeline.y_train = pipeline.y_train
                        temp_pipeline.y_test = pipeline.y_test
                        temp_pipeline.data = pipeline.data
                        
                        # Fit and evaluate
                        temp_pipeline.fit()
                        results = temp_pipeline.evaluate()
                        comparison_results[model_name] = results
                        
                        progress_bar.progress((i + 1) / len(models_to_compare))
                        
                    except Exception as e:
                        st.warning(f"⚠️ Could not evaluate {model_name}: {str(e)}")
                
                if comparison_results:
                    # Plot comparison
                    fig_comparison = plot_model_comparison(comparison_results)
                    st.plotly_chart(fig_comparison, use_container_width=True)
                    
                    # Show detailed comparison table
                    comparison_df = pd.DataFrame({
                        model_name: {
                            'MAE': results['mae'],
                            'RMSE': results['rmse'],
                            'MSE': results['mse'],
                            'R²': results['r2']
                        } for model_name, results in comparison_results.items()
                    }).T.round(4)
                    
                    st.markdown("### 📋 Detailed Comparison")
                    st.dataframe(comparison_df)
                    
                    # Best model recommendation
                    best_model_mae = comparison_df['MAE'].idxmin()
                    best_model_r2 = comparison_df['R²'].idxmax()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.success(f"🏆 Best MAE: **{best_model_mae}** ({comparison_df.loc[best_model_mae, 'MAE']:.4f})")
                    with col2:
                        st.success(f"🏆 Best R²: **{best_model_r2}** ({comparison_df.loc[best_model_r2, 'R²']:.4f})")
        else:
            st.warning("⚠️ No trained model available for comparison. Train a model first.")
    
    # Cross-validation feature
    if st.sidebar.button("🔄 Cross Validation"):
        if 'trained_pipeline' in st.session_state:
            st.markdown("### 🔄 Cross Validation Results")
            
            try:
                from sklearn.model_selection import cross_val_score
                
                pipeline = st.session_state['trained_pipeline']
                
                with st.spinner("Performing cross validation..."):
                    # Prepare data for CV
                    X_processed = pipeline.preprocessor.transform(pipeline.X_train)
                    
                    # Perform cross validation
                    cv_scores = cross_val_score(
                        pipeline.model, X_processed, pipeline.y_train,
                        cv=5, scoring='r2'
                    )
                    
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("CV Mean R²", f"{cv_scores.mean():.4f}")
                    with col2:
                        st.metric("CV Std R²", f"{cv_scores.std():.4f}")
                    with col3:
                        st.metric("CV Range", f"{cv_scores.max() - cv_scores.min():.4f}")
                    
                    # Plot CV scores
                    fig_cv = go.Figure(data=go.Bar(
                        x=[f"Fold {i+1}" for i in range(len(cv_scores))],
                        y=cv_scores,
                        marker_color='lightblue'
                    ))
                    
                    fig_cv.add_hline(y=cv_scores.mean(), line_dash="dash", line_color="red", 
                                    annotation_text=f"Mean: {cv_scores.mean():.4f}")
                    
                    fig_cv.update_layout(
                        title="Cross Validation R² Scores",
                        yaxis_title="R² Score",
                        height=400
                    )
                    
                    st.plotly_chart(fig_cv, use_container_width=True)
                    
                    # Interpretation
                    if cv_scores.std() < 0.05:
                        st.success("✅ Model shows consistent performance across folds")
                    elif cv_scores.std() < 0.1:
                        st.info("📊 Model shows moderate consistency across folds")
                    else:
                        st.warning("⚠️ Model performance varies significantly across folds")
                
            except Exception as e:
                st.error(f"❌ Error in cross validation: {str(e)}")
        else:
            st.warning("⚠️ No trained model available. Train a model first.")
    
    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📚 About This App")
    st.sidebar.info("""
    **Travel Time Prediction App**
    
    **Features:**
    - 🔮 Single predictions
    - 📊 Model training & evaluation  
    - 📈 Batch predictions
    - 🏆 Model comparison
    - 🔄 Cross validation
    
    **Supported Models:**
    - Linear Regression
    - Ridge Regression  
    - Random Forest
    
    **Input Features:**
    - Route (start/end points)
    - Day and time information
    - Traffic & weather conditions
    - Vehicle & population density
    - Event count
    """)
    
    # Debug info (only show in development)
    if st.sidebar.checkbox("🐛 Show Debug Info"):
        st.sidebar.markdown("### 🔧 Debug Information")
        st.sidebar.write("**Session State Keys:**")
        st.sidebar.write(list(st.session_state.keys()))
        
        if 'trained_pipeline' in st.session_state:
            pipeline = st.session_state['trained_pipeline']
            st.sidebar.write("**Pipeline Info:**")
            st.sidebar.write(f"- Model: {type(pipeline.model).__name__}")
            st.sidebar.write(f"- Is Fitted: {pipeline.is_fitted}")
            st.sidebar.write(f"- Test Size: {pipeline.test_size}")

# Error handling untuk missing dependencies
def check_dependencies():
    """Check if all required dependencies are available"""
    required_packages = ['streamlit', 'pandas', 'numpy', 'plotly', 'sklearn', 'joblib']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        st.error(f"❌ Missing required packages: {missing_packages}")
        st.info("Install dengan: pip install " + " ".join(missing_packages))
        return False
    
    return True

if __name__ == "__main__":
    # Check dependencies
    if check_dependencies() and PIPELINE_AVAILABLE:
        main()
    elif not PIPELINE_AVAILABLE:
        st.error("❌ Pipeline modules tidak ditemukan!")
        st.info("""
        Pastikan struktur folder Anda seperti ini:
        ```
        your_project/
        ├── app.py
        ├── src/
        │   ├── __init__.py
        │   ├── pipeline.py
        │   └── preprocessor.py
        └── travel_time_model.pkl (opsional)
        ```
        """)
    else:
        st.error("❌ Missing required dependencies!")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666666;'>"
    "Travel Time Prediction App | Built with ❤️ using Streamlit"
    "</div>", 
    unsafe_allow_html=True
)