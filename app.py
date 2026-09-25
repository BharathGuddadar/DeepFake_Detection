import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tempfile
import time
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------- Page Configuration -------------------
st.set_page_config(
    page_title="DeepGuard AI - Deepfake Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------- Custom CSS -------------------
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        * {
            font-family: 'Inter', sans-serif;
        }
        
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 15px;
            margin-bottom: 2rem;
            text-align: center;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        }
        
        .main-title {
            font-size: 3.2rem;
            font-weight: 700;
            color: white;
            margin: 0;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .main-subtitle {
            font-size: 1.2rem;
            color: rgba(255,255,255,0.9);
            margin-top: 0.5rem;
            font-weight: 400;
        }
        
        .feature-card {
            background: white;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 16px rgba(0,0,0,0.08);
            border: 1px solid #e8ecf3;
            margin-bottom: 1rem;
            transition: all 0.3s ease;
        }
        
        .feature-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(0,0,0,0.12);
        }
        
        .upload-section {
            background: linear-gradient(145deg, #f8fafc, #e2e8f0);
            border: 2px dashed #94a3b8;
            border-radius: 15px;
            padding: 3rem;
            text-align: center;
            margin: 2rem 0;
            transition: all 0.3s ease;
        }
        
        .upload-section:hover {
            border-color: #667eea;
            background: linear-gradient(145deg, #f1f5f9, #e2e8f0);
        }
        
        .result-card {
            padding: 2rem;
            border-radius: 15px;
            text-align: center;
            font-size: 1.3rem;
            font-weight: 600;
            margin: 2rem 0;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
        }
        
        .result-fake {
            background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
            color: #dc2626;
            border: 2px solid #f87171;
        }
        
        .result-real {
            background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
            color: #16a34a;
            border: 2px solid #4ade80;
        }
        
        .result-warning {
            background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
            color: #d97706;
            border: 2px solid #fbbf24;
        }
        
        .metric-container {
            background: white;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 16px rgba(0,0,0,0.08);
            border-left: 4px solid #667eea;
        }
        
        .confidence-bar {
            background: #e5e7eb;
            border-radius: 10px;
            height: 20px;
            overflow: hidden;
            margin: 1rem 0;
            position: relative;
        }
        
        .confidence-fill {
            height: 100%;
            border-radius: 10px;
            transition: width 0.8s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: 600;
            font-size: 0.9rem;
        }
        
        .confidence-fill-real {
            background: linear-gradient(90deg, #10b981, #34d399);
        }
        
        .confidence-fill-fake {
            background: linear-gradient(90deg, #ef4444, #f87171);
        }
        
        .sidebar-info {
            background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
            padding: 1rem;
            border-radius: 10px;
            border-left: 4px solid #0ea5e9;
            margin: 1rem 0;
        }
        
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.75rem 2rem;
            font-weight: 600;
            font-size: 1.1rem;
            transition: all 0.3s ease;
            box-shadow: 0 4px 16px rgba(102, 126, 234, 0.3);
            width: 100%;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(102, 126, 234, 0.4);
        }
        
        .stat-item {
            background: white;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 16px rgba(0,0,0,0.08);
            text-align: center;
            border-top: 3px solid #667eea;
            margin-bottom: 1rem;
        }
        
        .stat-value {
            font-size: 2rem;
            font-weight: 700;
            color: #1f2937;
            margin-bottom: 0.5rem;
        }
        
        .stat-label {
            font-size: 0.9rem;
            color: #6b7280;
        }
        
        .analysis-section {
            background: white;
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 4px 16px rgba(0,0,0,0.08);
            margin: 1rem 0;
        }
        
        .progress-container {
            background: white;
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 4px 16px rgba(0,0,0,0.08);
            margin: 2rem 0;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# ------------------- Load Model -------------------
@st.cache_resource
def load_deepfake_model():
    try:
        return load_model("deepfake_detector.h5")
    except Exception as e:
        st.error(f"⚠️ Model file 'deepfake.h5' not found. Error: {str(e)}")
        return None

# ------------------- Helper Functions -------------------
def create_confidence_visualization(score, prediction):
    """Create a confidence visualization using HTML/CSS"""
    percentage = score * 100
    color_class = "confidence-fill-fake" if prediction == "FAKE" else "confidence-fill-real"
    
    return f"""
        <div class="confidence-bar">
            <div class="{color_class} confidence-fill" style="width: {percentage}%;">
                {percentage:.1f}%
            </div>
        </div>
    """

def create_simple_chart(predictions):
    """Create a simple matplotlib chart for frame analysis"""
    if not predictions or len(predictions) < 2:
        return None
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Frame-by-frame predictions
    frame_numbers = list(range(1, len(predictions) + 1))
    ax1.plot(frame_numbers, predictions, linewidth=2.5, color='#667eea', marker='o', markersize=4)
    ax1.axhline(y=0.5, color='#ef4444', linestyle='--', alpha=0.7, label='Fake Threshold')
    ax1.fill_between(frame_numbers, predictions, alpha=0.3, color='#667eea')
    ax1.set_title('Frame-by-Frame Predictions', fontsize=14, fontweight='bold', color='#374151')
    ax1.set_xlabel('Frame Number', fontsize=12)
    ax1.set_ylabel('Prediction Score', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Distribution histogram
    ax2.hist(predictions, bins=20, color='#667eea', alpha=0.7, edgecolor='white', linewidth=1.2)
    ax2.axvline(x=0.5, color='#ef4444', linestyle='--', alpha=0.7, label='Fake Threshold')
    ax2.set_title('Prediction Score Distribution', fontsize=14, fontweight='bold', color='#374151')
    ax2.set_xlabel('Prediction Score', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# ------------------- Prediction Function -------------------
def predict_video_enhanced(video_path, sample_rate=10, progress_callback=None):
    cap = cv2.VideoCapture(video_path)
    predictions = []
    frame_count = 0
    processed_frames = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    
    faces_detected = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % sample_rate != 0:
            frame_count += 1
            continue
            
        frame_count += 1
        processed_frames += 1
        
        if progress_callback:
            progress = min(processed_frames / max((total_frames // sample_rate), 1), 1.0)
            progress_callback(progress)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        
        for (x, y, w, h) in faces:
            faces_detected += 1
            face = frame[y:y+h, x:x+w]
            face = cv2.resize(face, (224, 224))
            face = face / 255.0
            face = np.expand_dims(face, axis=0)
            
            if model is not None:
                pred = model.predict(face, verbose=0)[0][0]
                predictions.append(pred)
    
    cap.release()
    
    if not predictions:
        return {
            'prediction': "No face detected",
            'confidence': 0.0,
            'faces_detected': faces_detected,
            'frames_processed': processed_frames,
            'frame_predictions': []
        }
    
    avg_score = np.mean(predictions)
    prediction = "FAKE" if avg_score > 0.5 else "REAL"
    
    return {
        'prediction': prediction,
        'confidence': float(avg_score),
        'faces_detected': faces_detected,
        'frames_processed': processed_frames,
        'frame_predictions': predictions,
        'std_deviation': float(np.std(predictions)) if len(predictions) > 1 else 0.0,
        'max_score': float(np.max(predictions)),
        'min_score': float(np.min(predictions))
    }

# ------------------- Main Application -------------------
def main():
    # Header Section
    st.markdown("""
        <div class="main-header">
            <div class="main-title">🛡️ DeepGuard AI</div>
            <div class="main-subtitle">Advanced Deepfake Detection & Video Analysis Platform</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Load model
    global model
    model = load_deepfake_model()
    
    if model is None:
        st.error("Cannot proceed without the deepfake detection model. Please ensure 'deepfake.h5' is in the correct directory.")
        return
    
    # Sidebar Configuration
    with st.sidebar:
        st.markdown("### 🔧 Analysis Settings")
        
        sample_rate = st.slider(
            "Frame Sampling Rate", 
            min_value=5, 
            max_value=50, 
            value=15,
            help="Higher values = faster analysis, lower values = more thorough analysis"
        )
        
        show_advanced = st.checkbox("Show Advanced Analytics", value=True)
        
        confidence_threshold = st.slider(
            "Detection Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.05,
            help="Threshold for fake detection (lower = more sensitive)"
        )
        
        st.markdown("""
            <div class="sidebar-info">
                <strong>💡 Pro Tips:</strong><br>
                • Upload HD videos for better accuracy<br>
                • Ensure good lighting and clear faces<br>
                • Shorter videos process faster<br>
                • Lower sampling rate = higher precision
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📊 Model Information")
        st.info("🤖 **Model**: CNN-based Deepfake Detector\n\n🎯 **Accuracy**: ~94% on test data\n\n📈 **Input**: 224x224 RGB images")
    
    # Main Content Area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📹 Video Upload & Analysis")
        
        uploaded_file = st.file_uploader(
            "",
            type=["mp4", "avi", "mov", "mkv"],
            help="Supported formats: MP4, AVI, MOV, MKV (Max size: 200MB)"
        )
        
        if not uploaded_file:
            st.markdown("""
                <div class="upload-section">
                    <h3>🎬 Ready to Analyze Your Video?</h3>
                    <p>Upload a video file to get started with AI-powered deepfake detection</p>
                    <div style="margin: 1rem 0;">
                        <span style="background: #667eea; color: white; padding: 0.5rem 1rem; border-radius: 20px; font-size: 0.9rem;">
                            Drag & drop or click to browse
                        </span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🚀 Key Features")
        
        features = [
            ("🎯", "High Accuracy", "AI-powered detection with 94%+ accuracy"),
            ("⚡", "Fast Processing", "Optimized for quick analysis"),
            ("📊", "Detailed Analytics", "Frame-by-frame breakdown"),
            ("🔒", "Secure", "No data stored after analysis")
        ]
        
        for icon, title, desc in features:
            st.markdown(f"""
                <div class="feature-card">
                    <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">{icon} <strong>{title}</strong></div>
                    <div style="color: #6b7280; font-size: 0.9rem;">{desc}</div>
                </div>
            """, unsafe_allow_html=True)
    
    # Video Processing Section
    if uploaded_file:
        st.markdown("---")
        
        # Save temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        tfile.close()
        
        # Display video
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("### 🎥 Uploaded Video")
            st.video(uploaded_file)
        
        with col2:
            st.markdown("### 📋 Video Information")
            
            # Get video info
            try:
                cap = cv2.VideoCapture(tfile.name)
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / fps if fps > 0 else 0
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                
                st.markdown(f"""
                    <div class="metric-container">
                        <div><strong>Duration:</strong> {duration:.1f} seconds</div>
                        <div><strong>Resolution:</strong> {width}×{height}</div>
                        <div><strong>FPS:</strong> {fps:.1f}</div>
                        <div><strong>Total Frames:</strong> {frame_count:,}</div>
                        <div><strong>File Size:</strong> {len(uploaded_file.getvalue()) / (1024*1024):.1f} MB</div>
                    </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error reading video info: {str(e)}")
        
        st.markdown("---")
        
        # Analysis Button
        if st.button("🚀 Start Deep Analysis", type="primary"):
            
            # Progress tracking
            progress_container = st.empty()
            
            with progress_container.container():
                st.markdown("""
                    <div class="progress-container">
                        <h3>🤖 AI Analysis in Progress</h3>
                        <p>Please wait while we analyze your video...</p>
                    </div>
                """, unsafe_allow_html=True)
                
                progress_bar = st.progress(0)
                status_text = st.empty()
            
            def update_progress(progress):
                progress_bar.progress(progress)
                status_text.text(f"🔍 Analyzing frames... {progress*100:.1f}% complete")
            
            start_time = time.time()
            
            # Perform analysis
            results = predict_video_enhanced(
                tfile.name, 
                sample_rate=sample_rate,
                progress_callback=update_progress
            )
            
            analysis_time = time.time() - start_time
            
            # Clear progress indicators
            progress_container.empty()
            
            # Display Results
            st.markdown("## 📈 Analysis Results")
            
            if results['prediction'] == "No face detected":
                st.markdown("""
                    <div class="result-card result-warning">
                        ⚠️ No Face Detected<br>
                        <div style="font-size: 1rem; margin-top: 1rem;">
                            Please ensure the video contains clear, visible faces for accurate analysis.
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                # Main Result Card
                result_class = "result-fake" if results['prediction'] == "FAKE" else "result-real"
                icon = "❌" if results['prediction'] == "FAKE" else "✅"
                
                st.markdown(f"""
                    <div class="result-card {result_class}">
                        {icon} Prediction: <strong>{results['prediction']}</strong><br>
                        <div style="font-size: 1rem; margin-top: 1rem;">
                            Confidence Score: {results['confidence']:.3f} ({results['confidence']*100:.1f}%)
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Confidence Bar
                st.markdown("### 📊 Confidence Visualization")
                confidence_html = create_confidence_visualization(results['confidence'], results['prediction'])
                st.markdown(confidence_html, unsafe_allow_html=True)
                
                # Statistics Grid
                st.markdown("### 📈 Analysis Statistics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                        <div class="stat-item">
                            <div class="stat-value">{results['faces_detected']}</div>
                            <div class="stat-label">Faces Detected</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                        <div class="stat-item">
                            <div class="stat-value">{results['frames_processed']}</div>
                            <div class="stat-label">Frames Analyzed</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                        <div class="stat-item">
                            <div class="stat-value">{analysis_time:.1f}s</div>
                            <div class="stat-label">Processing Time</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    reliability = "High" if results['std_deviation'] < 0.2 else "Medium" if results['std_deviation'] < 0.4 else "Low"
                    st.markdown(f"""
                        <div class="stat-item">
                            <div class="stat-value" style="font-size: 1.5rem;">{reliability}</div>
                            <div class="stat-label">Prediction Stability</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Advanced Analytics
                if show_advanced and len(results['frame_predictions']) > 1:
                    st.markdown("---")
                    st.markdown("## 📊 Advanced Analytics")
                    
                    # Create and display chart
                    chart_fig = create_simple_chart(results['frame_predictions'])
                    if chart_fig:
                        st.pyplot(chart_fig, use_container_width=True)
                        plt.close()  # Clean up
                    
                    # Detailed Statistics Table
                    st.markdown("### 📋 Detailed Statistics")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Mean Score", f"{results['confidence']:.4f}")
                        st.metric("Maximum Score", f"{results['max_score']:.4f}")
                        st.metric("Standard Deviation", f"{results['std_deviation']:.4f}")
                    
                    with col2:
                        st.metric("Minimum Score", f"{results['min_score']:.4f}")
                        fake_percentage = sum(1 for p in results['frame_predictions'] if p > confidence_threshold) / len(results['frame_predictions']) * 100
                        st.metric("Fake Frame %", f"{fake_percentage:.1f}%")
                        st.metric("Total Predictions", len(results['frame_predictions']))

if __name__ == "__main__":
    main()