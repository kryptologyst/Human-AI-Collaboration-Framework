"""Streamlit demo for Human-AI Collaboration Framework."""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from human_ai_collaboration import (
    TextClassifier,
    CollaborationModel,
    DataLoader,
    SyntheticDataGenerator,
    SHAPExplainer,
    LIMEExplainer,
    AttentionExplainer,
    IntegratedGradientsExplainer,
    ExplanationEnsemble,
    CollaborationMetrics,
    ExplanationMetrics,
    TrustMetrics,
    FairnessMetrics,
    set_seed,
    get_device,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Human-AI Collaboration Framework",
    page_icon="🤝",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Disclaimer
st.markdown("""
<div class="warning-box">
<h4>⚠️ Important Disclaimer</h4>
<p><strong>This is a research and educational tool only.</strong> The explanations and predictions provided by this system may be unstable, misleading, or incorrect. This tool is NOT intended for regulated decisions without human review. Always verify AI outputs and use human judgment for critical decisions.</p>
</div>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">🤝 Human-AI Collaboration Framework</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Configuration")

# Initialize session state
if "collaboration_model" not in st.session_state:
    st.session_state.collaboration_model = None
if "metrics" not in st.session_state:
    st.session_state.metrics = CollaborationMetrics()
if "explanation_metrics" not in st.session_state:
    st.session_state.explanation_metrics = ExplanationMetrics()
if "feedback_history" not in st.session_state:
    st.session_state.feedback_history = []

# Model selection
st.sidebar.subheader("Model Configuration")
model_name = st.sidebar.selectbox(
    "Model",
    ["distilbert-base-uncased", "bert-base-uncased", "roberta-base"],
    help="Select the pre-trained model to use"
)

confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.8,
    step=0.05,
    help="Threshold below which human feedback is requested"
)

# Data selection
st.sidebar.subheader("Data Configuration")
data_source = st.sidebar.radio(
    "Data Source",
    ["Synthetic Data", "20 Newsgroups", "Custom Text"],
    help="Choose the data source for testing"
)

# Explanation methods
st.sidebar.subheader("Explanation Methods")
explanation_methods = st.sidebar.multiselect(
    "Select Explanation Methods",
    ["SHAP", "LIME", "Attention", "IntegratedGradients"],
    default=["SHAP", "LIME"],
    help="Choose which explanation methods to use"
)

# Initialize model
@st.cache_resource
def initialize_model(model_name: str, confidence_threshold: float):
    """Initialize the collaboration model."""
    try:
        set_seed(42)
        device = get_device()
        
        # Create base model
        base_model = TextClassifier(model_name=model_name, num_classes=5)
        
        # Create collaboration model
        collaboration_model = CollaborationModel(
            base_model=base_model,
            confidence_threshold=confidence_threshold
        )
        
        return collaboration_model, device
    except Exception as e:
        st.error(f"Error initializing model: {e}")
        return None, None

# Load data
@st.cache_data
def load_data(data_source: str):
    """Load data based on selection."""
    try:
        if data_source == "Synthetic Data":
            generator = SyntheticDataGenerator(random_seed=42)
            texts, labels, label_names = generator.generate_text_classification_data(
                n_samples=100, n_classes=5
            )
        elif data_source == "20 Newsgroups":
            loader = DataLoader(random_seed=42)
            texts, labels, label_names = loader.load_20newsgroups(
                subset="test", categories=None
            )
            # Limit to first 100 samples for demo
            texts = texts[:100]
            labels = labels[:100]
        else:
            # Custom text - will be handled in the interface
            texts = ["Sample text for demonstration"]
            labels = [0]
            label_names = ["Class 0", "Class 1", "Class 2", "Class 3", "Class 4"]
        
        return texts, labels, label_names
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return [], [], []

# Initialize model and data
collaboration_model, device = initialize_model(model_name, confidence_threshold)
texts, labels, label_names = load_data(data_source)

if collaboration_model is None:
    st.error("Failed to initialize model. Please check the configuration.")
    st.stop()

# Main interface
tab1, tab2, tab3, tab4 = st.tabs(["🤖 AI Collaboration", "📊 Explanations", "📈 Metrics", "⚙️ Settings"])

with tab1:
    st.header("Human-AI Collaboration Interface")
    
    # Text input
    if data_source == "Custom Text":
        input_text = st.text_area(
            "Enter text to classify:",
            value="This is a sample text about technology and innovation.",
            height=100
        )
    else:
        # Sample selection
        sample_idx = st.selectbox(
            "Select a sample text:",
            range(len(texts)),
            format_func=lambda x: f"Sample {x+1}: {texts[x][:100]}..."
        )
        input_text = texts[sample_idx]
        true_label = label_names[labels[sample_idx]]
        
        st.text_area("Selected text:", value=input_text, height=100, disabled=True)
        st.info(f"True label: {true_label}")
    
    # AI prediction
    if st.button("Get AI Prediction", type="primary"):
        with st.spinner("Generating prediction..."):
            try:
                # Get AI suggestion
                suggested_label, confidence = collaboration_model.suggest_label(
                    input_text, label_names, return_confidence=True
                )
                
                # Display prediction
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Predicted Label", suggested_label)
                
                with col2:
                    st.metric("Confidence", f"{confidence:.3f}")
                
                with col3:
                    needs_feedback = collaboration_model.should_request_human_feedback(confidence)
                    st.metric("Needs Feedback", "Yes" if needs_feedback else "No")
                
                # Confidence visualization
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=confidence,
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": "Confidence Score"},
                    gauge={
                        "axis": {"range": [None, 1]},
                        "bar": {"color": "darkblue"},
                        "steps": [
                            {"range": [0, 0.5], "color": "lightgray"},
                            {"range": [0.5, 0.8], "color": "yellow"},
                            {"range": [0.8, 1], "color": "green"}
                        ],
                        "threshold": {
                            "line": {"color": "red", "width": 4},
                            "thickness": 0.75,
                            "value": confidence_threshold
                        }
                    }
                ))
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Store prediction for feedback
                st.session_state.current_prediction = {
                    "text": input_text,
                    "suggested_label": suggested_label,
                    "confidence": confidence,
                    "true_label": true_label if data_source != "Custom Text" else None
                }
                
            except Exception as e:
                st.error(f"Error generating prediction: {e}")
    
    # Human feedback
    if "current_prediction" in st.session_state:
        st.subheader("Human Feedback")
        
        prediction = st.session_state.current_prediction
        
        # Feedback options
        feedback_type = st.radio(
            "How would you rate this prediction?",
            ["Accept", "Reject", "Correct"],
            help="Accept: AI prediction is correct, Reject: AI prediction is wrong, Correct: Provide correct label"
        )
        
        if feedback_type == "Correct":
            human_label = st.selectbox(
                "Select the correct label:",
                label_names,
                help="Choose the correct label for this text"
            )
        else:
            human_label = prediction["suggested_label"]
        
        # Additional feedback
        additional_feedback = st.text_area(
            "Additional comments (optional):",
            placeholder="Any additional feedback about the prediction or explanation..."
        )
        
        if st.button("Submit Feedback"):
            try:
                # Process feedback
                result = collaboration_model.process_feedback(
                    text=prediction["text"],
                    ai_suggestion=prediction["suggested_label"],
                    human_label=human_label,
                    confidence=prediction["confidence"],
                    feedback_type=feedback_type.lower()
                )
                
                # Update metrics
                st.session_state.metrics.update(
                    prediction=prediction["suggested_label"],
                    ground_truth=human_label,
                    confidence=prediction["confidence"],
                    feedback_type=feedback_type.lower()
                )
                
                # Store feedback history
                feedback_entry = {
                    "text": prediction["text"],
                    "ai_suggestion": prediction["suggested_label"],
                    "human_label": human_label,
                    "confidence": prediction["confidence"],
                    "feedback_type": feedback_type.lower(),
                    "additional_feedback": additional_feedback,
                    "is_correct": prediction["suggested_label"] == human_label
                }
                st.session_state.feedback_history.append(feedback_entry)
                
                st.success("Feedback submitted successfully!")
                
                # Clear current prediction
                del st.session_state.current_prediction
                
                # Rerun to update interface
                st.rerun()
                
            except Exception as e:
                st.error(f"Error processing feedback: {e}")

with tab2:
    st.header("Explanation Analysis")
    
    if "current_prediction" in st.session_state:
        prediction = st.session_state.current_prediction
        
        # Generate explanations
        if st.button("Generate Explanations"):
            with st.spinner("Generating explanations..."):
                try:
                    # Initialize explainers
                    explainers = {}
                    
                    if "SHAP" in explanation_methods:
                        explainers["SHAP"] = SHAPExplainer(collaboration_model.base_model)
                    
                    if "LIME" in explanation_methods:
                        explainers["LIME"] = LIMEExplainer(collaboration_model.base_model)
                    
                    if "Attention" in explanation_methods:
                        try:
                            explainers["Attention"] = AttentionExplainer(collaboration_model.base_model)
                        except ValueError:
                            st.warning("Attention explainer not available for this model")
                    
                    if "IntegratedGradients" in explanation_methods:
                        explainers["IntegratedGradients"] = IntegratedGradientsExplainer(collaboration_model.base_model)
                    
                    # Generate explanations
                    explanations = {}
                    for method, explainer in explainers.items():
                        try:
                            explanations[method] = explainer.explain(prediction["text"])
                        except Exception as e:
                            st.warning(f"Failed to generate {method} explanation: {e}")
                            explanations[method] = {"error": str(e)}
                    
                    st.session_state.explanations = explanations
                    
                except Exception as e:
                    st.error(f"Error generating explanations: {e}")
        
        # Display explanations
        if "explanations" in st.session_state:
            explanations = st.session_state.explanations
            
            # Explanation tabs
            explanation_tabs = st.tabs(list(explanations.keys()))
            
            for i, (method, explanation) in enumerate(explanations.items()):
                with explanation_tabs[i]:
                    if "error" in explanation:
                        st.error(f"Error in {method}: {explanation['error']}")
                    else:
                        st.subheader(f"{method} Explanation")
                        
                        # Basic info
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Predicted Class", explanation.get("predicted_class", "N/A"))
                        with col2:
                            st.metric("Confidence", f"{explanation.get('confidence', 0):.3f}")
                        with col3:
                            st.metric("Method", method)
                        
                        # Feature importance
                        if "feature_importance" in explanation:
                            st.subheader("Feature Importance")
                            
                            features = explanation["feature_importance"][:10]
                            df = pd.DataFrame(features)
                            
                            # Create bar chart
                            fig = px.bar(
                                df,
                                x="importance",
                                y="feature",
                                orientation="h",
                                title=f"{method} Feature Importance",
                                color="importance",
                                color_continuous_scale="Blues"
                            )
                            fig.update_layout(height=400)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Display table
                            st.dataframe(df, use_container_width=True)
                        
                        # Attention visualization
                        if "attention_map" in explanation:
                            st.subheader("Attention Visualization")
                            
                            attention_map = explanation["attention_map"]
                            tokens = [item["token"] for item in attention_map]
                            attentions = [item["attention"] for item in attention_map]
                            
                            # Create heatmap
                            fig = go.Figure(data=go.Heatmap(
                                z=[attentions],
                                x=tokens,
                                y=["Attention"],
                                colorscale="Blues"
                            ))
                            fig.update_layout(
                                title="Attention Heatmap",
                                height=200
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Attribution visualization
                        if "attribution_map" in explanation:
                            st.subheader("Attribution Analysis")
                            
                            attribution_map = explanation["attribution_map"][:10]
                            df = pd.DataFrame(attribution_map)
                            
                            # Create bar chart
                            fig = px.bar(
                                df,
                                x="attribution",
                                y="token",
                                orientation="h",
                                title=f"{method} Token Attribution",
                                color="attribution",
                                color_continuous_scale="RdBu"
                            )
                            fig.update_layout(height=400)
                            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Please generate a prediction first to see explanations.")

with tab3:
    st.header("Collaboration Metrics")
    
    if st.session_state.feedback_history:
        # Convert feedback history to DataFrame
        df = pd.DataFrame(st.session_state.feedback_history)
        
        # Overall metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_feedback = len(df)
            st.metric("Total Feedback", total_feedback)
        
        with col2:
            accuracy = df["is_correct"].mean()
            st.metric("Overall Accuracy", f"{accuracy:.3f}")
        
        with col3:
            avg_confidence = df["confidence"].mean()
            st.metric("Average Confidence", f"{avg_confidence:.3f}")
        
        with col4:
            intervention_rate = (df["feedback_type"] != "accept").mean()
            st.metric("Intervention Rate", f"{intervention_rate:.3f}")
        
        # Detailed metrics
        st.subheader("Detailed Analysis")
        
        # Accuracy over time
        df["cumulative_accuracy"] = df["is_correct"].expanding().mean()
        
        fig = px.line(
            df,
            x=df.index,
            y="cumulative_accuracy",
            title="Cumulative Accuracy Over Time",
            labels={"x": "Feedback Number", "cumulative_accuracy": "Cumulative Accuracy"}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Confidence distribution
        fig = px.histogram(
            df,
            x="confidence",
            color="is_correct",
            title="Confidence Distribution by Accuracy",
            labels={"confidence": "Confidence Score", "count": "Count"}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Feedback type distribution
        feedback_counts = df["feedback_type"].value_counts()
        fig = px.pie(
            values=feedback_counts.values,
            names=feedback_counts.index,
            title="Feedback Type Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Raw data
        st.subheader("Feedback History")
        st.dataframe(df, use_container_width=True)
        
    else:
        st.info("No feedback data available yet. Start collaborating to see metrics!")

with tab4:
    st.header("Settings and Configuration")
    
    # Model settings
    st.subheader("Model Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Current Model:** {model_name}")
        st.write(f"**Device:** {device}")
        st.write(f"**Confidence Threshold:** {confidence_threshold}")
    
    with col2:
        st.write(f"**Data Source:** {data_source}")
        st.write(f"**Explanation Methods:** {', '.join(explanation_methods)}")
        st.write(f"**Total Samples:** {len(texts)}")
    
    # Reset options
    st.subheader("Reset Options")
    
    if st.button("Reset All Data", type="secondary"):
        st.session_state.feedback_history = []
        st.session_state.metrics = CollaborationMetrics()
        st.session_state.explanation_metrics = ExplanationMetrics()
        st.success("All data reset successfully!")
        st.rerun()
    
    # Export options
    st.subheader("Export Data")
    
    if st.session_state.feedback_history:
        # Convert to CSV
        df = pd.DataFrame(st.session_state.feedback_history)
        csv = df.to_csv(index=False)
        
        st.download_button(
            label="Download Feedback History",
            data=csv,
            file_name="feedback_history.csv",
            mime="text/csv"
        )
    
    # About section
    st.subheader("About")
    st.markdown("""
    This Human-AI Collaboration Framework demonstrates how humans and AI can work together
    to improve text classification tasks. The system provides:
    
    - **AI Predictions** with confidence scores
    - **Multiple Explanation Methods** (SHAP, LIME, Attention, Integrated Gradients)
    - **Human Feedback Interface** for accepting, rejecting, or correcting predictions
    - **Real-time Metrics** tracking collaboration effectiveness
    - **Interactive Visualizations** for understanding model behavior
    
    **Key Features:**
    - Adaptive confidence thresholds
    - Multiple explanation methods
    - Comprehensive metrics tracking
    - Interactive visualization dashboard
    
    **Use Cases:**
    - Research and education
    - Model debugging and validation
    - Human-AI interaction studies
    - Explainable AI demonstrations
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.8rem;">
    Human-AI Collaboration Framework | Research & Educational Use Only
</div>
""", unsafe_allow_html=True)
