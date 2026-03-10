# Human-AI Collaboration Framework

A comprehensive framework for human-AI collaboration in text classification tasks, emphasizing transparency, interpretability, and continuous learning from human feedback.

## ⚠️ Important Disclaimer

**This is a research and educational tool only.** The explanations and predictions provided by this system may be unstable, misleading, or incorrect. This tool is NOT intended for regulated decisions without human review. Always verify AI outputs and use human judgment for critical decisions.

## Overview

The Human-AI Collaboration Framework enables seamless interaction between humans and AI systems for text classification tasks. It provides:

- **AI Predictions** with confidence scores and uncertainty quantification
- **Multiple Explanation Methods** including SHAP, LIME, Attention visualization, and Integrated Gradients
- **Human Feedback Interface** for accepting, rejecting, or correcting AI predictions
- **Real-time Metrics** tracking collaboration effectiveness and model performance
- **Interactive Visualizations** for understanding model behavior and explanations

## Key Features

### AI Collaboration
- Adaptive confidence thresholds based on human feedback
- Real-time learning from human corrections
- Uncertainty quantification and calibration
- Multi-turn conversation support

### Explainability
- **SHAP**: Global and local feature importance
- **LIME**: Local interpretable model-agnostic explanations
- **Attention Visualization**: Transformer attention heatmaps
- **Integrated Gradients**: Gradient-based attribution methods
- **Ensemble Explanations**: Consensus across multiple methods

### Evaluation Metrics
- **Collaboration Metrics**: Accuracy, confidence calibration, intervention rates
- **Explanation Metrics**: Faithfulness, stability, consistency
- **Trust Metrics**: User satisfaction, explanation quality ratings
- **Fairness Metrics**: Bias detection across sensitive attributes

### Interactive Visualization
- Real-time explanation dashboards
- Collaboration performance tracking
- Confidence distribution analysis
- Feedback pattern visualization

## Installation

### Prerequisites
- Python 3.10+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)
- MPS (optional, for Apple Silicon)

### Quick Start

1. **Clone the repository:**
```bash
git clone https://github.com/kryptologyst/Human-AI-Collaboration-Framework.git
cd Human-AI-Collaboration-Framework
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
# or for development
pip install -e ".[dev]"
```

3. **Run the interactive demo:**
```bash
streamlit run demo/app.py
```

### Development Installation

```bash
# Install in development mode
pip install -e ".[dev,tracking,serving]"

# Set up pre-commit hooks
pre-commit install

# Run tests
pytest tests/

# Format code
black src/
ruff check src/
```

## Quick Start Guide

### Basic Usage

```python
from human_ai_collaboration import (
    TextClassifier,
    CollaborationModel,
    SHAPExplainer,
    LIMEExplainer,
    DataLoader
)

# Initialize model
model = TextClassifier(model_name="distilbert-base-uncased", num_classes=5)
collaboration_model = CollaborationModel(base_model=model)

# Load data
loader = DataLoader()
texts, labels, label_names = loader.load_20newsgroups(subset="test")

# Get AI prediction
suggested_label, confidence = collaboration_model.suggest_label(
    text="Sample text about technology",
    candidate_labels=label_names
)

# Process human feedback
result = collaboration_model.process_feedback(
    text="Sample text about technology",
    ai_suggestion=suggested_label,
    human_label="technology",
    confidence=confidence,
    feedback_type="accept"
)

# Generate explanations
shap_explainer = SHAPExplainer(model)
explanation = shap_explainer.explain("Sample text about technology")
```

### Interactive Demo

Launch the Streamlit demo for a full interactive experience:

```bash
streamlit run demo/app.py
```

The demo provides:
- Text input and AI prediction interface
- Multiple explanation visualization methods
- Human feedback collection and processing
- Real-time metrics and performance tracking
- Export capabilities for analysis

## Dataset Schema

### Supported Datasets

1. **20 Newsgroups**: Classic text classification dataset
2. **Synthetic Data**: Generated text data for testing
3. **Custom Data**: User-provided text classification data

### Data Format

```json
{
  "texts": ["Sample text 1", "Sample text 2", ...],
  "labels": [0, 1, 2, ...],
  "label_names": ["Class 0", "Class 1", "Class 2", ...],
  "metadata": {
    "num_classes": 5,
    "tokenizer_name": "distilbert-base-uncased",
    "max_length": 512
  }
}
```

### Feature Metadata

The framework tracks:
- **Text Features**: Token-level importance, attention weights
- **Sensitive Attributes**: For fairness evaluation (optional)
- **Monotonicity Constraints**: For interpretable models
- **Feature Types**: Categorical, numerical, text

## Training and Evaluation

### Training Commands

```bash
# Train with default configuration
python scripts/train.py --config configs/default.yaml

# Train with custom data
python scripts/train.py --data_path data/custom/ --model_name bert-base-uncased

# Train with specific explanation methods
python scripts/train.py --explanation_methods SHAP,LIME,Attention
```

### Evaluation Commands

```bash
# Evaluate model performance
python scripts/evaluate.py --model_path checkpoints/model.pth

# Evaluate explanation quality
python scripts/evaluate_explanations.py --model_path checkpoints/model.pth

# Run comprehensive evaluation
python scripts/full_evaluation.py --config configs/evaluation.yaml
```

### Configuration

The framework uses YAML configuration files:

```yaml
# configs/default.yaml
data:
  dataset_name: "20newsgroups"
  tokenizer_name: "distilbert-base-uncased"
  max_length: 512

model:
  name: "distilbert-base-uncased"
  num_classes: 20
  dropout_rate: 0.1

collaboration:
  confidence_threshold: 0.8
  feedback_weight: 0.1

explanation:
  methods: ["SHAP", "LIME", "Attention", "IntegratedGradients"]
```

## Explanation Methods

### SHAP (SHapley Additive exPlanations)
- **Type**: Model-agnostic feature importance
- **Use Case**: Global and local explanations
- **Output**: Feature importance scores

### LIME (Local Interpretable Model-agnostic Explanations)
- **Type**: Local surrogate model explanations
- **Use Case**: Understanding individual predictions
- **Output**: Local feature importance

### Attention Visualization
- **Type**: Transformer attention weights
- **Use Case**: Understanding model focus
- **Output**: Attention heatmaps

### Integrated Gradients
- **Type**: Gradient-based attribution
- **Use Case**: Feature attribution analysis
- **Output**: Token-level attributions

## Evaluation Metrics

### Collaboration Metrics
- **Overall Accuracy**: Model performance with human feedback
- **Confidence Calibration**: Alignment between confidence and accuracy
- **Intervention Rate**: Frequency of human corrections
- **Response Time**: Human feedback processing time

### Explanation Metrics
- **Faithfulness**: How well explanations reflect model behavior
- **Stability**: Consistency across similar inputs
- **Consistency**: Agreement between different explanation methods

### Trust Metrics
- **User Satisfaction**: Human ratings of explanations
- **Confidence Consistency**: Stability of confidence scores
- **Uncertainty Calibration**: Proper uncertainty quantification

### Fairness Metrics
- **Group Accuracy**: Performance across different groups
- **Fairness Gaps**: Disparities in model performance
- **Bias Detection**: Identification of systematic biases

## Limitations and Considerations

### Explanation Limitations
- **Instability**: Explanations may vary for similar inputs
- **Method Dependency**: Different methods may give conflicting results
- **Context Sensitivity**: Explanations depend on training data and model architecture
- **Human Interpretation**: Explanations require human judgment for validation

### Model Limitations
- **Domain Specificity**: Models trained on specific domains may not generalize
- **Bias Propagation**: Training data biases may be reflected in explanations
- **Confidence Calibration**: Confidence scores may not always reflect true uncertainty
- **Adversarial Robustness**: Models may be vulnerable to adversarial inputs

### Collaboration Limitations
- **Human Fatigue**: Continuous feedback may lead to decreased attention
- **Feedback Quality**: Inconsistent or incorrect human feedback affects learning
- **Scalability**: Human feedback may not scale to large datasets
- **Individual Differences**: Different humans may provide conflicting feedback

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Fork and clone the repository
git clone <your-fork-url>
cd human-ai-collaboration-framework

# Install development dependencies
pip install -e ".[dev]"

# Set up pre-commit hooks
pre-commit install

# Run tests
pytest tests/

# Run linting
black src/
ruff check src/
mypy src/
```

### Code Style
- Follow PEP 8 guidelines
- Use type hints for all functions
- Write comprehensive docstrings
- Include unit tests for new features

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{human_ai_collaboration_framework,
  title={Human-AI Collaboration Framework for Explainable AI},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Human-AI-Collaboration-Framework}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built on top of [Transformers](https://github.com/huggingface/transformers)
- Explanation methods from [SHAP](https://github.com/slundberg/shap), [LIME](https://github.com/marcotcr/lime), [Captum](https://github.com/pytorch/captum)
- Visualization powered by [Plotly](https://plotly.com/) and [Streamlit](https://streamlit.io/)

## Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Check the documentation
- Join our community discussions

---

**Remember**: This tool is for research and educational purposes only. Always use human judgment for critical decisions and verify AI outputs independently.
# Human-AI-Collaboration-Framework
