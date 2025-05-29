> 📌 This repository contains supplementary code and technical validation materials for the manuscript
> **"EmoWorker: A Multimodal Dataset for Assessing Emotion, Stress, and Emotional Workload in Interpersonal Work Scenario"** (*under review*)

<br>
<div align="center">
    <img src="resources/emoworker_logo.png", width="400"/>
    <br>
    <br> A Multimodal Dataset for Assessing Emotion, Stress, and Emotional Workload in Interpersonal Work Scenario 
</div>

# EmoWorker: Technical Validation Code

This repository contains the code for technical validation of the EmoWorker dataset. The dataset itself is available at [Zenodo](https://zenodo.org/uploads/15181220).

## 📁 Repository Structure

```
TECHNICAL_VALIDATION/
│
├── Dataset_Records.ipynb     # Data source summary and preprocessing overview
├── Label_Analysis.ipynb      # Label distribution, missing data, and correlation analysis
└── ML_analysis.ipynb         # Machine learning model implementation and evaluation
```

## 🚀 Getting Started

We recommend using **Python 3.10**. Some dependencies may not be fully compatible with Python 3.11 or above. All notebooks were developed and tested in a Jupyter Lab environment using Python 3.10.

1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run the notebooks:
- TECHNICAL_VALIDATION/Dataset_Records.ipynb
- TECHNICAL_VALIDATION/Label_Analysis.ipynb
- TECHNICAL_VALIDATION/ML_analysis.ipynb

## 📝 Notebook Overview

### `Dataset_Records.ipynb`
Summarizes the dataset structure and provides a high-level overview of data sources and preprocessing steps. This notebook includes:
- Data collection protocol details
- Signal preprocessing steps
- Data quality checks
- Feature extraction methods
- Missing data analysis
- Data synchronization procedures

### `Label_Analysis.ipynb`
Analyzes the distribution of self-reported labels (e.g., perceived arousal, stress, suppression, valence), investigates missing values, and explores correlations and group differences (e.g., by gender or role). Key analyses include:
- Label distribution visualization
- Missing data patterns
- Correlation analysis between different measures
- Statistical tests for group differences
- Temporal analysis of emotional responses

### `ML_analysis.ipynb`
Builds machine learning models to predict each of the following target variables:  
- `perceived_arousal`  
- `perceived_stress`  
- `perceived_suppression`  
- `perceived_valence`  

The following models are implemented and evaluated:
- Decision Tree  
- Random Forest  
- Support Vector Machine (SVM)  
- XGBoost  
- k-Nearest Neighbors (kNN)  

Each model's performance is assessed using:
- Accuracy
- F1 score
- Precision and Recall
- ROC-AUC (where applicable)

## 🤝 Contributing

We welcome contributions to improve the code and documentation. Please feel free to submit issues and pull requests.

## 📄 License

This project is licensed under the terms of the license included in the repository.