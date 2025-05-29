
<br>
<div align="center">
    <img src="figures/emoworker_logo.png", width="400"/>
    <br>
    <br> A Multimodal Dataset for Assessing **Emotion**, **Stress**, and **Emotional Workload** in Interpersonal Work Scenario 
</div>

# EmoWorker: Technical Validation Code
📌 This repository contains supplementary code and technical validation materials for the manuscript
> **"EmoWorker: A Multimodal Dataset for Assessing Emotion, Stress, and Emotional Workload in Interpersonal Work Scenario"** (*under review*)

The dataset itself is available at [Zenodo - EmoWorker](https://zenodo.org/uploads/15181220).

## 📁 Repository Structure

```
TECHNICAL_VALIDATION/
├── Dataset_Records.ipynb     # Data source summary and preprocessing overview
├── Label_Analysis.ipynb      # Label distribution, missing data, and correlation analysis
├── ML_analysis.ipynb         # Machine learning model implementation and evaluation
└── utils/                    # Utility scripts

RESULTS/
├── Condition/                # Session classification results (GT = session)
│   └── [model_name]/         # e.g., DecisionTree, RandomForest, ...
│       ├── all_runs_results.csv
│       └── summary_5runs.csv
├── Perceived/                # Label prediction results (GT = perceived_*)
│   └── [label_name]/         # e.g., perceived_arousal, perceived_stress, ...
│       └── [model_name]/     # e.g., XGBoost, SVM, ...
│           ├── all_runs_results.csv
│           └── summary_5runs.csv

figures/
├── emoworker_logo.png
├── sensor_data/              # Visualizations from Dataset_Records.ipynb
├── label_analysis/           # Visualizations from Label_Analysis.ipynb
└── model_results/            # Visualizations from ML_analysis.ipynb

LICENSE
README.md
requirements.txt
```

## 🚀 Getting Started

We recommend using **Python 3.10**. Some dependencies may not be fully compatible with Python 3.11. All notebooks were developed and tested using Python 3.10.

1. Clone this repository
```bash
git clone https://github.com/Kaist-ICLab/EmoWorker.git
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run the notebooks in **`TECHNICAL_VALIDATION`** folder:
- `Dataset_Records.ipynb`
- `Label_Analysis.ipynb`
- `ML_analysis.ipynb`

## 📝 Notebook Overview

### `Dataset_Records.ipynb`
Summarizes the dataset structure and provides a high-level overview of data sources and preprocessing steps. This notebook includes:
- Data collection protocol details
- Signal preprocessing steps
- Data quality checks
- Feature extraction methods
- Missing data analysis
- Data synchronization procedures

<img src="figures/sensor_data/p01_polar_hr.png" width="600"/>
<p align="center"><i>Example of heart rate signal collected from Polar H10</i></p>

Additional visualizations generated from this notebook are available in the [`figures/sensor_data/`](figures/sensor_data) directory.


### `Label_Analysis.ipynb`
Analyzes the distribution of self-reported labels (e.g., perceived arousal, stress, suppression, valence), investigates missing values, and explores correlations and group differences (e.g., by gender or role). Key analyses include:
- Label distribution visualization
- Missing data patterns
- Correlation analysis between different measures
- Statistical tests for group differences
- Temporal analysis of emotional responses

<img src="figures/label_analysis/arousal_valence_distribution.png" width="600"/>
<p align="center"><i>Distribution of perceived arousal and valence across all participants</i></p>

Additional visualizations generated from this notebook are available in the [`figures/label_analysis/`](figures/label_analysis) directory.

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

<img src="figures/model_results/auc_random_forest_by_pnum.png" width="600"/>
<p align="center"><i>Participant-wise AUC scores for session classification using a Random Forest model</i></p>

Additional visualizations generated from this notebook are available in the [`figures/model_results/`](figures/model_results) directory.

## 🤝 Contributing

We welcome contributions to improve the code and documentation. Please feel free to submit issues and pull requests.

## 📄 License

This project is licensed under the terms of the license included in the repository.