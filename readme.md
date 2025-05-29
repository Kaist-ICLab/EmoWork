
# EmoWorker: Technical Validation Code

<div align="center">
    <img src="resources/emoworker_logo.png", width="400"/>
</div>

> 📌 This repository contains supplementary code and technical validation materials for the manuscript
> **"EmoWorker: A Multimodal Dataset for Assessing Emotion, Stress, and Emotional Workload in Interpersonal Work Scenario"**  
> currently under review. This page is shared as part of the manuscript submission for reproducibility and transparency purposes.  
> **Citation and DOI will be updated upon publication.**

We provide Jupyter notebooks that support our dataset processing, label analysis, and machine learning modeling, under the `TECHNICAL_VALIDATION/` folder. These materials aim to enhance transparency and reproducibility of our findings.

## 📁 Folder Structure

```
TECHNICAL_VALIDATION/
│
├── Dataset_Records.ipynb     # Data source summary and preprocessing overview
├── Label_Analysis.ipynb      # Label distribution, missing data, and correlation analysis
└── ML_analysis.ipynb         # Machine learning model implementation and evaluation
```

## Requirements

To reproduce the analyses, install the required Python packages listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

We recommend using **Python 3.10**. Some dependencies may not be fully compatible with Python 3.11 or above. All notebooks were developed and tested in a Jupyter Lab environment using Python 3.10.

## Reproducibility

All random seeds were fixed to ensure reproducibility. The notebooks are organized to run sequentially, starting from data description to model training and evaluation.

---

## Notebook Overview

### `Dataset_Records.ipynb`
Summarizes the dataset structure and provides a high-level overview of data sources and preprocessing steps.

### `Label_Analysis.ipynb`
Analyzes the distribution of self-reported labels (e.g., perceived arousal, stress, suppression, valence), investigates missing values, and explores correlations and group differences (e.g., by gender or role).

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

Each model's performance is assessed using standard metrics (e.g., accuracy, F1 score), and cross-validation is applied for robust evaluation.

---

## Citation

If you use this code or refer to our dataset/analysis, please cite the following paper (to be updated upon publication):

```bibtex
@unpublished{emoworker2025,
  title     = {EmoWorker: A Multimodal Dataset for Assessing Emotion, Stress, and Emotional Workload in Interpersonal Work Scenario},
  author    = {Author1 and Author2 and Author3},
  year      = {2025},
  note      = {Manuscript in preparation. Citation and DOI to be updated upon publication.}
}
```