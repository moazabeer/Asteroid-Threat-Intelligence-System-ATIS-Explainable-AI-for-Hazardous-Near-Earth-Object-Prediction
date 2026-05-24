# Asteroid Threat Intelligence System (ATIS) ☄️🤖
### Explainable AI for Hazardous Near-Earth Object Prediction

Asteroids are significant celestial bodies whose tracking and classification are vital for planetary defense. **ATIS** is an end-to-end Machine Learning pipeline designed to predict Potentially Hazardous Asteroids (PHAs) by leveraging state-of-the-art gradient boosting algorithms and Explainable AI (XAI) for transparent decision-making.

---

## 🚀 Features

- **Automated Data Retrieval**: Downloads the latest asteroid data using `kagglehub`.
- **Advanced Preprocessing**: Robust handling of missing values, encoding of orbital parameters, and feature selection.
- **Class Imbalance Handling**: Utilizes **SMOTE** (Synthetic Minority Over-sampling Technique) to address the rarity of hazardous asteroids in datasets.
- **High-Performance Models**: Implements and compares multiple ensemble architectures:
  - Random Forest
  - XGBoost
  - LightGBM
  - CatBoost
- **Explainable AI (XAI)**: Integral use of **SHAP** (SHapley Additive exPlanations) to interpret model predictions and identify critical orbital indicators of threat.
- **Hyperparameter Optimization**: Use of **Optuna** for automated tuning to maximize F1-score and ROC-AUC.

## 🛠️ Tech Stack

- **Language:** Python
- **Data Manipulation:** `pandas`, `numpy`
- **Machine Learning:** `scikit-learn`, `xgboost`, `lightgbm`, `catboost`
- **Optimization:** `optuna`
- **Explainability:** `shap`
- **Visualisation:** `matplotlib`, `seaborn`
- **Imbalance Handling:** `imbalanced-learn`

## 📦 Installation

To run this project locally, clone the repository and install the required dependencies:

```bash
git clone https://github.com/your-username/ATIS-Asteroid-Threat-Intelligence.git
cd ATIS-Asteroid-Threat-Intelligence
pip install lightgbm xgboost catboost optuna shap imbalanced-learn scikit-learn pandas numpy matplotlib seaborn kagglehub
```

## 📖 Usage

1. **Dataset Download**: The script automatically downloads the dataset via `kagglehub`. Ensure you have a Kaggle account/API key configured if necessary, or simply run the script to let `kagglehub` handle it.
2. **Execution**:
   ```bash
   python asteroid_threat_intelligence_system_(atis).py
   ```
3. **Outputs**:
   - Model performance metrics (Accuracy, Precision, Recall, F1, AUC).
   - Confusion matrices.
   - SHAP summary plots indicating the most influential features for hazard prediction.

## 📊 Dataset

The system utilizes the **Asteroid Dataset** from Kaggle, which includes orbital parameters like:
- `a`: Semi-major axis
- `e`: Eccentricity
- `i`: Inclination
- `q`: Perihelion distance
- `pha`: Potentially Hazardous Asteroid (Target label)

## 🔍 Explainability (SHAP)

Understanding *why* an asteroid is flagged as a threat is as important as the prediction itself. ATIS uses SHAP to visualize how features like Perihelion distance or H (Absolute magnitude) contribute to the model's confidence in a threat assessment.

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---
**Disclaimer**: This project is for educational and research purposes only. For official planetary defense data, please refer to [NASA JPL CNEOS](https://cneos.jpl.nasa.gov/).
