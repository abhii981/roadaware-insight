# RoadAware Insight

Roads Accident Prediction is a machine learning-powered web application designed to analyze road accident data, identify accident-prone regions, and predict accident severity based on environmental and traffic-related factors.

The project combines machine learning, data analytics, geospatial visualization, and generative AI to provide meaningful insights into accident patterns and risk factors. The frontend was built using Lovable, while the backend and machine learning services were developed using FastAPI and Python.

---

## Key Highlights

- ML-powered accident severity prediction
- Interactive analytics dashboard
- Folium-based accident hotspot mapping
- Confusion matrix visualization
- Multi-model performance comparison
- Gemini + LangChain AI assistant
- FastAPI backend with REST APIs
- Responsive Lovable frontend

---

## Features

### Accident Severity Prediction

Predicts accident severity using machine learning models trained on real-world accident data. Users can input accident-related parameters and receive severity predictions instantly.

### Accident Hotspot Map

Built using Folium to visualize accident-prone locations on an interactive map. The hotspot view helps identify high-risk regions, geographical clusters, and accident concentration patterns.

### Analytics Dashboard

Provides interactive charts and visualizations for exploring accident trends and model performance, including:

- Accuracy comparison
- Precision comparison
- Recall comparison
- F1-Score comparison
- ROC-AUC comparison
- Accident severity distribution
- Traffic density analysis
- Weather impact analysis
- Road type analysis
- Feature importance visualization

### Confusion Matrix Visualization

Displays confusion matrices for trained models to better understand prediction performance and classification errors.

### AI Assistant

Integrated Gemini + LangChain powered assistant that allows users to ask questions about the dataset, accident trends, model outputs, and generated insights using natural language.

### Model Comparison

Compare multiple machine learning algorithms side-by-side and evaluate their strengths and weaknesses through visual analytics.

---

## Dataset

The project uses the Indian Road Accident Dataset obtained from Kaggle.

Dataset Characteristics:

- Approximately 20,000 accident records
- 24 features
- Weather conditions
- Traffic density
- Road type information
- Casualty details
- Accident severity labels
- Risk-related attributes

---

## Data Challenges

One of the major challenges in this project was the highly imbalanced nature of the dataset. Certain accident severity categories contained significantly fewer samples than others, making classification difficult.

Additionally, accident severity depends on a large number of interacting variables such as weather conditions, traffic density, road characteristics, time-related attributes, and environmental factors. Due to this high feature complexity, accident severity prediction is inherently a challenging multi-class classification problem.

To improve model performance, extensive preprocessing and balancing techniques were applied before training.

---

## Data Preprocessing & Analytics

The following techniques were used:

- Missing Value Handling
- IQR-Based Outlier Detection and Removal
- Label Encoding
- Feature Scaling
- MinMax Normalization
- Standardization
- Stratified Train-Test Split
- SMOTE Oversampling for Class Balancing
- Feature Importance Analysis
- 5-Fold Cross Validation

Additional analytics techniques:

- K-Means Clustering
- DBSCAN Anomaly Detection

---

## Machine Learning Models

The following models were trained and evaluated:

- XGBoost
- Random Forest
- Logistic Regression
- K-Nearest Neighbors (KNN)

### Best Performing Model: XGBoost

| Metric | Score |
|----------|----------|
| Accuracy | 67.27% |
| Precision | 60.52% |
| Recall | 67.27% |
| F1-Score | 60.65% |
| ROC-AUC | 0.7596 |

Although the accuracy is moderate, the results should be viewed in the context of a highly imbalanced dataset and the complex nature of accident severity prediction. Metrics such as F1-Score and ROC-AUC provide a more reliable assessment of model performance than accuracy alone.

---

## Tech Stack

### Frontend

- Lovable
- React
- TypeScript
- JavaScript
- Tailwind CSS

### Backend

- FastAPI
- Python

### Machine Learning

- Scikit-Learn
- XGBoost
- Pandas
- NumPy
- Imbalanced-Learn (SMOTE)

### AI Integration

- Google Gemini
- LangChain

### Visualization

- Folium
- Interactive JavaScript Charts
- Geospatial Hotspot Mapping

---

## Project Workflow

1. Collect and preprocess accident data.
2. Handle missing values and remove outliers.
3. Balance classes using SMOTE.
4. Train and evaluate multiple machine learning models.
5. Generate predictions through FastAPI APIs.
6. Visualize accident trends and model metrics.
7. Identify accident hotspots through Folium-based mapping.
8. Provide natural language insights using Gemini and LangChain.

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/your-username/roadaware-insight.git
cd roadaware-insight
```

### Backend Setup

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

### Frontend Setup

```bash
npm install
npm run dev
```

---

## Future Improvements

- Real-time traffic integration
- Live weather API integration
- Deep learning-based prediction models
- Route-wise accident risk prediction
- Real-time accident alerts
- Mobile application support

---

## Author

**Abhinandan**

Machine Learning • Data Analytics • Full Stack Development
