# TOOL-BOX Repository

A comprehensive machine learning and data science toolbox with **8 specialized modular tools**, API server, web interface, and example projects.

## 📁 Repository Structure

```
TOOL-BOX/
├── Tool_box/                         # Modular ML tools package
│   ├── __init__.py                   # Package initialization
│   ├── data_processing_tool.py       # Data cleaning & preprocessing
│   ├── classification_tool.py        # 15 classification algorithms
│   ├── regression_tool.py            # 16 regression algorithms
│   ├── model_evaluation_tool.py      # Model evaluation & comparison
│   ├── cross_validation_tool.py      # 6 cross-validation techniques
│   ├── clustering_tool.py            # 9 clustering algorithms
│   ├── optimizer.py                  # Hyperparameter optimization
│   ├── feature_selector.py           # Feature selection methods
│   ├── model_interpreter.py          # Model interpretation tools
│   └── decorators.py                 # Utility decorators
├── projects/                         # Example projects
│   ├── Diabet_project/               # Diabetes prediction
│   ├── Heartrate_project/            # Heart rate prediction
│   └── Housing_project/              # Housing price prediction
├── interface/                        # Web interfaces
│   ├── html/                         # HTML frontend (7 steps)
│   ├── html_new/                     # New HTML interface
│   └── streamlit/                    # Streamlit app
├── api_server.py                     # FastAPI REST server
├── run_api_server.py                 # Server launcher
├── run_pipeline_demo.py              # Pipeline demo script
└── requirements.txt                  # Python dependencies
```

## 🛠️ Available Tools

### Core ML Tools
- **Data Processing Tool**: Comprehensive data cleaning, preprocessing, and EDA
- **Classification Tool**: 15 algorithms (Logistic, RF, SVM, GB, KNN, NB, Decision Tree, AdaBoost, Extra Trees, MLP, XGBoost, LightGBM, CatBoost, QDA, Ridge)
- **Regression Tool**: 16 algorithms (Linear, Ridge, Lasso, ElasticNet, Huber, RF, SVR, GB, HistGradientBoosting, KNN, Decision Tree, Extra Trees, MLP, XGBoost, LightGBM, CatBoost)
- **Model Evaluation Tool**: Comprehensive evaluation metrics and visualization

### Advanced ML Tools
- **Cross Validation Tool**: 6 techniques (K-Fold, Stratified, Time Series, Repeated, Leave-One-Out, Shuffle Split)
- **Clustering Tool**: 9 algorithms (K-Means, Mini-Batch K-Means, DBSCAN, HDBSCAN, Hierarchical, Spectral, BIRCH, OPTICS, Gaussian Mixture)
- **Optimizer**: Hyperparameter optimization with grid and random search
- **Feature Selector**: Feature selection methods
- **Model Interpreter**: Model interpretation and explainability

### API & Interface
- **API Server**: FastAPI REST endpoints for ML pipeline
- **Web Interface**: HTML frontend with 7-step workflow
- **Streamlit App**: Interactive Streamlit interface

## 📊 Example Projects

### Classification
- **Diabetes Prediction**: Neural network classification using medical data

### Regression
- **Heart Rate Prediction**: Multi-algorithm comparison for physiological data
- **Housing Price Prediction**: California housing price prediction with feature engineering

## 🚀 Quick Start

### Complete ML Workflow
```python
from Tool_box import (
    DataProcessingTool,      # Data preparation
    ClassificationTool,      # Multiple classification models
    ModelEvaluationTool,     # Evaluation & comparison
    CrossValidationTool,     # Advanced validation
    HyperparameterTuningTool # Optimization
)

# Data processing
processor = DataProcessingTool()
data = processor.load_data("data.csv")
processed_data = processor.prepare_data_for_ml(data, target_column="target")

# Model training & evaluation
classifier = ClassificationTool()
models = classifier.train_multiple_models(processed_data['X_train'], processed_data['y_train'])

evaluator = ModelEvaluationTool()
results = evaluator.evaluate_classification_models(models, processed_data['X_test'], processed_data['y_test'])
evaluator.plot_model_comparison(results)
```

### Running Projects
```bash
cd projects/Diabet_project
python Diabet.py
```

## 📋 Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn (optional)

## 📖 Documentation

- [Tool Box Documentation](./Tool_box/README.md)
- [Projects Documentation](./projects/README.md)

## 🎯 Learning Objectives

This repository helps you learn:
- **Modular code organization** in Python
- **Complete ML pipeline construction** from data to deployment
- **Advanced data preprocessing** techniques
- **Model selection and evaluation** strategies
- **Hyperparameter tuning** and optimization
- **Feature engineering** and importance analysis
- **Clustering algorithms** and validation
- **Best practices** in ML project structure

## 🔧 Recent Bug Fixes (v3.0)

### Critical Fixes
- **Data leakage prevented**: `prepare_data_for_ml` now splits data before scaling
- **Consistent scaling**: `api_server.py` fits scaler on train data only
- **Stale models fixed**: `train_multiple_models` returns only newly trained models

### Security Fixes
- **SSRF protection**: `/load-from-url` blocks private/loopback IPs
- **CORS restricted**: Default to localhost, configurable via `CORS_ORIGINS` env var
- **Deserialization warning**: `joblib.load` warns about untrusted files

### Other Fixes
- **Clustering compare**: Tests all `n_clusters` values, reports best silhouette score
- **Scaler consistency**: All scaling methods store scaler in `self.scaler`

## 🤝 Contributing

1. Follow Python naming conventions (snake_case)
2. Add comprehensive docstrings with type hints
3. Include proper error handling
4. Test implementations thoroughly
5. Update documentation for all languages (EN/FA/DE)

## 📄 License

This toolbox is provided as-is for educational and research purposes.

---

## فارسی (Persian)

# ریپازتوری TOOL-BOX

یک جعبه ابزار جامع یادگیری ماشین و علوم داده با **8 ابزار تخصصی ماژولار**، سرور API، رابط کاربری وب و پروژه‌های نمونه.

## 📁 ساختار ریپازتوری

```
TOOL-BOX/
├── Tool_box/                         # بسته ابزارهای ماژولار ML
│   ├── __init__.py                   # مقداردهی اولیه بسته
│   ├── data_processing_tool.py       # پاکسازی و پیش‌پردازش داده
│   ├── classification_tool.py        # 15 الگوریتم دسته‌بندی
│   ├── regression_tool.py            # 16 الگوریتم رگرسیون
│   ├── model_evaluation_tool.py      # ارزیابی و مقایسه مدل‌ها
│   ├── cross_validation_tool.py      # 6 تکنیک اعتبارسنجی متقابل
│   ├── clustering_tool.py            # 9 الگوریتم خوشه‌بندی
│   ├── optimizer.py                  # بهینه‌سازی ابرپارامترها
│   ├── feature_selector.py           # روش‌های انتخاب ویژگی
│   ├── model_interpreter.py          # ابزارهای تفسیر مدل
│   └── decorators.py                 # دکوراتورهای کمکی
├── projects/                         # پروژه‌های نمونه
│   ├── Diabet_project/               # پیش‌بینی دیابت
│   ├── Heartrate_project/            # پیش‌بینی ضربان قلب
│   └── Housing_project/              # پیش‌بینی قیمت مسکن
├── interface/                        # رابط‌های کاربری وب
│   ├── html/                         # رابط HTML (7 مرحله)
│   ├── html_new/                     # رابط HTML جدید
│   └── streamlit/                    # اپلیکیشن Streamlit
├── api_server.py                     # سرور REST با FastAPI
├── run_api_server.py                 # اجرای سرور
├── run_pipeline_demo.py              # دموی پایپلاین
└── requirements.txt                  # وابستگی‌های پایتون
```

## 🛠️ ابزارهای موجود

### ابزارهای اصلی ML
- **ابزار پردازش داده**: پاکسازی جامع، پیش‌پردازش و تحلیل اکتشافی داده
- **ابزار دسته‌بندی**: 15 الگوریتم (لجستیک، RF، SVM، GB، KNN، NB، درخت تصمیم، AdaBoost، Extra Trees، MLP، XGBoost، LightGBM، CatBoost، QDA، Ridge)
- **ابزار رگرسیون**: 16 الگوریتم (خطی، Ridge، Lasso، ElasticNet، Huber، RF، SVR، GB، HistGradientBoosting، KNN، درخت تصمیم، Extra Trees، MLP، XGBoost، LightGBM، CatBoost)
- **ابزار ارزیابی مدل**: معیارهای جامع ارزیابی و نمایش گرافیکی

### ابزارهای پیشرفته ML
- **ابزار اعتبارسنجی متقابل**: 6 تکنیک (K-Fold، Stratified، Time Series، Repeated، Leave-One-Out، Shuffle Split)
- **ابزار خوشه‌بندی**: 9 الگوریتم (K-Means، Mini-Batch K-Means، DBSCAN، HDBSCAN، Hierarchical، Spectral، BIRCH، OPTICS، Gaussian Mixture)
- **اپتیمایزر**: بهینه‌سازی ابرپارامترها با Grid و Random Search
- **انتخاب‌گر ویژگی**: روش‌های انتخاب ویژگی
- **تفسیرگر مدل**: تفسیر و توضیح‌پذیری مدل

### API و رابط کاربری
- **سرور API**: اندپوینت‌های REST با FastAPI
- **رابط کاربری وب**: رابط HTML با گردش کار 7 مرحله‌ای
- **اپلیکیشن Streamlit**: رابط تعاملی Streamlit

## 📊 پروژه‌های نمونه

### دسته‌بندی
- **پیش‌بینی دیابت**: دسته‌بندی شبکه عصبی با استفاده از داده‌های پزشکی

### رگرسیون
- **پیش‌بینی نرخ قلب**: مقایسه الگوریتم‌های چندگانه برای داده‌های فیزیولوژیکی
- **پیش‌بینی قیمت مسکن**: پیش‌بینی قیمت مسکن کالیفرنیا با مهندسی ویژگی

## 🚀 شروع سریع

### گردش کار کامل ML
```python
from Tool_box import (
    DataProcessingTool,      # آماده‌سازی داده
    ClassificationTool,      # مدل‌های دسته‌بندی چندگانه
    ModelEvaluationTool,     # ارزیابی و مقایسه
    CrossValidationTool,     # اعتبارسنجی پیشرفته
)

# پردازش داده
processor = DataProcessingTool()
data = processor.load_data("data.csv")
processed_data = processor.prepare_data_for_ml(data, target_column="target")

# آموزش و ارزیابی مدل
classifier = ClassificationTool()
models = classifier.train_multiple_models(processed_data['X_train'], processed_data['y_train'])

evaluator = ModelEvaluationTool()
results = evaluator.evaluate_classification_models(models, processed_data['X_test'], processed_data['y_test'])
evaluator.plot_model_comparison(results)
```

### اجرای پروژه‌ها
```bash
cd projects/Diabet_project
python Diabet.py
```

## 📋 پیش‌نیازها

- Python 3.7+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn (اختیاری)

## 📖 مستندات

- [مستندات Tool Box](./Tool_box/README.md)
- [مستندات پروژه‌ها](./projects/README.md)

## 🎯 اهداف یادگیری

این مخزن به شما کمک می‌کند یاد بگیرید:
- سازماندهی مدولار کد در Python
- ساخت پایپلاین یادگیری ماشین
- تکنیک‌های پیش‌پردازش داده
- انتخاب و ارزیابی مدل
- بهترین روش‌ها در ساختار پروژه ML

## 🤝 مشارکت

1. پیروی از قراردادهای نام‌گذاری Python (snake_case)
2. اضافه کردن docstring‌های جامع
3. شامل مدیریت خطا
4. تست کامل پیاده‌سازی‌ها
5. به‌روزرسانی مستندات

## 📄 مجوز

این جعبه ابزار به صورت as-is برای اهداف آموزشی و پژوهشی ارائه شده است.

---

## Deutsch (German)

# TOOL-BOX Repository

Eine umfassende Toolbox für maschinelles Lernen und Datenwissenschaft mit **8 spezialisierten modularen Tools**, API-Server, Weboberfläche und Beispielprojekten.

## 📁 Repository-Struktur

```
TOOL-BOX/
├── Tool_box/                         # Modulares ML-Tools-Paket
│   ├── __init__.py                   # Paket-Initialisierung
│   ├── data_processing_tool.py       # Datenbereinigung & -vorverarbeitung
│   ├── classification_tool.py        # 15 Klassifikationsalgorithmen
│   ├── regression_tool.py            # 16 Regressionsalgorithmen
│   ├── model_evaluation_tool.py      # Modellbewertung & -vergleich
│   ├── cross_validation_tool.py      # 6 Kreuzvalidierungstechniken
│   ├── clustering_tool.py            # 9 Clustering-Algorithmen
│   ├── optimizer.py                  # Hyperparameter-Optimierung
│   ├── feature_selector.py           # Feature-Auswahlmethoden
│   ├── model_interpreter.py          # Modell-Interpretationswerkzeuge
│   └── decorators.py                 # Hilfsdekorateure
├── projects/                         # Beispielprojekte
│   ├── Diabet_project/               # Diabetes-Vorhersage
│   ├── Heartrate_project/            # Herzfrequenz-Vorhersage
│   └── Housing_project/              # Wohnungspreis-Vorhersage
├── interface/                        # Weboberflächen
│   ├── html/                         # HTML-Frontend (7 Schritte)
│   ├── html_new/                     # Neues HTML-Interface
│   └── streamlit/                    # Streamlit-App
├── api_server.py                     # FastAPI REST-Server
├── run_api_server.py                 # Server-Startskript
├── run_pipeline_demo.py              # Pipeline-Demo
└── requirements.txt                  # Python-Abhängigkeiten
```

## 🛠️ Verfügbare Tools

### Kern-ML-Tools
- **Datenverarbeitungs-Tool**: Umfassende Datenbereinigung, -vorverarbeitung und EDA
- **Klassifikations-Tool**: 15 Algorithmen (Logistic, RF, SVM, GB, KNN, NB, Decision Tree, AdaBoost, Extra Trees, MLP, XGBoost, LightGBM, CatBoost, QDA, Ridge)
- **Regressions-Tool**: 16 Algorithmen (Linear, Ridge, Lasso, ElasticNet, Huber, RF, SVR, GB, HistGradientBoosting, KNN, Decision Tree, Extra Trees, MLP, XGBoost, LightGBM, CatBoost)
- **Modellbewertungs-Tool**: Umfassende Bewertungsmetriken und Visualisierung

### Erweiterte ML-Tools
- **Kreuzvalidierungs-Tool**: 6 Techniken (K-Fold, Stratified, Time Series, Repeated, Leave-One-Out, Shuffle Split)
- **Clustering-Tool**: 9 Algorithmen (K-Means, Mini-Batch K-Means, DBSCAN, HDBSCAN, Hierarchical, Spectral, BIRCH, OPTICS, Gaussian Mixture)
- **Optimierer**: Hyperparameter-Optimierung mit Grid- und Random Search
- **Feature-Auswahl**: Feature-Auswahlmethoden
- **Modell-Interpreter**: Modellinterpretation und -erklärbarkeit

### API & Oberfläche
- **API-Server**: REST-Endpunkte mit FastAPI
- **Weboberfläche**: HTML-Frontend mit 7-Schritt-Workflow
- **Streamlit-App**: Interaktive Streamlit-Oberfläche

## 📊 Beispielprojekte

### Klassifikation
- **Diabetes-Vorhersage**: Neuronale Netzwerk-Klassifikation mit medizinischen Daten

### Regression
- **Herzfrequenz-Vorhersage**: Multi-Algorithmus-Vergleich für physiologische Daten
- **Wohnungspreis-Vorhersage**: Kalifornien Wohnungspreis-Vorhersage mit Feature Engineering

## 🚀 Schnellstart

### Kompletter ML-Workflow
```python
from Tool_box import (
    DataProcessingTool,      # Datenaufbereitung
    ClassificationTool,      # Mehrfache Klassifikationsmodelle
    ModelEvaluationTool,     # Bewertung & Vergleich
    CrossValidationTool,     # Erweiterte Validierung
)

# Datenverarbeitung
processor = DataProcessingTool()
data = processor.load_data("data.csv")
processed_data = processor.prepare_data_for_ml(data, target_column="target")

# Modelltraining & Bewertung
classifier = ClassificationTool()
models = classifier.train_multiple_models(processed_data['X_train'], processed_data['y_train'])

evaluator = ModelEvaluationTool()
results = evaluator.evaluate_classification_models(models, processed_data['X_test'], processed_data['y_test'])
evaluator.plot_model_comparison(results)
```

### Projekte ausführen
```bash
cd projects/Diabet_project
python Diabet.py
```

## 📋 Anforderungen

- Python 3.7+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn (optional)

## 📖 Dokumentation

- [Tool Box Dokumentation](./Tool_box/README.md)
- [Projekt-Dokumentation](./projects/README.md)

## 🎯 Lernziele

Dieses Repository hilft Ihnen zu lernen:
- Modulare Code-Organisation in Python
- ML-Pipeline-Konstruktion
- Datenverarbeitungstechniken
- Modellauswahl und -bewertung
- Best Practices in ML-Projektstruktur

## 🤝 Mitwirkung

1. Python Namenskonventionen folgen (snake_case)
2. Umfassende Docstrings hinzufügen
3. Fehlerbehandlung einschließen
4. Implementierungen gründlich testen
5. Dokumentation aktualisieren

## 📄 Lizenz

Diese Toolbox wird as-is für Bildungs- und Forschungszwecke bereitgestellt.
