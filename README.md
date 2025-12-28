# TOOL-BOX Repository

A comprehensive machine learning and data science toolbox with **8 specialized modular tools** and example projects.

## 📁 Repository Structure

```
TOOL-BOX/
├── Tool_box/                         # Modular ML tools package
│   ├── __init__.py                   # Package initialization
│   ├── README.md                     # Tool documentation
│   ├── data_processing_tool.py       #  Data cleaning & preprocessing
│   ├── classification_tool.py        #  7 classification algorithms
│   ├── regression_tool.py            #  8 regression algorithms
│   ├── model_evaluation_tool.py      #  Model evaluation & comparison
│   ├── cross_validation_tool.py      #  6 cross-validation techniques
│   ├── hyperparameter_tuning_tool.py #  Grid & random search tuning
│   ├── feature_importance_tool.py    #  Multi-method importance analysis
│   └── clustering_tool.py            #  9 clustering algorithms
├── projects/                   #  Example projects
│   ├── README.md               #  Projects documentation
│   ├── Diabet_project/         # 🩺 Diabetes prediction
│   ├── Heartrate_project/      # ❤️ Heart rate prediction
│   └── Housing_project/        # 🏠 Housing price prediction
├── reposetori.md               # ℹ️ Repository information
└── README.md                   # 🌍 Main README (EN/FA/DE)
```

## 🛠️ Available Tools

### Core ML Tools
- **Data Processing Tool**: Comprehensive data cleaning, preprocessing, and EDA
- **Classification Tool**: 7 algorithms (Logistic, RF, SVM, GB, KNN, NB, Decision Tree)
- **Regression Tool**: 8 algorithms (Linear, Ridge, Lasso, RF, SVM, GB, KNN, Decision Tree)
- **Model Evaluation Tool**: Comprehensive evaluation metrics and visualization

### Advanced ML Tools
- **Cross Validation Tool**: 6 techniques (K-Fold, Stratified, Time Series, Repeated, Leave-One-Out, Shuffle Split)
- **Hyperparameter Tuning Tool**: Grid search and randomized search optimization
- **Feature Importance Tool**: Multi-method importance analysis (tree, linear, permutation, univariate)
- **Clustering Tool**: 9 algorithms (K-Means, DBSCAN, Hierarchical, Spectral, BIRCH, OPTICS, Mean Shift, Affinity Propagation, Gaussian Mixture)

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

#ریپازتوری TOOL-BOX

یک جعبه ابزار جامع یادگیری ماشین و علوم داده با **8 ابزار تخصصی ماژولار** و پروژه های نمونه.

## 📁 ساختار ریپازتوری

```
TOOL-BOX/
├── Tool_box/                         # بسته ابزارهای ماژولار ML
│   ├── __init__.py                   # مقداردهی اولیه بسته
│   ├── README.md                     # مستندات ابزارها
│   ├── data_processing_tool.py       # پاکسازی و پیش‌پردازش داده
│   ├── classification_tool.py        # 7 الگوریتم دسته‌بندی
│   ├── regression_tool.py            # 8 الگوریتم رگرسیون
│   ├── model_evaluation_tool.py      # ارزیابی و مقایسه مدل‌ها
│   ├── cross_validation_tool.py      # 6 تکنیک اعتبارسنجی متقابل
│   ├── hyperparameter_tuning_tool.py # تنظیم پارامترها
│   ├── feature_importance_tool.py    # تحلیل اهمیت ویژگی‌ها
│   └── clustering_tool.py            # 9 الگوریتم خوشه‌بندی
├── projects/                   #  پروژه های نمونه
│   ├── README.md               # 📖 مستندات پروژه ها
│   ├── Diabet_project/         # 🩺 پیش‌بینی دیابت
│   ├── Heartrate_project/      # ❤️ پیش‌بینی ضربان قلب
│   └── Housing_project/        # 🏠 پیش‌بینی قیمت مسکن
├── reposetori.md               # ℹ️ اطلاعات مخزن
└── README.md                   # 🌍 README اصلی (EN/FA/DE)
```

## 🛠️ ابزارهای موجود

### ابزارهای اصلی ML
- **ابزار پردازش داده**: پاکسازی جامع، پیش‌پردازش و تحلیل اکتشافی داده
- **ابزار دسته‌بندی**: 7 الگوریتم (لجستیک، RF، SVM، GB، KNN، NB، درخت تصمیم)
- **ابزار رگرسیون**: 8 الگوریتم (خطی، Ridge، Lasso، RF، SVM، GB، KNN، درخت تصمیم)
- **ابزار ارزیابی مدل**: معیارهای جامع ارزیابی و نمایش گرافیکی

### ابزارهای پیشرفته ML
- **ابزار اعتبارسنجی متقابل**: 6 تکنیک (K-Fold، Stratified، Time Series، Repeated، Leave-One-Out، Shuffle Split)
- **ابزار تنظیم پارامترها**: جستجوی شبکه و تصادفی برای بهینه‌سازی
- **ابزار اهمیت ویژگی**: تحلیل چند روشه اهمیت (درخت، خطی، پرموتاسیون، تک متغیره)
- **ابزار خوشه‌بندی**: 9 الگوریتم (K-Means، DBSCAN، Hierarchical، Spectral، BIRCH، OPTICS، Mean Shift، Affinity Propagation، Gaussian Mixture)

## 📊 پروژه های نمونه

### دسته‌بندی
- **پیش‌بینی دیابت**: دسته‌بندی شبکه عصبی با استفاده از داده های پزشکی

### رگرسیون
- **پیش‌بینی نرخ قلب**: مقایسه الگوریتم های چندگانه برای داده های فیزیولوژیکی
- **پیش‌بینی قیمت مسکن**: پیش‌بینی قیمت مسکن کالیفرنیا با مهندسی ویژگی

## 🚀 شروع سریع

### استفاده از ابزارهای جداگانه
```python
from Tool_box import data_cleaner_tool, regression_tool

# پاکسازی داده های شما
cleaned_data = data_cleaner_tool.clean_data(your_dataframe)

# اجرای تحلیل رگرسیون
# (دنبال کردن پایپلاین تعاملی در regression_tool.py)
```

### اجرای پروژه ها
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
- [مستندات پروژه ها](./projects/README.md)

## 🎯 اهداف یادگیری

این مخزن به شما کمک می‌کند یاد بگیرید:
- سازماندهی مدولار کد در Python
- ساخت پایپلاین یادگیری ماشین
- تکنیک‌های پیش‌پردازش داده
- انتخاب و ارزیابی مدل
- بهترین روش‌ها در ساختار پروژه ML

## 🤝 مشارکت

1. پیروی از قراردادهای نام‌گذاری Python (snake_case)
2. اضافه کردن docstring های جامع
3. شامل مدیریت خطا
4. تست کامل پیاده‌سازی ها
5. به‌روزرسانی مستندات

## 📄 مجوز

این جعبه ابزار به صورت as-is برای اهداف آموزشی و پژوهشی ارائه شده است.

---

## Deutsch (German)

# TOOL-BOX Repository

Eine umfassende Toolbox für maschinelles Lernen und Datenwissenschaft mit **8 spezialisierten modularen Tools** und Beispielprojekten.

## 📁 Repository-Struktur

```
TOOL-BOX/
├── Tool_box/                         # Modulares ML-Tools-Paket
│   ├── __init__.py                   # Paket-Initialisierung
│   ├── README.md                     # Tool-Dokumentation
│   ├── data_processing_tool.py       # Datenbereinigung & -vorverarbeitung
│   ├── classification_tool.py        # 7 Klassifikationsalgorithmen
│   ├── regression_tool.py            # 8 Regressionsalgorithmen
│   ├── model_evaluation_tool.py      # Modellbewertung & -vergleich
│   ├── cross_validation_tool.py      # 6 Kreuzvalidierungstechniken
│   ├── hyperparameter_tuning_tool.py # Grid- & Random-Search-Tuning
│   ├── feature_importance_tool.py    # Mehrfachmethoden-Importance-Analyse
│   └── clustering_tool.py            # 9 Clustering-Algorithmen
├── projects/                   # Beispielprojekte
│   ├── README.md               # 📖 Projektdokumentation
│   ├── Diabet_project/         # 🩺 Diabetes-Vorhersage
│   ├── Heartrate_project/      # ❤️ Herzfrequenz-Vorhersage
│   └── Housing_project/        # 🏠 Wohnungs-Preis-Vorhersage
├── reposetori.md               # ℹ️ Repository-Informationen
└── README.md                   # 🌍 Haupt-README (EN/FA/DE)
```

## 🛠️ Verfügbare Tools

### Kern-ML-Tools
- **Datenverarbeitungs-Tool**: Umfassende Datenbereinigung, -vorverarbeitung und EDA
- **Klassifikations-Tool**: 7 Algorithmen (Logistic, RF, SVM, GB, KNN, NB, Decision Tree)
- **Regressions-Tool**: 8 Algorithmen (Linear, Ridge, Lasso, RF, SVM, GB, KNN, Decision Tree)
- **Modellbewertungs-Tool**: Umfassende Bewertungsmetriken und Visualisierung

### Erweiterte ML-Tools
- **Kreuzvalidierungs-Tool**: 6 Techniken (K-Fold, Stratified, Time Series, Repeated, Leave-One-Out, Shuffle Split)
- **Hyperparameter-Tuning-Tool**: Grid-Search und Randomized-Search-Optimierung
- **Feature-Importance-Tool**: Mehrfachmethoden-Importance-Analyse (Baum, linear, Permutation, univariat)
- **Clustering-Tool**: 9 Algorithmen (K-Means, DBSCAN, Hierarchical, Spectral, BIRCH, OPTICS, Mean Shift, Affinity Propagation, Gaussian Mixture)

## 📊 Beispielprojekte

### Klassifikation
- **Diabetes-Vorhersage**: Neuronale Netzwerk-Klassifikation mit medizinischen Daten

### Regression
- **Herzfrequenz-Vorhersage**: Multi-Algorithmus-Vergleich für physiologische Daten
- **Wohnungspreis-Vorhersage**: Kalifornien Wohnungspreis-Vorhersage mit Feature Engineering

## 🚀 Schnellstart

### Einzelne Tools verwenden
```python
from Tool_box import data_cleaner_tool, regression_tool

# Ihre Daten bereinigen
cleaned_data = data_cleaner_tool.clean_data(your_dataframe)

# Regressionsanalyse ausführen
# (Interaktive Pipeline in regression_tool.py folgen)
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
