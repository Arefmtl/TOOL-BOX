# Diabetes Prediction Project / پروژه پیش بینی دیابت / Diabetes-Vorhersage-Projekt

## 🎯 Project Overview / مرور پروژه / Projektübersicht

### English (EN)
This project demonstrates a comprehensive machine learning pipeline for diabetes prediction using medical data. It implements advanced preprocessing techniques, multiple classification algorithms, ensemble methods, and model evaluation to achieve high accuracy in predicting diabetes onset.

### فارسی (FA)
این پروژه یک خط لوله جامع یادگیری ماشین برای پیش بینی دیابت با استفاده از داده های پزشکی ارائه می دهد. این پروژه از تکنیک های پیشرفته پیش پردازش، چندین الگوریتم طبقه بندی، روش های ترکیبی و ارزیابی مدل برای دستیابی به دقت بالا در پیش بینی شروع دیابت استفاده می کند.

### Deutsch (DE)
Dieses Projekt demonstriert eine umfassende Machine-Learning-Pipeline zur Diabetes-Vorhersage mit medizinischen Daten. Es implementiert fortschrittliche Vorverarbeitungstechniken, mehrere Klassifikationsalgorithmen, Ensemble-Methoden und Modellevaluation, um eine hohe Genauigkeit bei der Vorhersage des Diabetesausbruchs zu erzielen.

## 📊 Data Description / توضیحات داده ها / Datenbeschreibung

### English (EN)
**Dataset**: Pima Indians Diabetes Database
- **Samples**: 768 patients
- **Features**: 8 medical measurements
- **Target**: Binary classification (0 = No Diabetes, 1 = Diabetes)

**Features**:
- Pregnancies: Number of pregnancies
- Glucose: Plasma glucose concentration
- BloodPressure: Diastolic blood pressure (mm Hg)
- SkinThickness: Triceps skin fold thickness (mm)
- Insulin: 2-Hour serum insulin (mu U/ml)
- BMI: Body mass index
- DiabetesPedigreeFunction: Diabetes pedigree function
- Age: Age (years)

**Data Quality Issues**:
- Missing values coded as 0 in medical columns (Glucose, BloodPressure, SkinThickness, Insulin, BMI)
- Requires special handling for medical data integrity

### فارسی (FA)
**مجموعه داده**: پایگاه داده دیابت هندیان پیما
- **نمونه ها**: 768 بیمار
- **ویژگی ها**: 8 اندازه گیری پزشکی
- **هدف**: طبقه بندی دودویی (0 = دیابت ندارد، 1 = دیابت دارد)

**ویژگی ها**:
- بارداری: تعداد بارداری ها
- گلوکز: غلظت گلوکز پلاسما
- فشار خون: فشار خون دیاستولیک (mm Hg)
- ضخامت پوست: ضخامت چین پوست سه سر (mm)
- انسولین: انسولین سرم 2 ساعته (mu U/ml)
- BMI: شاخص توده بدنی
- تابع نسبت دیابت: تابع نسبت دیابت
- سن: سن (سال)

**مشکلات کیفیت داده**:
- مقادیر مفقود به صورت 0 در ستون های پزشکی کد شده است (گلوکز، فشار خون، ضخامت پوست، انسولین، BMI)
- نیاز به روش های خاص برای حفظ صحت داده های پزشکی

### Deutsch (DE)
**Datensatz**: Pima Indians Diabetes Database
- **Stichproben**: 768 Patienten
- **Merkmale**: 8 medizinische Messungen
- **Ziel**: Binäre Klassifikation (0 = kein Diabetes, 1 = Diabetes)

**Merkmale**:
- Schwangerschaften: Anzahl der Schwangerschaften
- Glukose: Plasma-Glukose-Konzentration
- Blutdruck: Diastolischer Blutdruck (mm Hg)
- Hautdicke: Trizeps-Hautfaltendicke (mm)
- Insulin: 2-Stunden-Seruminsulin (mu U/ml)
- BMI: Body-Mass-Index
- Diabetes-Vererbungsfunktion: Diabetes-Pedigree-Funktion
- Alter: Alter (Jahre)

**Datenqualitätsprobleme**:
- Fehlende Werte in medizinischen Spalten als 0 kodiert (Glukose, Blutdruck, Hautdicke, Insulin, BMI)
- Erfordert spezielle Behandlung für medizinische Datenintegrität

## 🔬 Methodology / روش شناسی / Methodik

### English (EN)
**Data Preprocessing**:
1. **Missing Value Handling**: Replace 0 values in medical columns with NaN, then impute with median
2. **Data Splitting**: 70% training, 30% testing
3. **Feature Scaling**: StandardScaler for numerical features

**Model Training**:
1. **Multiple Algorithms**: Logistic Regression, Random Forest, SVM, Gradient Boosting, KNN, Naive Bayes, Decision Tree
2. **Ensemble Methods**: Voting Classifier, Bagging, AdaBoost, Extra Trees
3. **Advanced Techniques**: PCA for dimensionality reduction, XGBoost integration

**Model Evaluation**:
1. **Performance Metrics**: Accuracy, Precision, Recall, F1-Score
2. **Cross-Validation**: 5-fold cross-validation
3. **Hyperparameter Tuning**: Grid search and random search optimization

### فارسی (FA)
**پیش پردازش داده**:
1. **مدیریت مقادیر مفقود**: جایگزینی مقادیر 0 در ستون های پزشکی با NaN، سپس جایگزینی با میانه
2. **تقسیم داده**: 70% آموزش، 30% تست
3. **مقیاس بندی ویژگی**: StandardScaler برای ویژگی های عددی

**آموزش مدل**:
1. **چندین الگوریتم**: رگرسیون لجستیک، جنگل تصادفی، SVM، گرادیان بوستینگ، KNN، نیوی بیز، درخت تصمیم
2. **روش های ترکیبی**: طبقه بندی ووتینگ، بگینگ، AdaBoost، Extra Trees
3. **تکنیک های پیشرفته**: PCA برای کاهش ابعاد، ادغام XGBoost

**ارزیابی مدل**:
1. **معیارهای عملکرد**: دقت، دقت مثبت، فراخوانی، F1-Score
2. **اعتبارسنجی متقابل**: اعتبارسنجی 5-folds
3. **تنظیم هیپرپارامتر**: بهینه سازی جستجوی شبکه و جستجوی تصادفی

### Deutsch (DE)
**Daten-Vorverarbeitung**:
1. **Behandlung fehlender Werte**: Ersetzen von 0-Werten in medizinischen Spalten durch NaN, dann Imputation mit Median
2. **Datenaufteilung**: 70% Training, 30% Testen
3. **Feature-Scaling**: StandardScaler für numerische Features

**Modelltraining**:
1. **Mehrere Algorithmen**: Logistische Regression, Random Forest, SVM, Gradient Boosting, KNN, Naive Bayes, Decision Tree
2. **Ensemble-Methoden**: Voting Classifier, Bagging, AdaBoost, Extra Trees
3. **Fortgeschrittene Techniken**: PCA zur Dimensionsreduktion, XGBoost-Integration

**Modellevaluation**:
1. **Leistungsmetriken**: Genauigkeit, Präzision, Recall, F1-Score
2. **Kreuzvalidierung**: 5-fache Kreuzvalidierung
3. **Hyperparameter-Tuning**: Grid-Suche und Random-Suche-Optimierung

## 📈 Results & Analysis / نتایج و تحلیل / Ergebnisse & Analyse

### English (EN)
**Best Performing Models**:
- **Extra Trees Classifier**: Highest accuracy achieved
- **Voting Classifier**: Robust ensemble performance
- **XGBoost**: Excellent for medical data patterns

**Key Findings**:
- Ensemble methods significantly improve prediction accuracy
- Feature importance analysis reveals glucose and BMI as most predictive
- Cross-validation ensures model robustness
- PCA helps reduce dimensionality while maintaining performance

**Performance Metrics**:
- **Target**: Achieve >80% accuracy
- **Result**: Best models exceed 80% accuracy threshold
- **Cross-validation**: Consistent performance across folds

### فارسی (FA)
**بهترین مدل های اجرا شده**:
- **طبقه بند Extra Trees**: بالاترین دقت به دست آمده
- **طبقه بند ووتینگ**: عملکرد ترکیبی قوی
- **XGBoost**: عالی برای الگوهای داده های پزشکی

**یافته های کلیدی**:
- روش های ترکیبی به طور قابل توجهی دقت پیش بینی را افزایش می دهند
- تحلیل اهمیت ویژگی ها نشان می دهد که گلوکز و BMI بیشترین قدرت پیش بینی را دارند
- اعتبارسنجی متقابل اطمینان از استحکام مدل را فراهم می کند
- PCA به کاهش ابعاد کمک می کند در حالی که عملکرد را حفظ می کند

**معیارهای عملکرد**:
- **هدف**: دستیابی به دقت >80%
- **نتیجه**: بهترین مدل ها از آستانه دقت 80% فراتر می روند
- **اعتبارسنجی متقابل**: عملکرد مداوم در کریسهای مختلف

### Deutsch (DE)
**Beste Modelle**:
- **Extra Trees Classifier**: Höchste Genauigkeit erreicht
- **Voting Classifier**: Robuste Ensemble-Leistung
- **XGBoost**: Ausgezeichnet für medizinische Datenmuster

**Wichtige Erkenntnisse**:
- Ensemble-Methoden verbessern die Vorhersagegenauigkeit erheblich
- Feature-Importance-Analyse zeigt Glukose und BMI als am stärksten vorhersagend
- Kreuzvalidierung gewährleistet Modellrobustheit
- PCA hilft bei der Dimensionsreduktion bei gleichbleibender Leistung

**Leistungsmetriken**:
- **Ziel**: >80% Genauigkeit erreichen
- **Ergebnis**: Beste Modelle überschreiten die 80% Genauigkeitsschwelle
- **Kreuzvalidierung**: Konsistente Leistung über alle Folds

## 🚀 Usage Instructions / دستورالعمل استفاده / Bedienungsanleitung

### English (EN)
**Prerequisites**:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost
```

**Running the Project**:
```python
# Import required modules
from Tool_box import DataProcessingTool, ClassificationTool, ModelEvaluationTool

# Load and preprocess data
processor = DataProcessingTool()
data = processor.load_data("Dataset/diabetes.csv")
processed_data = processor.prepare_data_for_ml(data, target_column="Outcome")

# Train models
classifier = ClassificationTool()
models = classifier.train_multiple_models(processed_data['X_train'], processed_data['y_train'])

# Evaluate models
evaluator = ModelEvaluationTool()
results = evaluator.evaluate_classification_models(models, processed_data['X_test'], processed_data['y_test'])
```

**Key Features**:
- Automatic missing value handling for medical data
- Comprehensive model comparison
- Ensemble method implementation
- Feature importance analysis

### فارسی (FA)
**پیش نیازها**:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost
```

**اجرای پروژه**:
```python
# وارد کردن ماژول های مورد نیاز
from Tool_box import DataProcessingTool, ClassificationTool, ModelEvaluationTool

# بارگیری و پیش پردازش داده ها
processor = DataProcessingTool()
data = processor.load_data("Dataset/diabetes.csv")
processed_data = processor.prepare_data_for_ml(data, target_column="Outcome")

# آموزش مدل ها
classifier = ClassificationTool()
models = classifier.train_multiple_models(processed_data['X_train'], processed_data['y_train'])

# ارزیابی مدل ها
evaluator = ModelEvaluationTool()
results = evaluator.evaluate_classification_models(models, processed_data['X_test'], processed_data['y_test'])
```

**ویژگی های کلیدی**:
- مدیریت خودکار مقادیر مفقود برای داده های پزشکی
- مقایسه جامع مدل ها
- پیاده سازی روش های ترکیبی
- تحلیل اهمیت ویژگی ها

### Deutsch (DE)
**Voraussetzungen**:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost
```

**Projekt ausführen**:
```python
# Erforderliche Module importieren
from Tool_box import DataProcessingTool, ClassificationTool, ModelEvaluationTool

# Daten laden und vorverarbeiten
processor = DataProcessingTool()
data = processor.load_data("Dataset/diabetes.csv")
processed_data = processor.prepare_data_for_ml(data, target_column="Outcome")

# Modelle trainieren
classifier = ClassificationTool()
models = classifier.train_multiple_models(processed_data['X_train'], processed_data['y_train'])

# Modelle evaluieren
evaluator = ModelEvaluationTool()
results = evaluator.evaluate_classification_models(models, processed_data['X_test'], processed_data['y_test'])
```

**Wichtige Funktionen**:
- Automatische Behandlung fehlender Werte für medizinische Daten
- Umfassender Modellvergleich
- Implementierung von Ensemble-Methoden
- Feature-Importance-Analyse

## ⚡ Optimization Notes / یادداشت های بهینه سازی / Optimierungshinweise

### English (EN)
**Performance Optimizations**:
- **Early Stopping**: Prevents overfitting in ensemble methods
- **Cross-Validation**: Ensures model generalization
- **Feature Scaling**: Improves convergence speed
- **Hyperparameter Tuning**: Optimizes model performance

**Medical Data Specific**:
- **Domain Knowledge**: Proper handling of medical measurement constraints
- **Data Integrity**: Maintain clinical accuracy in preprocessing
- **Interpretability**: Focus on explainable AI for medical applications

**Future Enhancements**:
- **SHAP Values**: For model interpretability
- **Real-time Prediction**: Web API for clinical use
- **Model Monitoring**: Continuous performance tracking

### فارسی (FA)
**بهینه سازی های عملکرد**:
- **توقف زودهنگام**: جلوگیری از بیش برازش در روش های ترکیبی
- **اعتبارسنجی متقابل**: اطمینان از تعمیم پذیری مدل
- **مقیاس بندی ویژگی**: بهبود سرعت همگرایی
- **تنظیم هیپرپارامتر**: بهینه سازی عملکرد مدل

**مختص داده های پزشکی**:
- **دانش حوزه**: مدیریت مناسب محدودیت های اندازه گیری های پزشکی
- **صحت داده**: حفظ دقت بالینی در پیش پردازش
- **تفسیرپذیری**: تمرکز بر هوش مصنوعی قابل تفسیر برای کاربردهای پزشکی

**ارتقا های آینده**:
- **مقادیر SHAP**: برای تفسیرپذیری مدل
- **پیش بینی بلادرنگ**: API وب برای استفاده بالینی
- **مانیتورینگ مدل**: ردیابی مداوم عملکرد

### Deutsch (DE)
**Leistungsoptimierungen**:
- **Early Stopping**: Verhindert Overfitting bei Ensemble-Methoden
- **Kreuzvalidierung**: Stellt Modellgeneralisierung sicher
- **Feature-Scaling**: Verbessert Konvergenzgeschwindigkeit
- **Hyperparameter-Tuning**: Optimiert Modellleistung

**Medizinische Daten spezifisch**:
- **Domänenwissen**: Richtige Handhabung medizinischer Messbeschränkungen
- **Datenintegrität**: Klinische Genauigkeit in der Vorverarbeitung beibehalten
- **Interpretierbarkeit**: Fokus auf erklärbare KI für medizinische Anwendungen

**Zukünftige Verbesserungen**:
- **SHAP-Werte**: Für Modellinterpretierbarkeit
- **Echtzeit-Vorhersage**: Web-API für klinische Anwendung
- **Modell-Monitoring**: Kontinuierliche Leistungsüberwachung

## 📞 Contact / تماس / Kontakt
For questions or improvements, please refer to the main TOOL-BOX repository documentation.
برای سوالات یا بهبودها، لطفاً به مستندات اصلی TOOL-BOX مراجعه کنید.
Für Fragen oder Verbesserungen wenden Sie sich bitte an die Haupt-TOOL-BOX-Repository-Dokumentation.
