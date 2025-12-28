# Projects Documentation / مستندات پروژه ها / Projektdokumentation

This directory contains example projects demonstrating the usage of various machine learning and data science techniques using the Tool Box.

این دایرکتوری شامل پروژه های نمونه ای است که استفاده از تکنیک های مختلف یادگیری ماشین و علوم داده با استفاده از Tool Box را نشان می دهد.

Dieses Verzeichnis enthält Beispielprojekte, die die Verwendung verschiedener Machine-Learning- und Data-Science-Techniken mit der Tool Box demonstrieren.

## Available Projects / پروژه های موجود / Verfügbare Projekte

### 1. Diabetes Prediction Project (`Diabet_project/`) / پروژه پیش بینی دیابت / Diabetes-Vorhersage-Projekt
**Purpose**: Classification model to predict diabetes based on medical diagnostic measurements

**هدف**: مدل دسته‌بندی برای پیش‌بینی دیابت بر اساس اندازه‌گیری‌های تشخیصی پزشکی

**Zweck**: Klassifikationsmodell zur Vorhersage von Diabetes basierend auf medizinischen Diagnosemessungen

**Dataset**: diabetes.csv (Pima Indians Diabetes Database)
- Features: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
- Target: Outcome (0 = No diabetes, 1 = Diabetes)

**مجموعه داده**: diabetes.csv (پایگاه داده دیابت هندیان پیما)
- ویژگی‌ها: بارداری، گلوکز، فشار خون، ضخامت پوست، انسولین، BMI، تابع نسب دیابت، سن
- هدف: نتیجه (0 = دیابت ندارد، 1 = دیابت دارد)

**Datensatz**: diabetes.csv (Pima Indians Diabetes Database)
- Merkmale: Schwangerschaften, Glukose, Blutdruck, Hautdicke, Insulin, BMI, Diabetes-Vererbungsfunktion, Alter
- Ziel: Ergebnis (0 = kein Diabetes, 1 = Diabetes)

**Implementation** (`Diabet.py`):
- Uses MLPClassifier (Neural Network) from scikit-learn
- Includes PCA for dimensionality reduction
- Standardizes features before PCA
- Evaluates model with accuracy, precision, and recall

**پیاده‌سازی** (`Diabet.py`):
- از MLPClassifier (شبکه عصبی) از scikit-learn استفاده می‌کند
- شامل PCA برای کاهش ابعاد
- ویژگی‌ها را قبل از PCA استاندارد می‌کند
- مدل را با دقت، دقت مثبت و فراخوانی ارزیابی می‌کند

**Implementierung** (`Diabet.py`):
- Verwendet MLPClassifier (Neural Network) von scikit-learn
- Enthält PCA zur Dimensionsreduktion
- Standardisiert Features vor PCA
- Evaluiert Modell mit Genauigkeit, Präzision und Recall

**Key Features**:
- Neural network classification
- Feature scaling and PCA
- Performance metrics calculation

**ویژگی‌های کلیدی**:
- دسته‌بندی شبکه عصبی
- مقیاس‌بندی ویژگی و PCA
- محاسبه معیارهای عملکرد

**Wichtige Merkmale**:
- Neuronale Netzwerk-Klassifikation
- Feature-Skalierung und PCA
- Leistungsmetriken-Berechnung

### 2. Heart Rate Prediction Project (`Heartrate_project/`) / پروژه پیش‌بینی ضربان قلب / Herzfrequenz-Vorhersage-Projekt
**Purpose**: Regression model to predict heart rate based on various physiological measurements

**هدف**: مدل رگرسیون برای پیش‌بینی ضربان قلب بر اساس اندازه‌گیری‌های فیزیولوژیکی مختلف

**Zweck**: Regressionsmodell zur Vorhersage der Herzfrequenz basierend auf verschiedenen physiologischen Messungen

**Dataset**: heart.csv (Heart rate dataset)
- Multiple physiological features
- Target: Heart rate values

**مجموعه داده**: heart.csv (مجموعه داده ضربان قلب)
- چندین ویژگی فیزیولوژیکی
- هدف: مقادیر ضربان قلب

**Datensatz**: heart.csv (Herzfrequenz-Datensatz)
- Mehrere physiologische Merkmale
- Ziel: Herzfrequenz-Werte

**Implementation** (`Heartrate.py`):
- Compares multiple regression algorithms:
  - Random Forest Regressor
  - Gradient Boosting Regressor
  - Support Vector Regressor (SVR)
  - K-Nearest Neighbors Regressor
- Uses 10-fold cross-validation for robust evaluation
- Standardizes features and handles missing values

**پیاده‌سازی** (`Heartrate.py`):
- چندین الگوریتم رگرسیون را مقایسه می‌کند:
  - رگرسور جنگل تصادفی
  - رگرسور گرادیان بوستینگ
  - رگرسور بردار پشتیبانی (SVR)
  - رگرسور K-نزدیک‌ترین همسایگان
- از اعتبارسنجی متقابل 10-fold برای ارزیابی قوی استفاده می‌کند
- ویژگی‌ها را استاندارد کرده و مقادیر مفقود را مدیریت می‌کند

**Implementierung** (`Heartrate.py`):
- Vergleicht mehrere Regressionsalgorithmen:
  - Random Forest Regressor
  - Gradient Boosting Regressor
  - Support Vector Regressor (SVR)
  - K-Nearest Neighbors Regressor
- Verwendet 10-fache Kreuzvalidierung für robuste Evaluation
- Standardisiert Features und behandelt fehlende Werte

**Evaluation Metrics**:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R-squared (R²) coefficient

**معیارهای ارزیابی**:
- میانگین مربعات خطا (MSE)
- جذر میانگین مربعات خطا (RMSE)
- ضریب R-squared (R²)

**Evaluationsmetriken**:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R-squared (R²) Koeffizient

### 3. Housing Price Prediction Project (`Housing_project/`) / پروژه پیش‌بینی قیمت مسکن / Immobilienpreis-Vorhersage-Projekt
**Purpose**: Comprehensive regression analysis for California housing prices

**هدف**: تحلیل رگرسیون جامع برای قیمت مسکن کالیفرنیا

**Zweck**: Umfassende Regressionsanalyse für kalifornische Immobilienpreise

**Dataset**: housing.csv (California Housing dataset)
- Features: longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income, ocean_proximity
- Target: median_house_value

**مجموعه داده**: housing.csv (مجموعه داده مسکن کالیفرنیا)
- ویژگی‌ها: طول جغرافیایی، عرض جغرافیایی، میانگین سن مسکن، کل اتاق‌ها، کل اتاق‌های خواب، جمعیت، خانوارها، میانگین درآمد، مجاورت با اقیانوس
- هدف: میانگین ارزش مسکن

**Datensatz**: housing.csv (Kalifornien-Immobilien-Datensatz)
- Merkmale: Längengrad, Breitengrad, medianes Hausalter, Gesamtzimmer, Gesamtschlafzimmer, Bevölkerung, Haushalte, medianes Einkommen, Ozeannähe
- Ziel: medianer Hauswert

**Implementation** (`Housing.py`):
- Complete machine learning pipeline:
  - Data exploration and visualization
  - Feature engineering (derived features)
  - Multiple regression algorithms
  - Hyperparameter tuning with GridSearchCV

**پیاده‌سازی** (`Housing.py`):
- پایپلاین کامل یادگیری ماشین:
  - کاوش داده‌ها و نمایش گرافیکی
  - مهندسی ویژگی (ویژگی‌های مشتق شده)
  - چندین الگوریتم رگرسیون
  - تنظیم پارامترها با GridSearchCV

**Implementierung** (`Housing.py`):
- Vollständige Machine-Learning-Pipeline:
  - Datenexploration und Visualisierung
  - Feature-Engineering (abgeleitete Merkmale)
  - Mehrere Regressionsalgorithmen
  - Hyperparameter-Tuning mit GridSearchCV

**Algorithms Used**:
- Linear Regression
- Decision Tree Regression
- Random Forest Regression
- Pipeline with preprocessing steps

**الگوریتم‌های استفاده شده**:
- رگرسیون خطی
- رگرسیون درخت تصمیم
- رگرسیون جنگل تصادفی
- پایپلاین با مراحل پیش‌پردازش

**Verwendete Algorithmen**:
- Lineare Regression
- Entscheidungsbaum-Regression
- Random Forest Regression
- Pipeline mit Vorverarbeitungsschritten

**Key Features**:
- Correlation analysis with visualization
- Custom feature engineering
- Cross-validation evaluation
- Hyperparameter optimization

**ویژگی‌های کلیدی**:
- تحلیل همبستگی با نمایش گرافیکی
- مهندسی ویژگی سفارشی
- ارزیابی اعتبارسنجی متقابل
- بهینه‌سازی پارامترها

**Wichtige Merkmale**:
- Korrelationsanalyse mit Visualisierung
- Benutzerdefiniertes Feature-Engineering
- Kreuzvalidierungs-Evaluation
- Hyperparameter-Optimierung

## Project Structure / ساختار پروژه / Projektstruktur

Each project follows this structure:
```
Project_Name/
├── Project_Name.py    # Main implementation file
├── Project_Name.md    # Project documentation (if available)
├── Dataset/          # Data files
│   └── dataset.csv   # Raw data
└── README.md         # Project-specific documentation
```

هر پروژه از این ساختار پیروی می‌کند:
```
Project_Name/
├── Project_Name.py    # فایل پیاده‌سازی اصلی
├── Project_Name.md    # مستندات پروژه (در صورت موجود بودن)
├── Dataset/          # فایل‌های داده
│   └── dataset.csv   # داده خام
└── README.md         # مستندات خاص پروژه
```

Jedes Projekt folgt dieser Struktur:
```
Project_Name/
├── Project_Name.py    # Hauptimplementierungsdatei
├── Project_Name.md    # Projektdokumentation (falls verfügbar)
├── Dataset/          # Datendateien
│   └── dataset.csv   # Rohdaten
└── README.md         # Projektspezifische Dokumentation
```

## Running the Projects / اجرای پروژه‌ها / Projekte ausführen

### Prerequisites / پیش‌نیازها / Voraussetzungen
- Python 3.x
- Required libraries: pandas, numpy, scikit-learn, matplotlib
- Dataset files in appropriate directories

### Execution / اجرا / Ausführung
1. Navigate to the project directory
2. Run the Python script: `python Project_Name.py`
3. Check console output for results and metrics

## Learning Objectives / اهداف یادگیری / Lernziele

These projects demonstrate:
- **Data Preprocessing**: Cleaning, scaling, encoding
- **Feature Engineering**: Creating meaningful features
- **Model Selection**: Comparing different algorithms
- **Model Evaluation**: Using appropriate metrics
- **Pipeline Construction**: End-to-end ML workflows

این پروژه‌ها نشان می‌دهند:
- **پیش‌پردازش داده**: پاکسازی، مقیاس‌بندی، کدگذاری
- **مهندسی ویژگی**: ایجاد ویژگی‌های معنادار
- **انتخاب مدل**: مقایسه الگوریتم‌های مختلف
- **ارزیابی مدل**: استفاده از معیارهای مناسب
- **ساخت پایپلاین**: گردش کار end-to-end ML

Diese Projekte demonstrieren:
- **Daten-Vorverarbeitung**: Bereinigung, Skalierung, Kodierung
- **Feature-Engineering**: Erstellung sinnvoller Merkmale
- **Modellauswahl**: Vergleich verschiedener Algorithmen
- **Modellevaluation**: Verwendung geeigneter Metriken
- **Pipeline-Konstruktion**: End-to-End-ML-Workflows

## Tool Box Integration / یکپارچگی Tool Box / Tool Box Integration

Each project showcases different tools from the Tool Box:
- `data_cleaner_tool.py`: Data preprocessing
- `regression_tool.py`: Complete regression pipelines
- `regression_selector_tool.py`: Model comparison
- `k_fold_algorithm_tool.py`: Cross-validation
- `reading_csv_and_plotting_corr_tool.py`: EDA and visualization

هر پروژه ابزارهای مختلف Tool Box را نمایش می‌دهد:
- `data_cleaner_tool.py`: پیش‌پردازش داده
- `regression_tool.py`: پایپلاین‌های رگرسیون کامل
- `regression_selector_tool.py`: مقایسه مدل
- `k_fold_algorithm_tool.py`: اعتبارسنجی متقابل
- `reading_csv_and_plotting_corr_tool.py`: EDA و نمایش گرافیکی

Jedes Projekt zeigt verschiedene Tools der Tool Box:
- `data_cleaner_tool.py`: Daten-Vorverarbeitung
- `regression_tool.py`: Vollständige Regressions-Pipelines
- `regression_selector_tool.py`: Modellvergleich
- `k_fold_algorithm_tool.py`: Kreuzvalidierung
- `reading_csv_and_plotting_corr_tool.py`: EDA und Visualisierung

## Contributing / مشارکت / Mitwirkung

To add new projects:
1. Create a new directory with project name
2. Include main Python script and dataset
3. Add comprehensive documentation
4. Follow the established naming conventions
5. Test the implementation thoroughly

برای اضافه کردن پروژه‌های جدید:
1. یک دایرکتوری جدید با نام پروژه ایجاد کنید
2. اسکریپت پایتون اصلی و مجموعه داده را وارد کنید
3. مستندات جامع اضافه کنید
4. از قراردادهای نام‌گذاری تعیین شده پیروی کنید
5. پیاده‌سازی را به طور کامل تست کنید

Um neue Projekte hinzuzufügen:
1. Neues Verzeichnis mit Projektname erstellen
2. Haupt-Python-Skript und Datensatz einbeziehen
3. Umfassende Dokumentation hinzufügen
4. Etablierte Namenskonventionen befolgen
5. Implementierung gründlich testen
