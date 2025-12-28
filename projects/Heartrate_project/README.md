# Heart Rate Prediction Project / پروژه پیش بینی ضربان قلب / Herzfrequenz-Vorhersage-Projekt

## 🎯 Project Overview / مرور پروژه / Projektübersicht

### English (EN)
This project demonstrates machine learning techniques for predicting heart rate using physiological data. It implements comprehensive data preprocessing, multiple regression algorithms, and advanced feature engineering to model the relationship between various physiological measurements and heart rate.

### فارسی (FA)
این پروژه تکنیک های یادگیری ماشین برای پیش بینی ضربان قلب با استفاده از داده های فیزیولوژیک را نشان می دهد. این پروژه از پیش پردازش جامع داده ها، چندین الگوریتم رگرسیون و مهندسی ویژگی پیشرفته برای مدل کردن رابطه بین اندازه گیری های فیزیولوژیک مختلف و ضربان قلب استفاده می کند.

### Deutsch (DE)
Dieses Projekt demonstriert Machine-Learning-Techniken zur Vorhersage der Herzfrequenz anhand physiologischer Daten. Es implementiert eine umfassende Daten-Vorverarbeitung, mehrere Regressionsalgorithmen und fortgeschrittenes Feature-Engineering, um die Beziehung zwischen verschiedenen physiologischen Messungen und der Herzfrequenz zu modellieren.

## 📊 Data Description / توضیحات داده ها / Datenbeschreibung

### English (EN)
**Dataset**: Physiological Measurements Database
- **Samples**: Variable (depends on data source)
- **Features**: Multiple physiological measurements
- **Target**: Heart rate (beats per minute)

**Key Features**:
- **Age**: Patient age in years
- **Gender**: Male/Female (encoded)
- **Height**: Height in centimeters
- **Weight**: Weight in kilograms
- **Blood Pressure**: Systolic and diastolic measurements
- **Cholesterol Levels**: Total cholesterol, HDL, LDL
- **Smoking Status**: Binary indicator
- **Physical Activity**: Exercise frequency/duration
- **Stress Level**: Self-reported stress measurement
- **Sleep Quality**: Sleep duration and quality metrics

**Data Characteristics**:
- **Physiological Constraints**: Values must be within realistic medical ranges
- **Correlation Patterns**: Strong relationships between related measurements
- **Missing Data**: Common in self-reported metrics
- **Outliers**: Possible due to measurement errors or medical conditions

### فارسی (FA)
**مجموعه داده**: پایگاه داده اندازه گیری های فیزیولوژیک
- **نمونه ها**: متغیر (بسته به منبع داده)
- **ویژگی ها**: چندین اندازه گیری فیزیولوژیک
- **هدف**: ضربان قلب (تعداد ضربان در دقیقه)

**ویژگی های کلیدی**:
- **سن**: سن بیمار بر حسب سال
- **جنسیت**: مرد/زن (کدگذاری شده)
- **قد**: قد بر حسب سانتی متر
- **وزن**: وزن بر حسب کیلوگرم
- **فشار خون**: اندازه گیری های سیستولیک و دیاستولیک
- **سطوح کلسترول**: کل کلسترول، HDL، LDL
- **وضعیت سیگار کشیدن**: شاخص دودویی
- **فعالیت بدنی**: فراوانی/مدت تمرین
- **سطح استرس**: اندازه گیری استرس گزارش شده توسط فرد
- **کیفیت خواب**: معیارهای مدت و کیفیت خواب

**ویژگی های داده**:
- **محدودیت های فیزیولوژیک**: مقادیر باید در محدوده های پزشکی واقعی باشند
- **الگوهای همبستگی**: روابط قوی بین اندازه گیری های مرتبط
- **داده های مفقود**: رایج در معیارهای گزارش شده توسط فرد
- **داده های پرت**: ممکن است به دلیل خطاهای اندازه گیری یا شرایط پزشکی باشد

### Deutsch (DE)
**Datensatz**: Physiologische Messdatenbank
- **Stichproben**: Variabel (abhängig von Datenquelle)
- **Merkmale**: Mehrere physiologische Messungen
- **Ziel**: Herzfrequenz (Schläge pro Minute)

**Wichtige Merkmale**:
- **Alter**: Patientenalter in Jahren
- **Geschlecht**: Männlich/Weiblich (kodiert)
- **Körpergröße**: Größe in Zentimetern
- **Körpergewicht**: Gewicht in Kilogramm
- **Blutdruck**: Systolische und diastolische Messungen
- **Cholesterinspiegel**: Gesamtcholesterin, HDL, LDL
- **Raucherstatus**: Binärer Indikator
- **Körperliche Aktivität**: Trainingshäufigkeit/-dauer
- **Stresslevel**: Selbstberichtete Stressmessung
- **Schlafqualität**: Schlafdauer- und Qualitätsmetriken

**Datenmerkmale**:
- **Physiologische Einschränkungen**: Werte müssen innerhalb realistischer medizinischer Bereiche liegen
- **Korrelationsmuster**: Starke Beziehungen zwischen verwandten Messungen
- **Fehlende Daten**: Häufig bei selbstberichteten Metriken
- **Ausreißer**: Möglicherweise aufgrund von Messfehlern oder medizinischen Zuständen

## 🔬 Methodology / روش شناسی / Methodik

### English (EN)
**Data Preprocessing**:
1. **Physiological Validation**: Check values against medical normal ranges
2. **Missing Value Imputation**: Use domain-specific strategies for physiological data
3. **Feature Engineering**: Create interaction terms and derived metrics
4. **Outlier Detection**: Identify and handle physiologically impossible values
5. **Data Scaling**: Normalize features for optimal model performance

**Model Training**:
1. **Multiple Regression Algorithms**: Linear, Ridge, Lasso, Random Forest, SVM, Gradient Boosting, KNN, Decision Tree, XGBoost
2. **Feature Selection**: Identify most predictive physiological indicators
3. **Cross-Validation**: Ensure model generalization across different patient groups
4. **Hyperparameter Optimization**: Fine-tune models for physiological data patterns

**Advanced Techniques**:
1. **Time Series Analysis**: If temporal data is available
2. **Anomaly Detection**: Identify unusual heart rate patterns
3. **Ensemble Methods**: Combine multiple models for robust predictions

### فارسی (FA)
**پیش پردازش داده**:
1. **اعتبارسنجی فیزیولوژیک**: بررسی مقادیر در برابر محدوده های طبیعی پزشکی
2. **جایگزینی مقادیر مفقود**: استفاده از راهبردهای خاص حوزه برای داده های فیزیولوژیک
3. **مهندسی ویژگی**: ایجاد جملات تعاملی و معیارهای مشتق شده
4. **تشخیص داده های پرت**: شناسایی و مدیریت مقادیر فیزیولوژیک غیرممکن
5. **مقیاس بندی داده**: نرمال سازی ویژگی ها برای عملکرد بهینه مدل

**آموزش مدل**:
1. **چندین الگوریتم رگرسیون**: خطی، Ridge، Lasso، جنگل تصادفی، SVM، گرادیان بوستینگ، KNN، درخت تصمیم، XGBoost
2. **انتخاب ویژگی**: شناسایی شاخص های فیزیولوژیک پیش بینی کننده تر
3. **اعتبارسنجی متقابل**: اطمینان از تعمیم پذیری مدل در گروه های بیماران مختلف
4. **بهینه سازی هیپرپارامتر**: تنظیم دقیق مدل ها برای الگوهای داده های فیزیولوژیک

**تکنیک های پیشرفته**:
1. **تحلیل سری زمانی**: اگر داده های زمانی موجود باشد
2. **تشخیص ناهنجاری**: شناسایی الگوهای غیرعادی ضربان قلب
3. **روش های ترکیبی**: ترکیب چندین مدل برای پیش بینی های قوی

### Deutsch (DE)
**Daten-Vorverarbeitung**:
1. **Physiologische Validierung**: Werte anhand medizinischer Normalbereiche prüfen
2. **Imputation fehlender Werte**: Domänenspezifische Strategien für physiologische Daten verwenden
3. **Feature-Engineering**: Interaktionsterme und abgeleitete Metriken erstellen
4. **Ausreißer-Erkennung**: Physiologisch unmögliche Werte identifizieren und behandeln
5. **Daten-Skalierung**: Features für optimale Modellleistung normalisieren

**Modelltraining**:
1. **Mehrere Regressionsalgorithmen**: Linear, Ridge, Lasso, Random Forest, SVM, Gradient Boosting, KNN, Decision Tree, XGBoost
2. **Feature-Selection**: Die vorhersagekräftigsten physiologischen Indikatoren identifizieren
3. **Kreuzvalidierung**: Modellgeneralisierung über verschiedene Patientengruppen sicherstellen
4. **Hyperparameter-Optimierung**: Modelle für physiologische Datenmuster feinabstimmen

**Fortgeschrittene Techniken**:
1. **Zeitreihenanalyse**: Wenn zeitliche Daten verfügbar sind
2. **Anomalie-Erkennung**: Ungewöhnliche Herzfrequenzmuster identifizieren
3. **Ensemble-Methoden**: Mehrere Modelle für robuste Vorhersagen kombinieren

## 📈 Results & Analysis / نتایج و تحلیل / Ergebnisse & Analyse

### English (EN)
**Best Performing Models**:
- **XGBoost Regressor**: Highest prediction accuracy for physiological data
- **Random Forest**: Robust performance with feature importance insights
- **Gradient Boosting**: Excellent for complex physiological relationships

**Key Findings**:
- **Age and Weight**: Strongest predictors of resting heart rate
- **Physical Activity**: Significant negative correlation with heart rate
- **Blood Pressure**: Moderate correlation with heart rate variability
- **Feature Interactions**: Age × Weight interaction improves predictions

**Performance Metrics**:
- **R² Score**: Measures explained variance in heart rate
- **Mean Absolute Error**: Average prediction error in BPM
- **Root Mean Square Error**: Penalizes larger errors more heavily
- **Cross-Validation Score**: Ensures model stability across different data splits

**Clinical Insights**:
- Models can identify patients with abnormal heart rate patterns
- Feature importance helps understand physiological relationships
- Predictions can support clinical decision-making

### فارسی (FA)
**بهترین مدل های اجرا شده**:
- **رگرسور XGBoost**: بالاترین دقت پیش بینی برای داده های فیزیولوژیک
- **جنگل تصادفی**: عملکرد قوی با بینش های اهمیت ویژگی
- **گرادیان بوستینگ**: عالی برای روابط فیزیولوژیک پیچیده

**یافته های کلیدی**:
- **سن و وزن**: قوی ترین پیش بینی کننده های ضربان قلب در حالت استراحت
- **فعالیت بدنی**: همبستگی منفی قابل توجه با ضربان قلب
- **فشار خون**: همبستگی متوسط با تغییرات ضربان قلب
- **تعامل ویژگی ها**: تعامل سن × وزن پیش بینی ها را بهبود می بخشد

**معیارهای عملکرد**:
- **نمره R²**: معیار واریانس توضیح داده شده در ضربان قلب
- **میانگین خطای مطلق**: خطای پیش بینی متوسط بر حسب BPM
- **جذر میانگین خطای مربع**: خطاهای بزرگتر را شدیدتر تنبیه می کند
- **نمره اعتبارسنجی متقابل**: اطمینان از ثبات مدل در تقسیم بندی های مختلف داده

**بینش های بالینی**:
- مدل ها می توانند بیماران با الگوهای غیرطبیعی ضربان قلب را شناسایی کنند
- اهمیت ویژگی به درک روابط فیزیولوژیک کمک می کند
- پیش بینی ها می توانند تصمیم گیری بالینی را پشتیبانی کنند

### Deutsch (DE)
**Beste Modelle**:
- **XGBoost Regressor**: Höchste Vorhersagegenauigkeit für physiologische Daten
- **Random Forest**: Robuste Leistung mit Einblicken in die Feature-Importance
- **Gradient Boosting**: Ausgezeichnet für komplexe physiologische Beziehungen

**Wichtige Erkenntnisse**:
- **Alter und Gewicht**: Stärkste Prädiktoren der Ruheherzfrequenz
- **Körperliche Aktivität**: Signifikante negative Korrelation mit der Herzfrequenz
- **Blutdruck**: Moderate Korrelation mit der Herzfrequenzvariabilität
- **Feature-Interaktionen**: Alter × Gewicht-Interaktion verbessert Vorhersagen

**Leistungsmetriken**:
- **R²-Score**: Misst erklärte Varianz in der Herzfrequenz
- **Mean Absolute Error**: Durchschnittlicher Vorhersagefehler in BPM
- **Root Mean Square Error**: Bestraft größere Fehler stärker
- **Kreuzvalidierungsscore**: Stellt Modellstabilität über verschiedene Datenaufteilungen sicher

**Klinische Erkenntnisse**:
- Modelle können Patienten mit abnormalen Herzfrequenzmustern identifizieren
- Feature-Importance hilft, physiologische Beziehungen zu verstehen
- Vorhersagen können die klinische Entscheidungsfindung unterstützen

## 🚀 Usage Instructions / دستورالعمل استفاده / Bedienungsanleitung

### English (EN)
**Prerequisites**:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost
```

**Running the Project**:
```python
# Import required modules
from Tool_box import DataProcessingTool, RegressionTool, ModelEvaluationTool

# Load and preprocess physiological data
processor = DataProcessingTool()
data = processor.load_data("Dataset/heart_data.csv")

# Apply physiological data validation
# (Custom preprocessing for medical constraints)

processed_data = processor.prepare_data_for_ml(data, target_column="HeartRate")

# Train regression models
regressor = RegressionTool()
models = regressor.train_multiple_models(processed_data['X_train'], processed_data['y_train'])

# Evaluate models
evaluator = ModelEvaluationTool()
results = evaluator.evaluate_regression_models(models, processed_data['X_test'], processed_data['y_test'])
```

**Key Features**:
- Physiological data validation and constraint checking
- Advanced feature engineering for medical data
- Comprehensive regression model comparison
- Clinical interpretation of results

### فارسی (FA)
**پیش نیازها**:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost
```

**اجرای پروژه**:
```python
# وارد کردن ماژول های مورد نیاز
from Tool_box import DataProcessingTool, RegressionTool, ModelEvaluationTool

# بارگیری و پیش پردازش داده های فیزیولوژیک
processor = DataProcessingTool()
data = processor.load_data("Dataset/heart_data.csv")

# اعمال اعتبارسنجی داده های فیزیولوژیک
# (پیش پردازش سفارشی برای محدودیت های پزشکی)

processed_data = processor.prepare_data_for_ml(data, target_column="HeartRate")

# آموزش مدل های رگرسیون
regressor = RegressionTool()
models = regressor.train_multiple_models(processed_data['X_train'], processed_data['y_train'])

# ارزیابی مدل ها
evaluator = ModelEvaluationTool()
results = evaluator.evaluate_regression_models(models, processed_data['X_test'], processed_data['y_test'])
```

**ویژگی های کلیدی**:
- اعتبارسنجی داده های فیزیولوژیک و بررسی محدودیت ها
- مهندسی ویژگی پیشرفته برای داده های پزشکی
- مقایسه جامع مدل های رگرسیون
- تفسیر بالینی نتایج

### Deutsch (DE)
**Voraussetzungen**:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost
```

**Projekt ausführen**:
```python
# Erforderliche Module importieren
from Tool_box import DataProcessingTool, RegressionTool, ModelEvaluationTool

# Physiologische Daten laden und vorverarbeiten
processor = DataProcessingTool()
data = processor.load_data("Dataset/heart_data.csv")

# Physiologische Datenvalidierung anwenden
# (Benutzerdefinierte Vorverarbeitung für medizinische Einschränkungen)

processed_data = processor.prepare_data_for_ml(data, target_column="HeartRate")

# Regressionsmodelle trainieren
regressor = RegressionTool()
models = regressor.train_multiple_models(processed_data['X_train'], processed_data['y_train'])

# Modelle evaluieren
evaluator = ModelEvaluationTool()
results = evaluator.evaluate_regression_models(models, processed_data['X_test'], processed_data['y_test'])
```

**Wichtige Funktionen**:
- Physiologische Datenvalidierung und Constraint-Prüfung
- Fortgeschrittenes Feature-Engineering für medizinische Daten
- Umfassender Regressionsmodellvergleich
- Klinische Interpretation der Ergebnisse

## ⚡ Optimization Notes / یادداشت های بهینه سازی / Optimierungshinweise

### English (EN)
**Performance Optimizations**:
- **Feature Scaling**: Critical for physiological data with different units
- **Cross-Validation**: Essential for medical data generalization
- **Regularization**: Prevents overfitting with correlated physiological features
- **Ensemble Methods**: Combine models for robust predictions

**Physiological Data Specific**:
- **Medical Constraints**: Enforce realistic value ranges
- **Domain Knowledge**: Use physiological relationships in feature engineering
- **Outlier Handling**: Careful treatment of extreme but valid measurements
- **Temporal Patterns**: Consider time-based features if available

**Future Enhancements**:
- **Real-time Monitoring**: Continuous heart rate prediction
- **Anomaly Detection**: Identify dangerous heart rate patterns
- **Personalized Models**: Patient-specific model adaptation
- **Integration with Wearables**: Direct data from fitness trackers

### فارسی (FA)
**بهینه سازی های عملکرد**:
- **مقیاس بندی ویژگی**: حیاتی برای داده های فیزیولوژیک با واحدهای مختلف
- **اعتبارسنجی متقابل**: ضروری برای تعمیم داده های پزشکی
- **منظم سازی**: جلوگیری از بیش برازش با ویژگی های فیزیولوژیک همبسته
- **روش های ترکیبی**: ترکیب مدل ها برای پیش بینی های قوی

**مختص داده های فیزیولوژیک**:
- **محدودیت های پزشکی**: اعمال محدوده های واقعی مقادیر
- **دانش حوزه**: استفاده از روابط فیزیولوژیک در مهندسی ویژگی
- **مدیریت داده های پرت**: برخورد دقیق با اندازه گیری های حدی اما معتبر
- **الگوهای زمانی**: در نظر گرفتن ویژگی های مبتنی بر زمان در صورت موجود بودن

**ارتقا های آینده**:
- **مانیتورینگ بلادرنگ**: پیش بینی مداوم ضربان قلب
- **تشخیص ناهنجاری**: شناسایی الگوهای خطرناک ضربان قلب
- **مدل های شخصی سازی شده**: سازگاری مدل خاص بیمار
- **ادغام با لوازم پوشیدنی**: داده مستقیم از ردیاب های تناسب اندام

### Deutsch (DE)
**Leistungsoptimierungen**:
- **Feature-Scaling**: Kritisch für physiologische Daten mit verschiedenen Einheiten
- **Kreuzvalidierung**: Wesentlich für die Generalisierung medizinischer Daten
- **Regularisierung**: Verhindert Overfitting bei korrelierten physiologischen Features
- **Ensemble-Methoden**: Modelle für robuste Vorhersagen kombinieren

**Physiologische Daten spezifisch**:
- **Medizinische Constraints**: Realistische Wertebereiche durchsetzen
- **Domänenwissen**: Physiologische Beziehungen im Feature-Engineering verwenden
- **Ausreißer-Behandlung**: Sorgfältige Behandlung extremer, aber gültiger Messungen
- **Zeitliche Muster**: Zeitbasierte Features berücksichtigen, wenn verfügbar

**Zukünftige Verbesserungen**:
- **Echtzeit-Monitoring**: Kontinuierliche Herzfrequenzvorhersage
- **Anomalie-Erkennung**: Gefährliche Herzfrequenzmuster identifizieren
- **Personalisierte Modelle**: Patientenspezifische Modelle anpassen
- **Integration mit Wearables**: Direkte Daten von Fitness-Trackern

## 📞 Contact / تماس / Kontakt
For questions or improvements, please refer to the main TOOL-BOX repository documentation.
برای سوالات یا بهبودها، لطفاً به مستندات اصلی TOOL-BOX مراجعه کنید.
Für Fragen oder Verbesserungen wenden Sie sich bitte an die Haupt-TOOL-BOX-Repository-Dokumentation.
