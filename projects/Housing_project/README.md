# Housing Price Prediction Project / پروژه پیش بینی قیمت مسکن / Immobilien-Vorhersage-Projekt

## 🎯 Project Overview / مرور پروژه / Projektübersicht

### English (EN)
This project demonstrates machine learning techniques for predicting housing prices using real estate data. It implements comprehensive feature engineering, multiple regression algorithms, and advanced data preprocessing to model the complex relationships between property characteristics and market prices.

### فارسی (FA)
این پروژه تکنیک های یادگیری ماشین برای پیش بینی قیمت مسکن با استفاده از داده های املاک را نشان می دهد. این پروژه از مهندسی ویژگی جامع، چندین الگوریتم رگرسیون و پیش پردازش داده های پیشرفته برای مدل کردن روابط پیچیده بین ویژگی های ملک و قیمت های بازار استفاده می کند.

### Deutsch (DE)
Dieses Projekt demonstriert Machine-Learning-Techniken zur Vorhersage von Immobilienpreisen anhand von Immobiliendaten. Es implementiert umfassendes Feature-Engineering, mehrere Regressionsalgorithmen und fortgeschrittene Daten-Vorverarbeitung, um die komplexen Beziehungen zwischen Immobilienmerkmalen und Marktpreisen zu modellieren.

## 📊 Data Description / توضیحات داده ها / Datenbeschreibung

### English (EN)
**Dataset**: California Housing Prices
- **Samples**: 20,640 housing districts
- **Features**: 8+ property and location characteristics
- **Target**: Median house value (in hundreds of thousands of dollars)

**Key Features**:
- **MedInc**: Median income in block group
- **HouseAge**: Median house age in block
- **AveRooms**: Average number of rooms
- **AveBedrms**: Average number of bedrooms
- **Population**: Block population
- **AveOccup**: Average number of household members
- **Latitude**: Property latitude coordinate
- **Longitude**: Property longitude coordinate

**Advanced Features** (Engineered):
- **Rooms per Household**: AveRooms / AveOccup
- **Bedrooms per Room**: AveBedrms / AveRooms
- **Population per Household**: Population / AveOccup
- **Household Density**: Population / (AveOccup * HouseAge)
- **Income per Room**: MedInc / AveRooms

**Data Characteristics**:
- **Geospatial Data**: Latitude/Longitude for location-based analysis
- **Census Block Data**: Aggregated demographic information
- **Price Capping**: Maximum value capped at $500,000
- **Missing Values**: Some features may have missing data

### فارسی (FA)
**مجموعه داده**: قیمت مسکن کالیفرنیا
- **نمونه ها**: 20,640 منطقه مسکونی
- **ویژگی ها**: 8+ ویژگی ملک و موقعیت
- **هدف**: میانه قیمت خانه (به صد هزار دلار)

**ویژگی های کلیدی**:
- **MedInc**: میانه درآمد در گروه بلوک
- **HouseAge**: میانه سن خانه در بلوک
- **AveRooms**: میانگین تعداد اتاق ها
- **AveBedrms**: میانگین تعداد اتاق های خواب
- **Population**: جمعیت بلوک
- **AveOccup**: میانگین تعداد اعضای خانوار
- **Latitude**: مختصات عرض جغرافیایی ملک
- **Longitude**: مختصات طول جغرافیایی ملک

**ویژگی های پیشرفته** (مهندسی شده):
- **اتاق ها به ازای هر خانوار**: AveRooms / AveOccup
- **اتاق های خواب به ازای هر اتاق**: AveBedrms / AveRooms
- **جمعیت به ازای هر خانوار**: Population / AveOccup
- **تراکم خانوار**: Population / (AveOccup * HouseAge)
- **درآمد به ازای هر اتاق**: MedInc / AveRooms

**ویژگی های داده**:
- **داده های مکانی**: عرض و طول جغرافیایی برای تحلیل مبتنی بر موقعیت
- **داده های منطقه سرشماری**: اطلاعات دموگرافیک تجمیع شده
- **سقف قیمت**: حداکثر مقدار تا 500,000 دلار محدود شده است
- **مقادیر مفقود**: برخی ویژگی ها ممکن است داده های مفقود داشته باشند

### Deutsch (DE)
**Datensatz**: Kalifornische Immobilienpreise
- **Stichproben**: 20.640 Wohngebiete
- **Merkmale**: 8+ Immobilien- und Standortmerkmale
- **Ziel**: Median-Hauswert (in Hunderttausend Dollar)

**Wichtige Merkmale**:
- **MedInc**: Median-Einkommen in der Blockgruppe
- **HouseAge**: Median-Hausalter im Block
- **AveRooms**: Durchschnittliche Anzahl der Zimmer
- **AveBedrms**: Durchschnittliche Anzahl der Schlafzimmer
- **Population**: Blockbevölkerung
- **AveOccup**: Durchschnittliche Anzahl der Haushaltsmitglieder
- **Latitude**: Immobilien-Breitengradkoordinate
- **Longitude**: Immobilien-Längengradkoordinate

**Erweiterte Merkmale** (Feature-Engineering):
- **Zimmer pro Haushalt**: AveRooms / AveOccup
- **Schlafzimmer pro Zimmer**: AveBedrms / AveRooms
- **Bevölkerung pro Haushalt**: Population / AveOccup
- **Haushaltsdichte**: Population / (AveOccup * HouseAge)
- **Einkommen pro Zimmer**: MedInc / AveRooms

**Datenmerkmale**:
- **Geodaten**: Breiten- und Längenkoordinaten für standortbasierte Analyse
- **Census-Block-Daten**: Aggregierte demografische Informationen
- **Preisobergrenze**: Maximaler Wert auf 500.000 Dollar begrenzt
- **Fehlende Werte**: Einige Merkmale können fehlende Daten aufweisen

## 🔬 Methodology / روش شناسی / Methodik

### English (EN)
**Data Preprocessing**:
1. **Geospatial Analysis**: Convert coordinates to meaningful location features
2. **Feature Engineering**: Create interaction terms and derived metrics
3. **Outlier Detection**: Identify and handle extreme property values
4. **Missing Value Handling**: Impute missing data using domain knowledge
5. **Feature Scaling**: Normalize features for optimal model performance

**Model Training**:
1. **Multiple Regression Algorithms**: Linear, Ridge, Lasso, Random Forest, SVM, Gradient Boosting, KNN, Decision Tree, XGBoost
2. **Geospatial Features**: Incorporate location-based predictors
3. **Cross-Validation**: Ensure model generalization across different regions
4. **Hyperparameter Optimization**: Fine-tune models for real estate patterns

**Advanced Techniques**:
1. **Geographic Clustering**: Group properties by location similarity
2. **Market Trend Analysis**: Identify regional price patterns
3. **Feature Importance**: Understand key drivers of property values
4. **Ensemble Methods**: Combine multiple models for robust predictions

### فارسی (FA)
**پیش پردازش داده**:
1. **تحلیل مکانی**: تبدیل مختصات به ویژگی های موقعیت معنادار
2. **مهندسی ویژگی**: ایجاد جملات تعاملی و معیارهای مشتق شده
3. **تشخیص داده های پرت**: شناسایی و مدیریت مقادیر ملکی حدی
4. **مدیریت مقادیر مفقود**: جایگزینی داده های مفقود با استفاده از دانش حوزه
5. **مقیاس بندی ویژگی**: نرمال سازی ویژگی ها برای عملکرد بهینه مدل

**آموزش مدل**:
1. **چندین الگوریتم رگرسیون**: خطی، Ridge، Lasso، جنگل تصادفی، SVM، گرادیان بوستینگ، KNN، درخت تصمیم، XGBoost
2. **ویژگی های مکانی**: ادغام پیش بینی کننده های مبتنی بر موقعیت
3. **اعتبارسنجی متقابل**: اطمینان از تعمیم پذیری مدل در مناطق مختلف
4. **بهینه سازی هیپرپارامتر**: تنظیم دقیق مدل ها برای الگوهای املاک

**تکنیک های پیشرفته**:
1. **خوشه بندی جغرافیایی**: گروه بندی ملک ها بر اساس شباهت موقعیت
2. **تحلیل روند بازار**: شناسایی الگوهای قیمت منطقه ای
3. **اهمیت ویژگی**: درک عوامل کلیدی ارزش ملک
4. **روش های ترکیبی**: ترکیب چندین مدل برای پیش بینی های قوی

### Deutsch (DE)
**Daten-Vorverarbeitung**:
1. **Geospatiale Analyse**: Koordinaten in sinnvolle Standortmerkmale umwandeln
2. **Feature-Engineering**: Interaktionsterme und abgeleitete Metriken erstellen
3. **Ausreißer-Erkennung**: Extreme Immobilienwerte identifizieren und behandeln
4. **Behandlung fehlender Werte**: Fehlende Daten mit Domänenwissen auffüllen
5. **Feature-Scaling**: Features für optimale Modellleistung normalisieren

**Modelltraining**:
1. **Mehrere Regressionsalgorithmen**: Linear, Ridge, Lasso, Random Forest, SVM, Gradient Boosting, KNN, Decision Tree, XGBoost
2. **Geospatiale Features**: Standortbasierte Prädiktoren integrieren
3. **Kreuzvalidierung**: Modellgeneralisierung über verschiedene Regionen sicherstellen
4. **Hyperparameter-Optimierung**: Modelle für Immobilienmuster feinabstimmen

**Fortgeschrittene Techniken**:
1. **Geografisches Clustering**: Immobilien nach Standortähnlichkeit gruppieren
2. **Markttrend-Analyse**: Regionale Preismuster identifizieren
3. **Feature-Importance**: Schlüsseltreiber von Immobilienwerten verstehen
4. **Ensemble-Methoden**: Mehrere Modelle für robuste Vorhersagen kombinieren

## 📈 Results & Analysis / نتایج و تحلیل / Ergebnisse & Analyse

### English (EN)
**Best Performing Models**:
- **XGBoost Regressor**: Highest prediction accuracy for real estate data
- **Random Forest**: Robust performance with feature importance insights
- **Gradient Boosting**: Excellent for complex property relationships

**Key Findings**:
- **Location Features**: Latitude/Longitude are critical predictors
- **Income Correlation**: Strong positive correlation with house prices
- **Room Density**: Important indicator of property value
- **Geographic Patterns**: Clear regional price variations

**Performance Metrics**:
- **R² Score**: Measures explained variance in housing prices
- **Mean Absolute Error**: Average prediction error in $100,000 units
- **Root Mean Square Error**: Penalizes larger errors more heavily
- **Cross-Validation Score**: Ensures model stability across regions

**Market Insights**:
- Models can identify undervalued properties
- Feature importance reveals key market drivers
- Geographic clustering shows regional trends
- Predictions support investment decision-making

### فارسی (FA)
**بهترین مدل های اجرا شده**:
- **رگرسور XGBoost**: بالاترین دقت پیش بینی برای داده های املاک
- **جنگل تصادفی**: عملکرد قوی با بینش های اهمیت ویژگی
- **گرادیان بوستینگ**: عالی برای روابط پیچیده ملک

**یافته های کلیدی**:
- **ویژگی های موقعیت**: عرض و طول جغرافیایی پیش بینی کننده های حیاتی هستند
- **همبستگی درآمد**: همبستگی مثبت قوی با قیمت خانه
- **تراکم اتاق**: شاخص مهم ارزش ملک
- **الگوهای جغرافیایی**: تغییرات قیمت منطقه ای واضح

**معیارهای عملکرد**:
- **نمره R²**: معیار واریانس توضیح داده شده در قیمت مسکن
- **میانگین خطای مطلق**: خطای پیش بینی متوسط بر حسب 100,000 دلار
- **جذر میانگین خطای مربع**: خطاهای بزرگتر را شدیدتر تنبیه می کند
- **نمره اعتبارسنجی متقابل**: اطمینان از ثبات مدل در مناطق مختلف

**بینش های بازار**:
- مدل ها می توانند ملک های کم ارزش را شناسایی کنند
- اهمیت ویژگی نشان دهنده عوامل کلیدی بازار است
- خوشه بندی جغرافیایی روندهای منطقه ای را نشان می دهد
- پیش بینی ها تصمیم گیری سرمایه گذاری را پشتیبانی می کنند

### Deutsch (DE)
**Beste Modelle**:
- **XGBoost Regressor**: Höchste Vorhersagegenauigkeit für Immobiliendaten
- **Random Forest**: Robuste Leistung mit Einblicken in die Feature-Importance
- **Gradient Boosting**: Ausgezeichnet für komplexe Immobilienbeziehungen

**Wichtige Erkenntnisse**:
- **Standortmerkmale**: Breiten- und Längengrad sind kritische Prädiktoren
- **Einkommenskorrelation**: Starke positive Korrelation mit Hauspreisen
- **Zimmerdichte**: Wichtiger Indikator für Immobilienwert
- **Geografische Muster**: Klar erkennbare regionale Preisschwankungen

**Leistungsmetriken**:
- **R²-Score**: Misst erklärte Varianz in Immobilienpreisen
- **Mean Absolute Error**: Durchschnittlicher Vorhersagefehler in 100.000-Dollar-Einheiten
- **Root Mean Square Error**: Bestraft größere Fehler stärker
- **Kreuzvalidierungsscore**: Stellt Modellstabilität über verschiedene Regionen sicher

**Markteinblicke**:
- Modelle können unterbewertete Immobilien identifizieren
- Feature-Importance zeigt Schlüsseltreiber des Marktes auf
- Geografisches Clustering zeigt regionale Trends
- Vorhersagen unterstützen die Investitionsentscheidungsfindung

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

# Load and preprocess housing data
processor = DataProcessingTool()
data = processor.load_data("Dataset/housing.csv")

# Apply advanced feature engineering
# (Create interaction terms and geospatial features)

processed_data = processor.prepare_data_for_ml(data, target_column="median_house_value")

# Train regression models
regressor = RegressionTool()
models = regressor.train_multiple_models(processed_data['X_train'], processed_data['y_train'])

# Evaluate models
evaluator = ModelEvaluationTool()
results = evaluator.evaluate_regression_models(models, processed_data['X_test'], processed_data['y_test'])
```

**Key Features**:
- Advanced feature engineering for real estate data
- Geospatial analysis and location-based features
- Comprehensive regression model comparison
- Market trend analysis and insights

### فارسی (FA)
**پیش نیازها**:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost
```

**اجرای پروژه**:
```python
# وارد کردن ماژول های مورد نیاز
from Tool_box import DataProcessingTool, RegressionTool, ModelEvaluationTool

# بارگیری و پیش پردازش داده های مسکن
processor = DataProcessingTool()
data = processor.load_data("Dataset/housing.csv")

# اعمال مهندسی ویژگی پیشرفته
# (ایجاد جملات تعاملی و ویژگی های مکانی)

processed_data = processor.prepare_data_for_ml(data, target_column="median_house_value")

# آموزش مدل های رگرسیون
regressor = RegressionTool()
models = regressor.train_multiple_models(processed_data['X_train'], processed_data['y_train'])

# ارزیابی مدل ها
evaluator = ModelEvaluationTool()
results = evaluator.evaluate_regression_models(models, processed_data['X_test'], processed_data['y_test'])
```

**ویژگی های کلیدی**:
- مهندسی ویژگی پیشرفته برای داده های املاک
- تحلیل مکانی و ویژگی های مبتنی بر موقعیت
- مقایسه جامع مدل های رگرسیون
- تحلیل روند بازار و بینش ها

### Deutsch (DE)
**Voraussetzungen**:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost
```

**Projekt ausführen**:
```python
# Erforderliche Module importieren
from Tool_box import DataProcessingTool, RegressionTool, ModelEvaluationTool

# Immobiliendaten laden und vorverarbeiten
processor = DataProcessingTool()
data = processor.load_data("Dataset/housing.csv")

# Fortgeschrittenes Feature-Engineering anwenden
# (Interaktionsterme und geospatiale Features erstellen)

processed_data = processor.prepare_data_for_ml(data, target_column="median_house_value")

# Regressionsmodelle trainieren
regressor = RegressionTool()
models = regressor.train_multiple_models(processed_data['X_train'], processed_data['y_train'])

# Modelle evaluieren
evaluator = ModelEvaluationTool()
results = evaluator.evaluate_regression_models(models, processed_data['X_test'], processed_data['y_test'])
```

**Wichtige Funktionen**:
- Fortgeschrittenes Feature-Engineering für Immobiliendaten
- Geospatiale Analyse und standortbasierte Features
- Umfassender Regressionsmodellvergleich
- Marktrendite-Analyse und Einblicke

## ⚡ Optimization Notes / یادداشت های بهینه سازی / Optimierungshinweise

### English (EN)
**Performance Optimizations**:
- **Geospatial Features**: Critical for location-based price prediction
- **Feature Engineering**: Interaction terms improve model accuracy
- **Cross-Validation**: Essential for regional generalization
- **Regularization**: Prevents overfitting with many engineered features

**Real Estate Data Specific**:
- **Market Knowledge**: Use domain expertise in feature creation
- **Geographic Constraints**: Consider location-based limitations
- **Price Capping**: Handle maximum value constraints appropriately
- **Seasonal Patterns**: Consider temporal factors if available

**Future Enhancements**:
- **Interactive Dashboard**: Web-based visualization for property analysis
- **Market Trend Analysis**: Time-series analysis of price changes
- **Investment Recommendations**: ROI-based property suggestions
- **Integration with APIs**: Real-time market data integration

### فارسی (FA)
**بهینه سازی های عملکرد**:
- **ویژگی های مکانی**: حیاتی برای پیش بینی قیمت مبتنی بر موقعیت
- **مهندسی ویژگی**: جملات تعاملی دقت مدل را بهبود می بخشد
- **اعتبارسنجی متقابل**: ضروری برای تعمیم منطقه ای
- **منظم سازی**: جلوگیری از بیش برازش با ویژگی های مهندسی شده بسیاری

**مختص داده های املاک**:
- **دانش بازار**: استفاده از تخصص حوزه در ایجاد ویژگی
- **محدودیت های جغرافیایی**: در نظر گرفتن محدودیت های مبتنی بر موقعیت
- **سقف قیمت**: مدیریت مناسب محدودیت های حداکثر مقدار
- **الگوهای فصلی**: در نظر گرفتن عوامل زمانی در صورت موجود بودن

**ارتقا های آینده**:
- **داشبورد تعاملی**: تصویرسازی مبتنی بر وب برای تحلیل ملک
- **تحلیل روند بازار**: تحلیل سری زمانی تغییرات قیمت
- **توصیه های سرمایه گذاری**: پیشنهادات ملک مبتنی بر بازده سرمایه گذاری
- **ادغام با API ها**: ادغام داده های بازار بلادرنگ

### Deutsch (DE)
**Leistungsoptimierungen**:
- **Geospatiale Features**: Kritisch für standortbasierte Preisvorhersage
- **Feature-Engineering**: Interaktionsterme verbessern die Modellgenauigkeit
- **Kreuzvalidierung**: Wesentlich für regionale Generalisierung
- **Regularisierung**: Verhindert Overfitting mit vielen engineered Features

**Immobilien-Daten spezifisch**:
- **Marktwissen**: Domänenexpertise bei der Merkmalerstellung verwenden
- **Geografische Einschränkungen**: Standortbasierte Einschränkungen berücksichtigen
- **Preisobergrenze**: Maximale Wertbeschränkungen angemessen behandeln
- **Saisonale Muster**: Zeitliche Faktoren berücksichtigen, wenn verfügbar

**Zukünftige Verbesserungen**:
- **Interaktives Dashboard**: Webbasierte Visualisierung für Immobilienanalyse
- **Markttrend-Analyse**: Zeitreihenanalyse von Preisschwankungen
- **Investitionsempfehlungen**: ROI-basierte Immobilienempfehlungen
- **Integration mit APIs**: Echtzeit-Marktdatenintegration

## 📞 Contact / تماس / Kontakt
For questions or improvements, please refer to the main TOOL-BOX repository documentation.
برای سوالات یا بهبودها، لطفاً به مستندات اصلی TOOL-BOX مراجعه کنید.
Für Fragen oder Verbesserungen wenden Sie sich bitte an die Haupt-TOOL-BOX-Repository-Dokumentation.
