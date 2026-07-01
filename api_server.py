"""
TOOL-BOX API Server
FastAPI server to expose ML tools as REST API endpoints for the HTML interface
"""

import os
import tempfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
import uvicorn
import base64
import io
import json
import joblib
import urllib.request
import urllib.error

# Import TOOL-BOX modules
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Tool_box.data_processing_tool import DataProcessingTool
from Tool_box.classification_tool import ClassificationTool
from Tool_box.regression_tool import RegressionTool
from Tool_box.model_evaluation_tool import ModelEvaluationTool
from Tool_box.cross_validation_tool import CrossValidationTool
from Tool_box.clustering_tool import ClusteringTool
from Tool_box.optimizer import Optimizer
from Tool_box.feature_selector import FeatureSelector
from Tool_box.model_interpreter import ModelInterpreter

# Initialize FastAPI app
app = FastAPI(
    title="TOOL-BOX API",
    description="Professional ML Toolkit API for Data Scientists",
    version="1.0.0"
)

# Add CORS middleware to allow frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the HTML frontends
STATIC_DIR_OLD = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'interface', 'html')
STATIC_DIR_NEW = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'interface', 'html_new')
if os.path.isdir(STATIC_DIR_OLD):
    app.mount("/ui", StaticFiles(directory=STATIC_DIR_OLD, html=True), name="ui")
if os.path.isdir(STATIC_DIR_NEW):
    app.mount("/app", StaticFiles(directory=STATIC_DIR_NEW, html=True), name="app")

# Global instances of ML tools
data_processor = DataProcessingTool()
classifier = ClassificationTool()
regressor = RegressionTool()
evaluator = ModelEvaluationTool()
cv_tool = CrossValidationTool()
clusterer = ClusteringTool()
tuner = Optimizer()
feature_analyzer = FeatureSelector()
interpreter = ModelInterpreter()

# Pydantic models for request/response
class DataUploadResponse(BaseModel):
    filename: str
    shape: tuple
    columns: List[str]
    dtypes: Dict[str, str]
    preview: Dict[str, Any]

class LoadFromUrlRequest(BaseModel):
    url: str

class ProcessDataRequest(BaseModel):
    target_column: Optional[str] = None
    test_size: float = 0.2
    preprocessing_steps: List[str] = ["clean", "encode", "scale"]
    cv_folds: int = 5
    cv_method: str = "kfold"

class TrainModelsRequest(BaseModel):
    model_type: str
    model_list: Optional[List[str]] = None
    n_jobs: int = 1
    test_size: float = 0.2
    cv_folds: int = 0

class ModelEvaluationRequest(BaseModel):
    model_type: str

class CrossValidationRequest(BaseModel):
    model_name: str
    folds: int = 5
    method: str = "k_fold"

class HyperparameterTuningRequest(BaseModel):
    model_type: str
    model_name: str
    method: str = "optuna"

class FeatureAnalysisRequest(BaseModel):
    model_name: Optional[str] = None

class ModelInterpretRequest(BaseModel):
    model_name: str
    method: str = "lime"
    feature_idx: Optional[int] = None
    instance_idx: int = 0

# Global storage for session data
session_data = {
    'uploaded_data': None,
    'processed_data': None,
    'trained_models': {},
    'evaluation_results': None
}

def get_data_overview(data: pd.DataFrame) -> Dict[str, Any]:
    overview = data_processor.data_overview(data)
    return {
        'shape': overview['shape'],
        'columns': list(overview['columns']),
        'dtypes': {k: str(v) for k, v in overview['dtypes'].items()},
        'missing_values': overview['missing_values'],
        'duplicates': overview['duplicates'],
        'numerical_summary': overview.get('numerical_summary', {})
    }

def encode_dataframe_for_json(df: pd.DataFrame, max_rows: int = 100) -> Dict[str, Any]:
    preview_df = df.head(max_rows)
    return {
        'columns': list(preview_df.columns),
        'data': preview_df.to_dict('records'),
        'shape': df.shape,
        'truncated': len(df) > max_rows
    }



# API Routes
@app.get("/")
async def root():
    return {
        "message": "TOOL-BOX API Server",
        "version": "1.0.0",
        "status": "running",
        "ui": "Open /ui/step1_load_data.html (old) or /app/app.html (new) in your browser",
        "endpoints": [
            "/upload-data", "/load-from-url", "/process-data", "/analyze-eda",
            "/train-models", "/evaluate-models", "/cross-validate",
            "/tune-hyperparameters", "/analyze-features", "/interpret-model",
            "/export-model", "/export-report", "/get-recommendation",
            "/get-models", "/get-data-info", "/predict", "/reset-session"
        ]
    }

@app.post("/upload-data", response_model=DataUploadResponse)
async def upload_data(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        file_extension = file.filename.split('.')[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as tmp_file:
            tmp_file.write(contents)
            tmp_file_path = tmp_file.name
        data = data_processor.load_data(tmp_file_path)
        session_data['uploaded_data'] = data
        overview = get_data_overview(data)
        preview = encode_dataframe_for_json(data, max_rows=50)
        os.unlink(tmp_file_path)
        return DataUploadResponse(
            filename=file.filename,
            shape=overview['shape'],
            columns=overview['columns'],
            dtypes=overview['dtypes'],
            preview=preview
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")

def _is_private_ip(hostname: str) -> bool:
    import ipaddress, socket
    try:
        ip = ipaddress.ip_address(socket.gethostbyname(hostname))
        return ip.is_private or ip.is_loopback or ip.is_reserved or ip.is_link_local
    except (ValueError, socket.gaierror):
        return False

@app.post("/load-from-url", response_model=DataUploadResponse)
async def load_from_url(request: LoadFromUrlRequest):
    try:
        from urllib.parse import urlparse
        url = request.url.strip()
        if not url.startswith(('http://', 'https://')):
            raise HTTPException(status_code=400, detail="Invalid URL. Must start with http:// or https://")
        hostname = urlparse(url).hostname
        if not hostname or _is_private_ip(hostname):
            raise HTTPException(status_code=400, detail="URL must point to a public host")
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=60) as resp:
            content = resp.read()
        filename = url.split('/')[-1].split('?')[0] or 'remote_data'
        file_extension = filename.split('.')[-1].lower() if '.' in filename else 'csv'
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as tmp_file:
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        data = data_processor.load_data(tmp_file_path)
        session_data['uploaded_data'] = data
        overview = get_data_overview(data)
        preview = encode_dataframe_for_json(data, max_rows=50)
        os.unlink(tmp_file_path)
        return DataUploadResponse(
            filename=filename,
            shape=overview['shape'],
            columns=overview['columns'],
            dtypes=overview['dtypes'],
            preview=preview
        )
    except urllib.error.HTTPError as e:
        raise HTTPException(status_code=400, detail=f"URL download failed (HTTP {e.code}): {e.reason}")
    except urllib.error.URLError as e:
        raise HTTPException(status_code=400, detail=f"URL unreachable: {e.reason}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error loading from URL: {str(e)}")

@app.post("/process-data")
async def process_data(request: ProcessDataRequest):
    try:
        if session_data['uploaded_data'] is None:
            raise HTTPException(status_code=400, detail="No data uploaded. Upload data first.")
        data = session_data['uploaded_data'].copy()
        target = request.target_column

        # Split BEFORE scaling to avoid scaling the target
        if target and target in data.columns:
            X = data.drop(target, axis=1)
            y = data[target]
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=request.test_size, random_state=42)

            # Drop high-cardinality columns (likely IDs/names, not useful for ML)
            high_card_cols = [c for c in X_train.columns if X_train[c].nunique() > 50 and X_train[c].dtype == 'object']
            if high_card_cols:
                X_train = X_train.drop(columns=high_card_cols)
                X_test = X_test.drop(columns=high_card_cols)

            # Preprocess only feature columns
            steps = request.preprocessing_steps or []
            if 'clean' in steps:
                X_train = data_processor.clean_data(X_train)
                X_test = data_processor.clean_data(X_test)
            if 'encode' in steps:
                X_train, _ = data_processor.encode_categorical(X_train)
                X_test, _ = data_processor.encode_categorical(X_test)
            if 'scale' in steps:
                numeric_cols = X_train.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    X_train = X_train.copy()
                    X_test = X_test.copy()
                    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
                    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

            # Ensure columns match (reindex X_test to X_train columns)
            X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

            processed_result = {
                'X_train': X_train, 'X_test': X_test,
                'y_train': y_train, 'y_test': y_test
            }
        else:
            processed_result = data_processor.prepare_data_for_ml(
                data=data, target_column=target,
                test_size=request.test_size,
                preprocessing_steps=request.preprocessing_steps)

        session_data['processed_data'] = processed_result
        session_data['validation_config'] = {
            'test_size': request.test_size,
            'cv_folds': request.cv_folds,
            'cv_method': request.cv_method
        }
        shape = processed_result.get('X_train', processed_result.get('processed_data', pd.DataFrame())).shape
        features = []
        if 'X_train' in processed_result:
            features = list(processed_result['X_train'].columns)
        elif 'processed_data' in processed_result:
            features = list(processed_result['processed_data'].columns)
        return {
            "message": "Data processed successfully",
            "processed_shape": list(shape),
            "target_column": target,
            "has_split": 'X_train' in processed_result,
            "preprocessing_steps": request.preprocessing_steps,
            "features": features
        }
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Error processing data: {str(e)}")

def _convert_to_json_safe(obj):
    """Recursively convert numpy/pandas types to JSON-safe native Python types."""
    if isinstance(obj, dict):
        return {k: _convert_to_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_to_json_safe(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return _convert_to_json_safe(obj.tolist())
    elif isinstance(obj, np.dtype):
        return str(obj)
    elif hasattr(obj, 'dtype'):  # pandas Series, etc.
        return _convert_to_json_safe(obj.to_dict())
    elif pd.isna(obj):
        return None
    return obj

@app.post("/analyze-eda")
async def analyze_eda():
    try:
        if session_data['uploaded_data'] is None:
            raise HTTPException(status_code=400, detail="No data uploaded.")
        data = session_data['uploaded_data']
        result = data_processor.generate_eda_summary(data)
        safe_result = _convert_to_json_safe(result)
        return {
            "message": "EDA analysis complete",
            "summary": safe_result
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"EDA error: {str(e)}")

@app.post("/train-models")
async def train_models(request: TrainModelsRequest):
    try:
        if session_data['processed_data'] is None:
            raise HTTPException(status_code=400, detail="No processed data available. Process data first.")
        processed_data = session_data['processed_data']
        if 'X_train' not in processed_data or 'y_train' not in processed_data:
            raise HTTPException(status_code=400, detail="Data not properly split. Ensure target column is specified.")
        X_train = processed_data['X_train']
        y_train = processed_data['y_train']
        models = {}

        # Store validation config for later steps
        session_data['validation_config'] = {
            'test_size': request.test_size,
            'cv_folds': request.cv_folds
        }

        if request.model_type == "classification":
            models = classifier.train_multiple_models(X_train, y_train, request.model_list, n_jobs=request.n_jobs)
        elif request.model_type == "regression":
            models = regressor.train_multiple_models(X_train, y_train, request.model_list, n_jobs=request.n_jobs)
        elif request.model_type == "clustering":
            method_map = {
                'kmeans': 'kmeans_clustering', 'mini_batch_kmeans': 'mini_batch_kmeans',
                'dbscan': 'dbscan_clustering', 'hdbscan': 'hdbscan_clustering',
                'hierarchical': 'hierarchical_clustering', 'spectral': 'spectral_clustering',
                'birch': 'birch_clustering', 'optics': 'optics_clustering',
                'mean_shift': 'mean_shift_clustering',
                'affinity_propagation': 'affinity_propagation',
                'gmm': 'gmm_clustering', 'kmedoids': 'kmedoids_clustering',
                'fuzzy_cmeans': 'fuzzy_cmeans'
            }
            if request.model_list:
                for algo in request.model_list:
                    method_name = method_map.get(algo)
                    if method_name:
                        try:
                            result = getattr(clusterer, method_name)(X_train)
                            models[algo] = result
                        except Exception as e:
                            print(f"Clustering {algo} failed: {e}")
        else:
            raise HTTPException(status_code=400, detail="Invalid model_type. Use 'classification', 'regression', or 'clustering'.")
        session_data['trained_models'] = models
        trained_names = [name for name, model in models.items() if model is not None]
        return {
            "message": f"Trained {len(trained_names)} {request.model_type} models successfully",
            "trained_models": trained_names,
            "model_type": request.model_type,
            "test_size": request.test_size,
            "cv_folds": request.cv_folds
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error training models: {str(e)}")

@app.post("/evaluate-models")
async def evaluate_models(request: ModelEvaluationRequest):
    try:
        if not session_data['trained_models']:
            raise HTTPException(status_code=400, detail="No trained models available.")
        if session_data['processed_data'] is None:
            raise HTTPException(status_code=400, detail="No test data available.")
        processed_data = session_data['processed_data']
        if 'X_test' not in processed_data or 'y_test' not in processed_data:
            raise HTTPException(status_code=400, detail="Test data not available.")
        X_test = processed_data['X_test']
        y_test = processed_data['y_test']
        models = session_data['trained_models']
        evaluation_results = {}

        if request.model_type == "classification":
            eval_models = {k: v for k, v in models.items() if hasattr(v, 'predict') and not k.startswith('clustering_')}
            if eval_models:
                evaluation_results = evaluator.evaluate_classification_models(eval_models, X_test, y_test)
                # Add confusion matrix and ROC data for each model
                from sklearn.metrics import confusion_matrix, roc_curve
                for name, model in eval_models.items():
                    if name not in evaluation_results:
                        continue
                    try:
                        y_pred = model.predict(X_test)
                        y_test_arr = np.asarray(y_test)
                        cm = confusion_matrix(y_test_arr, y_pred).tolist()
                        evaluation_results[name]['confusion_matrix'] = cm
                        if hasattr(model, 'predict_proba') and len(set(y_test_arr)) == 2:
                            y_prob = model.predict_proba(X_test)
                            if y_prob.shape[1] == 2:
                                fpr, tpr, _ = roc_curve(y_test_arr, y_prob[:, 1])
                                step = max(1, len(fpr) // 100)
                                evaluation_results[name]['roc_curve'] = {
                                    'fpr': fpr[::step].tolist(),
                                    'tpr': tpr[::step].tolist()
                                }
                    except Exception as e:
                        print(f"Error adding charts for {name}: {e}")
                        pass
        elif request.model_type == "regression":
            eval_models = {k: v for k, v in models.items() if hasattr(v, 'predict')}
            if eval_models:
                evaluation_results = evaluator.evaluate_regression_models(eval_models, X_test, y_test)
                # Add residual data for each model
                for name, model in eval_models.items():
                    try:
                        y_pred = model.predict(X_test)
                        residuals = (y_test - y_pred).tolist()
                        actual = y_test.tolist()
                        predicted = y_pred.tolist()
                        # Sample to ~200 points
                        step = max(1, len(actual) // 200)
                        evaluation_results[name]['residuals'] = {
                            'actual': actual[::step],
                            'predicted': predicted[::step],
                            'residuals': residuals[::step]
                        }
                    except Exception:
                        pass
        elif request.model_type == "clustering":
            for name, result in models.items():
                if isinstance(result, dict) and 'labels' in result:
                    evaluation_results[name] = {
                        'labels_count': len(set(result['labels'])),
                        'n_clusters': result.get('n_clusters', len(set(result['labels']))),
                        'status': 'trained'
                    }
                    if 'silhouette_score' in result:
                        evaluation_results[name]['silhouette_score'] = result['silhouette_score']
        else:
            raise HTTPException(status_code=400, detail="Invalid model_type.")
        session_data['evaluation_results'] = evaluation_results
        return {
            "message": "Models evaluated successfully",
            "evaluation_results": evaluation_results,
            "model_type": request.model_type
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error evaluating models: {str(e)}")

@app.post("/cross-validate")
async def cross_validate(request: CrossValidationRequest):
    try:
        if not session_data['trained_models'] or request.model_name not in session_data['trained_models']:
            raise HTTPException(status_code=400, detail=f"Model '{request.model_name}' not found.")
        if session_data['processed_data'] is None:
            raise HTTPException(status_code=400, detail="No processed data available.")
        processed_data = session_data['processed_data']
        model = session_data['trained_models'][request.model_name]
        if isinstance(model, dict):
            raise HTTPException(status_code=400, detail="Cross-validation not supported for clustering results.")
        X = pd.concat([processed_data['X_train'], processed_data['X_test']])
        y = pd.concat([processed_data['y_train'], processed_data['y_test']])

        method_map = {
            'k_fold': 'k_fold_cross_validation',
            'stratified': 'stratified_k_fold_cv',
            'group_kfold': 'group_kfold_cv',
        }
        method_name = method_map.get(request.method, 'k_fold_cross_validation')
        if hasattr(cv_tool, method_name):
            cv_method = getattr(cv_tool, method_name)
            cv_results = cv_method(model, X, y, n_splits=request.folds, n_jobs=1)
        else:
            cv_results = cv_tool.k_fold_cross_validation(model, X, y, n_splits=request.folds, n_jobs=1)
        return {
            "message": f"Cross-validation ({request.method}) completed for {request.model_name}",
            "model": request.model_name,
            "method": request.method,
            "folds": request.folds,
            "cv_results": cv_results
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Error in cross-validation: {str(e)}")

@app.post("/tune-hyperparameters")
async def tune_hyperparameters(request: HyperparameterTuningRequest):
    try:
        if session_data['processed_data'] is None:
            raise HTTPException(status_code=400, detail="No processed data available.")
        processed_data = session_data['processed_data']
        if 'X_train' not in processed_data or 'y_train' not in processed_data:
            raise HTTPException(status_code=400, detail="Training data not available.")
        X_train = processed_data['X_train']
        y_train = processed_data['y_train']
        method_map = {
            'optuna': 'optuna', 'grid': 'grid', 'random': 'random',
            'hyperopt': 'hyperopt', 'skopt': 'skopt'
        }
        opt_method = method_map.get(request.method.lower(), 'optuna')

        # Get model class from trained model if available, otherwise use model_name
        model_class = None
        if request.model_name in session_data.get('trained_models', {}):
            model = session_data['trained_models'][request.model_name]
            if not isinstance(model, dict):
                model_class = type(model)

        results = tuner.auto_optimize(
            model_class=model_class,
            model_name=request.model_name,
            X=X_train, y=y_train,
            method=opt_method,
            n_trials=50
        )
        return {
            "message": f"Hyperparameter tuning completed for {request.model_name}",
            "model": request.model_name,
            "method": request.method,
            "tuning_results": results
        }
    except Exception as e:
        import traceback
        err_msg = f"Error in hyperparameter tuning: {str(e)}\n{traceback.format_exc()}"
        print(err_msg)
        raise HTTPException(status_code=400, detail=err_msg)

@app.post("/analyze-features")
async def analyze_features(request: FeatureAnalysisRequest):
    try:
        if not session_data['trained_models']:
            raise HTTPException(status_code=400, detail="No trained models available.")
        if session_data['processed_data'] is None:
            raise HTTPException(status_code=400, detail="No processed data available.")
        models = session_data['trained_models']
        X_train = session_data['processed_data']['X_train']
        y_train = session_data['processed_data']['y_train']
        model_name = request.model_name if request.model_name else list(models.keys())[0]
        if model_name not in models:
            model_name = list(models.keys())[0]
        model = models[model_name]
        if isinstance(model, dict):
            raise HTTPException(status_code=400, detail="Feature importance not supported for clustering results.")
        importance_results = feature_analyzer.analyze_feature_importance(model, X_train, y_train)
        return {
            "message": "Feature importance analysis completed",
            "model_used": model_name,
            "importance_results": importance_results,
            "available_models": list(models.keys())
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in feature analysis: {str(e)}")

@app.post("/interpret-model")
async def interpret_model(request: ModelInterpretRequest):
    try:
        if not session_data['trained_models'] or request.model_name not in session_data['trained_models']:
            raise HTTPException(status_code=400, detail=f"Model '{request.model_name}' not found.")
        if session_data['processed_data'] is None:
            raise HTTPException(status_code=400, detail="No processed data available.")
        model = session_data['trained_models'][request.model_name]
        if isinstance(model, dict):
            raise HTTPException(status_code=400, detail="Model interpretation not supported for clustering.")
        X_train = session_data['processed_data']['X_train']
        y_train = session_data['processed_data']['y_train']
        result = {}
        if request.method == "lime":
            try:
                explanation = interpreter.explain_instance_lime(model, X_train, instance_idx=request.instance_idx)
                result = {"explanation": str(explanation), "method": "lime"}
            except Exception as e:
                result = {"explanation": f"LIME unavailable: {e}", "method": "lime"}
        elif request.method == "shap_summary":
            try:
                shap_values = interpreter.plot_shap_summary(model, X_train)
                result = {"shap_summary": "generated", "method": "shap_summary"}
            except Exception as e:
                result = {"shap_summary": f"SHAP unavailable: {e}", "method": "shap_summary"}
        elif request.method == "pdp":
            try:
                feature_name = X_train.columns[request.feature_idx] if request.feature_idx is not None and request.feature_idx < len(X_train.columns) else X_train.columns[0]
                interpreter.plot_partial_dependence(model, X_train, feature_name)
                result = {"partial_dependence": f"computed for {feature_name}", "method": "pdp"}
            except Exception as e:
                result = {"partial_dependence": f"PDP unavailable: {e}", "method": "pdp"}
        elif request.method == "feature_ranking":
            ranking = interpreter.feature_ranking(model, X_train, methods=['coef', 'importance', 'shap'])
            result = {"feature_ranking": ranking.to_dict() if hasattr(ranking, 'to_dict') else str(ranking), "method": "feature_ranking"}
        elif request.method == "full_report":
            report = interpreter.generate_interpretation_report(model, X_train, y_train)
            result = {"report": str(report), "method": "full_report"}
        return {
            "message": f"Model interpretation ({request.method}) completed for {request.model_name}",
            "model": request.model_name,
            "result": result
        }
    except Exception as e:
        import traceback
        err = f"Error interpreting model: {str(e)}"
        print(err)
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=err)
    finally:
        plt.close('all')

@app.get("/get-models")
async def get_models():
    return {
        "trained_models": list(session_data['trained_models'].keys()),
        "model_count": len(session_data['trained_models'])
    }

@app.get("/get-data-info")
async def get_data_info():
    info = {
        "has_uploaded_data": session_data['uploaded_data'] is not None,
        "has_processed_data": session_data['processed_data'] is not None,
        "trained_models": list(session_data['trained_models'].keys()),
        "has_evaluation_results": session_data['evaluation_results'] is not None
    }
    if session_data['uploaded_data'] is not None:
        info["uploaded_data_shape"] = session_data['uploaded_data'].shape
    if session_data['processed_data'] and 'X_train' in session_data['processed_data']:
        info["processed_data_shape"] = session_data['processed_data']['X_train'].shape
    return info

@app.post("/predict")
async def predict(model_name: str = Form(...), data: str = Form(...)):
    try:
        if model_name not in session_data['trained_models']:
            raise HTTPException(status_code=400, detail=f"Model '{model_name}' not found.")
        model = session_data['trained_models'][model_name]
        if isinstance(model, dict):
            raise HTTPException(status_code=400, detail="Predict not supported for clustering results.")
        import json
        input_data = json.loads(data)
        df = pd.DataFrame([input_data])
        if hasattr(model, 'predict_proba'):
            prediction = model.predict(df)[0]
            probabilities = model.predict_proba(df)[0]
            return {
                "model": model_name,
                "prediction": prediction,
                "probabilities": probabilities.tolist()
            }
        else:
            prediction = model.predict(df)[0]
            return {"model": model_name, "prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error making prediction: {str(e)}")

@app.post("/reset-session")
async def reset_session():
    global session_data
    session_data = {
        'uploaded_data': None,
        'processed_data': None,
        'trained_models': {},
        'evaluation_results': None
    }
    return {"message": "Session reset successfully"}

@app.post("/export-model")
async def export_model(model_name: str = Form(...)):
    try:
        if model_name not in session_data['trained_models']:
            raise HTTPException(status_code=400, detail=f"Model '{model_name}' not found.")
        model = session_data['trained_models'][model_name]
        if isinstance(model, dict):
            raise HTTPException(status_code=400, detail="Export not supported for clustering results.")
        with tempfile.NamedTemporaryFile(delete=False, suffix='.joblib') as tmp:
            joblib.dump(model, tmp.name)
            tmp_path = tmp.name
        return FileResponse(tmp_path, filename=f"{model_name}.joblib", media_type="application/octet-stream")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Export error: {str(e)}")

@app.post("/export-report")
async def export_report():
    try:
        models = list(session_data['trained_models'].keys())
        evals = session_data.get('evaluation_results', {}) or {}
        report_html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>TOOL-BOX Evaluation Report</title>
<style>
body {{ font-family: 'Segoe UI', sans-serif; background: #0a0e1a; color: #e0e8f0; padding: 40px; }}
h1 {{ color: #7dd3fc; }} table {{ border-collapse: collapse; width: 100%; }}
th, td {{ padding: 10px 14px; text-align: left; border-bottom: 1px solid rgba(125,211,252,0.15); }}
th {{ color: #a0b4c4; text-transform: uppercase; font-size: 11px; letter-spacing: 1px; }}
td {{ color: #e0e8f0; }}
</style></head><body>
<h1>TOOL-BOX Evaluation Report</h1>
<p>Trained Models: {', '.join(models) if models else 'None'}</p>
<h2>Metrics</h2>
<table><thead><tr><th>Model</th><th>Metric</th><th>Value</th></tr></thead><tbody>"""
        for model_name, metrics in evals.items():
            if isinstance(metrics, dict):
                for k, v in metrics.items():
                    val = f"{v:.4f}" if isinstance(v, float) else str(v)
                    report_html += f"<tr><td>{model_name}</td><td>{k}</td><td>{val}</td></tr>"
        report_html += "</tbody></table></body></html>"
        with tempfile.NamedTemporaryFile(delete=False, suffix='.html', mode='w', encoding='utf-8') as tmp:
            tmp.write(report_html)
            tmp_path = tmp.name
        return FileResponse(tmp_path, filename="toolbox_report.html", media_type="text/html")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Report error: {str(e)}")

@app.post("/get-recommendation")
async def get_recommendation():
    try:
        if session_data['uploaded_data'] is None:
            raise HTTPException(status_code=400, detail="No data uploaded.")
        data = session_data['uploaded_data']
        info = {}
        for col in data.columns:
            dtype = data[col].dtype
            nunique = data[col].nunique()
            missing = int(data[col].isna().sum())
            info[col] = {
                'dtype': str(dtype),
                'nunique': nunique,
                'missing': missing,
                'missing_pct': round(missing / len(data) * 100, 1),
                'is_numeric': pd.api.types.is_numeric_dtype(data[col]),
                'is_categorical': pd.api.types.is_object_dtype(data[col]) or nunique < 20
            }
        return {"columns": info}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Recommendation error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
