import pandas as pd
import warnings
from sklearn.preprocessing import StandardScaler
import json
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import mysql.connector
import ast
import logging
from sklearn.pipeline import Pipeline

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

if __name__ == "__main__": 

    # Registro de experimento en MLFlow
    mlflow.set_tracking_uri('http://localhost:5000')
    experiment = mlflow.set_experiment("tiempos_caja_polinomial_regression_experiment")

    print("mlflow tracking uri:", mlflow.tracking.get_tracking_uri())
    print("experiment:", experiment)
    warnings.filterwarnings("ignore")

    # Cargar el dataset
    # Configura conexión con base de datos de MYSql alojada en GCP
    db_connection = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Test123@",
    database="DB_ZAB"
    )
    cursor = db_connection.cursor()

    # Definimos una sucursal objetivo para nuestro análisis
    suc_objetivo = 'ZCL'

    # Query de consulta, el tamaño actual de registros para la sucursal ZCL es de 850 registros
    query = f"SELECT vch_identifier, txt_prediction, TXT_Real,INT_CheckoutsReal, FLT_TimeAttentionReal, DTT_Date FROM TBL_Snapshots where vch_identifier = '{suc_objetivo}' order by DTT_Date;"

    # Cargamos los datos en un DataFrame para realizar el análisis a continuación
    data = pd.read_sql(query, con=db_connection)
    db_connection.close()

    # Procesamiento de datos
    # Se procesan los datos a formato JSON para nuestro objeto TXT_Real
    data.TXT_Real = data.TXT_Real.apply(lambda x: json.loads(x) if isinstance(x, str) else x)

    # Se procesan los datos a formato JSON para nuestro objeto txt_prediction
    data.txt_prediction = data.txt_prediction.apply(lambda x: json.loads(x) if isinstance(x, str) else x)

    # Se extraen las características Llegadas y checkout por tiempo
    data['checkouts_by_time'] = data.TXT_Real.apply(lambda x: x['checkouts'] if x is not None else None)
    data['arrives_by_time'] = data.TXT_Real.apply(lambda x: x['arrives'] if x is not None else None)

    # Se extraen las característicast tiempo de servicio, costo promedio y tiempo de espera
    data['costs_avg_by_time'] = data.txt_prediction.apply(lambda x: x['graphs']['costs'][0])
    data['services_by_time'] = data.txt_prediction.apply(lambda x: x['graphs']['services'][0])
    data['waitings_by_time'] = data.txt_prediction.apply(lambda x: x['graphs']['waitings'][0])

    # Renombramos las columnas para facilitar su interpretación
    data = data.rename(columns={
        'vch_identifier': 'store', # Identificador de 3 letras de la sucursal
        'FLT_TimeAttentionReal': 'avg_real_time_service', # Tiempo promedio de servicio real
        'DTT_Date': 'date' # Fecha de la toma de datos
    })

    # Eliminamos las columnas que no son necesarias para nuestro análisis
    data = data.drop(columns=['txt_prediction','TXT_Real','INT_CheckoutsReal'])

    # Identificar las columnas con objetos JSON
    json_columns = ['services_by_time', 'arrives_by_time', 'checkouts_by_time', 'waitings_by_time', 'costs_avg_by_time']

    # Limpiar los valores NaN y valores float no convertibles
    data['arrives_by_time'] = data['arrives_by_time'].apply(lambda x: '{}' if isinstance(x, float) else x)
    data['checkouts_by_time'] = data['checkouts_by_time'].apply(lambda x: '{}' if isinstance(x, float) else x)

    # Convertir las columnas JSON de string a objetos
    for col in json_columns:
        data[col] = data[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # Expandimos las columnas JSON en columnas individuales
    data_clean = data.copy()
    for col in json_columns:
        if isinstance(data_clean[col][0], list):
            expanded_df = pd.DataFrame(data_clean[col].tolist(), index=data_clean.index)
            expanded_df.columns = [f"{col}_{i}" for i in range(len(expanded_df.columns))]
        else:
            expanded_df = pd.json_normalize(data_clean[col])
            expanded_df.columns = [f"{col}_{hour}" for hour in expanded_df.columns]
        data_clean = pd.concat([data_clean.drop(columns=[col]), expanded_df], axis=1)


    # Agrupación de datos

    data_clean_num = data_clean.apply(pd.to_numeric, errors='coerce')
    # Agrupar las columnas por categoría y calcular el promedio
    grouped_averages = {
        'services_by_time': data_clean_num.filter(regex='^services_by_time').mean(axis=1),
        'arrives_by_time': data_clean_num.filter(regex='^arrives_by_time').mean(axis=1),
        'checkouts_by_time': data_clean_num.filter(regex='^checkouts_by_time').mean(axis=1),
        'waitings_by_time': data_clean_num.filter(regex='^waitings_by_time').mean(axis=1),
        'costs_avg_by_time': data_clean_num.filter(regex='^costs_avg_by_time').mean(axis=1)
    }
    # Convertir a DataFrame y añadir las columnas 'store', 'date' y 'avg_real_time_service'
    data_group = pd.concat([data_clean[['store', 'date', 'avg_real_time_service']], pd.DataFrame(grouped_averages)], axis=1)
    data_group['date'] = pd.to_datetime(data_group['date']).dt.strftime('%Y-%m-%d')

    # Imputar valores faltantes con la media de cada columna
    columns_with_missing = data_group.columns[data_group.isnull().any()].tolist()
    data_group[columns_with_missing] = data_group[columns_with_missing].fillna(data_group[columns_with_missing].mean())

    # Normalización de datos
    
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_group[["services_by_time", "arrives_by_time", "checkouts_by_time", "waitings_by_time", "costs_avg_by_time"]])
    data_scaled = pd.DataFrame(data_scaled, columns=["services_by_time", "arrives_by_time", "checkouts_by_time", "waitings_by_time", "costs_avg_by_time"])
    data_scaled.head()

    data_scaled = data_group.copy()
    scaler = StandardScaler()
    data_scaled['services_by_time'] = scaler.fit_transform(data_scaled[['services_by_time']])
    data_scaled['arrives_by_time'] = scaler.fit_transform(data_scaled[['arrives_by_time']])
    data_scaled['checkouts_by_time'] = scaler.fit_transform(data_scaled[['checkouts_by_time']])
    data_scaled['waitings_by_time'] = scaler.fit_transform(data_scaled[['waitings_by_time']])
    data_scaled['costs_avg_by_time'] = scaler.fit_transform(data_scaled[['costs_avg_by_time']])

    # Modelado de datos

    # Dividir los datos en conjuntos de entrenamiento y prueba
    df_X = data_scaled[['services_by_time', 'arrives_by_time', 'waitings_by_time', 'costs_avg_by_time']]
    df_y = data_scaled['checkouts_by_time']

    # Separación de pruebas y entrenamiento con un 80% de datos de entrenamiento y 20% de datos de prueba
    X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.2, random_state=42)

    # Modelado con algoritmo Random Forest
    with mlflow.start_run(experiment_id=experiment.experiment_id):

        param_grid = {
            'polynomialfeatures__degree': [2, 3, 4, 5],  # Ajustar el grado del polinomio
            'linearregression__fit_intercept': [True, False]  # Ajuste de la intersección
        }

        pipeline = Pipeline([
            ('polynomialfeatures', PolynomialFeatures()),
            ('linearregression', LinearRegression())
        ])
        
        model = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)

        # Entrenar el modelo con los datos de entrenamiento
        model.fit(X_train, y_train)
        
        # Make predictions and evaluate the model
        y_pred = model.predict(X_test)

        # Calcular las métricas de regresión
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Log parameters and metrics to MLFlow
        mlflow.log_param("model_type", "Regresion Polinomial")
        mlflow.log_param("max_iter", 100)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        
        # Log the model
        model_info = mlflow.sklearn.log_model(model, "model")
        sklearn_pyfunc = mlflow.pyfunc.load_model(model_uri=model_info.model_uri)
        print(f"Model performance - MSE: {mse}, MAE: {mae}, R²: {r2}")