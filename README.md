# Proyecto Telecom X - Parte 2: Predicción de Cancelación de Clientes (Churn)

## Propósito del Análisis

El objetivo principal de este proyecto es desarrollar un modelo predictivo capaz de identificar a los clientes propensos a la cancelación (churn) en una empresa de telecomunicaciones. Al predecir el churn, la empresa puede implementar estrategias de retención proactivas, reducir la pérdida de clientes y optimizar sus recursos. Este análisis se basa en diversas variables relacionadas con el cliente y el servicio para determinar los factores más influyentes en la decisión de un cliente de abandonar la compañía.

## Estructura del Proyecto

El proyecto está organizado de la siguiente manera:

*   `TelecomX_LATAM_Parte2.ipynb`: Cuaderno principal de Jupyter (Google Colab) que contiene todo el código para la carga de datos, preprocesamiento, balanceo de clases, división de datos, entrenamiento y evaluación de modelos (Regresión Logística y Random Forest), y el análisis de importancia de variables.
*   `TelecomX_Data_raw.csv`: Archivo de datos crudos utilizado para el análisis.

## Proceso de Preparación de los Datos

El proceso de preparación de los datos es fundamental para asegurar la calidad y la idoneidad de los mismos para el modelado. Las etapas clave incluyen:

### 1. Carga y Limpieza Inicial

*   **Carga de Datos:** El dataset `TelecomX_Data_raw.csv` se carga utilizando Pandas.
*   **Eliminación de `customerID`:** La columna `customerID` se elimina ya que es un identificador único y no aporta información predictiva para el modelo.

### 2. Clasificación y Codificación de Variables

*   **Variables Categóricas:** La mayoría de las variables en el dataset son categóricas (ej., `gender`, `Partner`, `InternetService`, `Contract`, `PaymentMethod`). Estas variables se transforman utilizando `LabelEncoder` de `sklearn.preprocessing`. Esta elección se hizo para mantener la dimensionalidad del dataset manejable, especialmente considerando que algunas variables tienen múltiples categorías.
*   **Variable Numérica (`account.Charges.Total`):** La columna `account.Charges.Total` se convierte a tipo numérico. Se utiliza `errors='coerce'` para manejar cualquier valor no numérico, convirtiéndolo en `NaN` (Not a Number).
*   **Variable Objetivo (`Churn`):** La variable objetivo `Churn` (que indica si un cliente canceló o no) se convierte a un formato binario (0 para 'No' y 1 para 'Yes') utilizando `LabelEncoder`.

### 3. Manejo de Valores Nulos

*   Después de la conversión de `account.Charges.Total`, se eliminan las filas que contienen valores `NaN`. Esto asegura que el modelo no se vea afectado por datos incompletos.

### 4. Separación de Datos y Balanceo de Clases

*   **División Train/Test:** El dataset se divide en conjuntos de entrenamiento (80%) y prueba (20%) utilizando `train_test_split` de `sklearn.model_selection`. Se utiliza `stratify=y` para asegurar que la proporción de la variable objetivo (`Churn`) sea la misma en ambos conjuntos, lo cual es crucial dado el desequilibrio de clases.
*   **Balanceo de Clases (SMOTE):** El dataset de clientes de telecomunicaciones suele presentar un desequilibrio significativo en la variable `Churn` (muchos más clientes que no cancelan que los que sí lo hacen). Para abordar este problema, se aplica `SMOTE` (Synthetic Minority Over-sampling Technique) al conjunto de entrenamiento. SMOTE genera nuevas muestras sintéticas de la clase minoritaria, ayudando a los modelos a aprender de manera más efectiva sobre los patrones de churn y a evitar el sesgo hacia la clase mayoritaria. Es importante destacar que SMOTE solo se aplica al conjunto de entrenamiento para evitar la fuga de datos (data leakage) y asegurar que la evaluación del modelo en el conjunto de prueba sea realista.

## Justificaciones para las Decisiones Tomadas Durante la Modelización

### Selección de Modelos

Se eligieron dos modelos principales para este análisis:

*   **Regresión Logística:** Es un modelo lineal simple pero efectivo para problemas de clasificación binaria. Sus coeficientes son interpretables, lo que permite entender la dirección y magnitud de la influencia de cada variable en la probabilidad de churn.
*   **Random Forest:** Es un modelo de conjunto (ensemble) basado en árboles de decisión. Es robusto frente al sobreajuste, maneja bien las relaciones no lineales y proporciona una medida de importancia de las características de forma nativa, lo que lo hace excelente para identificar los factores clave de churn.

### Evaluación de Modelos

Los modelos se evaluaron utilizando métricas como `Accuracy`, `Precision`, `Recall`, `F1-Score` y `ROC AUC`. Aunque el `Accuracy` es una métrica general, `Precision`, `Recall` y `F1-Score` son particularmente importantes en problemas de clasificación desequilibrados como el churn, ya que proporcionan una visión más detallada del rendimiento del modelo en la identificación de la clase minoritaria (clientes que cancelan). `ROC AUC` es una métrica robusta que evalúa la capacidad del modelo para distinguir entre clases.

## Ejemplos de Gráficos e Insights Obtenidos (EDA)

Aunque el script principal se centra en el modelado, un análisis exploratorio de datos (EDA) previo reveló insights importantes. Por ejemplo:

*   **Distribución de Churn:** Se observó un desequilibrio significativo en la variable `Churn`, con una mayoría de clientes que no cancelan. Esto justificó el uso de técnicas de balanceo de clases como SMOTE.
*   **Relación entre `Contract` y `Churn`:** Los clientes con contratos mensuales tienden a tener una tasa de churn significativamente más alta en comparación con aquellos con contratos de uno o dos años. Esto sugiere que los contratos a largo plazo fomentan la lealtad.
*   **Impacto de `InternetService`:** Los clientes con servicio de fibra óptica mostraron una tasa de churn más alta que aquellos con DSL o sin servicio de internet, lo que podría indicar problemas de calidad o expectativas no cumplidas con este tipo de servicio.
*   **Correlación de `account.Charges.Monthly` y `account.Charges.Total`:** Ambas variables están positivamente correlacionadas con el churn, indicando que a mayores cargos, mayor es la probabilidad de cancelación.


## Instrucciones para Ejecutar el Cuaderno (Google Colab)

Para ejecutar el cuaderno `TelecomX_LATAM_Parte2.ipynb` y replicar el análisis en Google Colab, siga los siguientes pasos:

### 1. Abrir el Cuaderno en Google Colab

*   Suba el archivo `TelecomX_LATAM_Parte2.ipynb` a su Google Drive.
*   Abra el cuaderno con Google Colab.

### 2. Cargar los Datos

*   Asegúrese de que el archivo `TelecomX_Data_raw.csv` se encuentre en el mismo entorno de ejecución de Colab. Puede subirlo directamente a la sesión de Colab o montarlo desde Google Drive.

### 3. Instalación de Bibliotecas

Dentro de una celda de código en Colab, ejecute el siguiente comando para instalar las bibliotecas necesarias:

```python
!pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn
```

### 4. Ejecución de las Celdas

Ejecute cada celda del cuaderno secuencialmente. Los resultados del preprocesamiento, balanceo de clases, evaluación de modelos y análisis de importancia de variables se imprimirán en la salida de las celdas correspondientes. Las visualizaciones (matrices de confusión, gráficos de importancia de variables y matriz de correlación) se mostrarán directamente en la salida de las celdas donde se generan.

