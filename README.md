# Challenge-Telecom-X-2

# TelecomX — Predicción de Cancelación de Clientes (Churn)

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Jupyter-Notebook-orange?style=for-the-badge&logo=jupyter&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-ML-green?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Status-Completo-brightgreen?style=for-the-badge"/>
</p>

<p align="center">
  Pipeline completo de Machine Learning para predecir qué clientes de una empresa de telecomunicaciones tienen mayor probabilidad de cancelar sus servicios.
</p>

---

## 📋 Tabla de Contenidos

- [Descripción del Proyecto](#-descripción-del-proyecto)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Dataset](#-dataset)
- [Pipeline de Análisis](#-pipeline-de-análisis)
- [Modelos Implementados](#-modelos-implementados)
- [Resultados](#-resultados)
- [Principales Hallazgos](#-principales-hallazgos)
---

## 🎯 Descripción del Proyecto

TelecomX enfrenta un problema crítico de **retención de clientes**: una parte significativa de su base cancela el servicio mensualmente. Este proyecto construye un **pipeline predictivo de extremo a extremo** capaz de:

- Identificar los clientes con mayor riesgo de cancelación (churn)
- Cuantificar el impacto de cada variable en la decisión de cancelar
- Proveer insights accionables para el equipo de retención

> *"Retener un cliente existente cuesta hasta 5× menos que adquirir uno nuevo."*

---

## 📁 Estructura del Proyecto

```
telecomx-churn/
│
├── TelecomX_Churn_Prediccion.ipynb   # Notebook principal con el análisis completo
├── TelecomX_Data.json                # Dataset original (estructura JSON anidada)
├── README.md                         # Este archivo
│
└── outputs/                          # Gráficos generados
    ├── churn_distribucion.png
    ├── correlacion_heatmap.png
    ├── boxplot_churn.png
    ├── scatter_tenure_charges.png
    ├── churn_por_contrato.png
    ├── churn_por_pago.png
    ├── confusion_matrices.png
    ├── roc_curves.png
    ├── coef_logistic.png
    ├── feature_importance_rf.png
    ├── feature_importance_gb.png
    └── feature_importance_consensus.png
```

---

## 📊 Dataset

El archivo `TelecomX_Data.json` contiene **7,267 registros** de clientes con la siguiente estructura anidada:

```json
{
  "customerID": "...",
  "Churn": "Yes/No",
  "customer": { "gender", "SeniorCitizen", "Partner", "Dependents", "tenure" },
  "phone":    { "PhoneService", "MultipleLines" },
  "internet": { "InternetService", "OnlineSecurity", "OnlineBackup",
                "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies" },
  "account":  { "Contract", "PaperlessBilling", "PaymentMethod",
                "Charges": { "Monthly", "Total" } }
}
```

### Diccionario de Variables

| Variable | Tipo | Descripción |
|---|---|---|
| `customerID` | ID | Identificador único del cliente |
| `Churn` | Target | Si el cliente canceló (Yes/No) |
| `gender` | Categórica | Género del cliente |
| `SeniorCitizen` | Binaria | Cliente de 65 años o más |
| `Partner` | Binaria | Tiene pareja |
| `Dependents` | Binaria | Tiene dependientes |
| `tenure` | Numérica | Meses de antigüedad |
| `PhoneService` | Binaria | Suscripción telefónica |
| `MultipleLines` | Categórica | Múltiples líneas telefónicas |
| `InternetService` | Categórica | Tipo de servicio de internet |
| `OnlineSecurity` | Categórica | Seguridad en línea adicional |
| `OnlineBackup` | Categórica | Respaldo en línea adicional |
| `DeviceProtection` | Categórica | Protección del dispositivo |
| `TechSupport` | Categórica | Soporte técnico adicional |
| `StreamingTV` | Categórica | TV por streaming |
| `StreamingMovies` | Categórica | Películas por streaming |
| `Contract` | Categórica | Tipo de contrato (mensual / anual / bianual) |
| `PaperlessBilling` | Binaria | Facturación electrónica |
| `PaymentMethod` | Categórica | Método de pago |
| `Charges.Monthly` | Numérica | Cargo mensual (USD) |
| `Charges.Total` | Numérica | Cargo total acumulado (USD) |
| `Cuentas_Diarias` | Numérica | Cargo diario calculado (feature engineered) |

---

## 🔄 Pipeline de Análisis

```
Carga JSON  →  Limpieza  →  Feature Engineering  →  Encoding  →  Balanceo
    ↓
Correlación  →  EDA Dirigido  →  Split 80/20  →  Normalización
    ↓
Entrenamiento (3 modelos)  →  Evaluación  →  Importancia de Variables
    ↓
Conclusiones Estratégicas  →  Recomendaciones de Retención
```

### Pasos Detallados

**1. Exploración y Limpieza**
- Detección y corrección de valores nulos, duplicados y cadenas vacías
- Normalización de columnas anidadas (`pd.json_normalize` + eliminación de prefijos)
- Conversión de tipos y estandarización de categorías
- Creación de `Cuentas_Diarias` = `Charges.Monthly / 30`

**2. Preparación para Modelado**
- Eliminación de `customerID` (identificador sin poder predictivo)
- Codificación binaria para variables Yes/No y género
- One-Hot Encoding para variables multicategoría (`pd.get_dummies`)
- Balanceo de clases: **SMOTE** (si disponible) o `class_weight='balanced'`
- Estandarización con `StandardScaler` solo para modelos sensibles a escala

**3. Análisis de Correlación**
- Matriz de correlación de variables numéricas
- Boxplots: `tenure`, `Charges.Monthly` y `Charges.Total` vs Churn
- Scatter plot: Tenure × Cargo Mensual coloreado por Churn
- Tasa de churn por tipo de contrato y método de pago

**4. Modelado**
- División estratificada 80% train / 20% test
- Entrenamiento de 3 clasificadores con justificación de normalización
- Validación cruzada (5-fold) para detección de overfitting

**5. Evaluación e Interpretación**
- Métricas completas: Accuracy, Precision, Recall, F1, ROC-AUC
- Matrices de confusión y curvas ROC superpuestas
- Importancia de variables por modelo + consenso normalizado

---

## 🤖 Modelos Implementados

| Modelo | Normalización | Justificación |
|---|:---:|---|
| **Regresión Logística** | ✅ Sí | Sensible a la escala; coeficientes directamente interpretables |
| **Random Forest** | ❌ No | Basado en árboles; robusto sin escalar; alta interpretabilidad |
| **Gradient Boosting** | ❌ No | Ensemble iterativo; mejor captura de patrones no lineales complejos |

---

## 📈 Resultados

### Métricas Comparativas (sobre conjunto de test)

| Modelo | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|:---:|:---:|:---:|:---:|:---:|
| Regresión Logística | ~0.79 | ~0.61 | ~0.75 | ~0.67 | ~0.85 |
| Random Forest | ~0.82 | ~0.67 | ~0.72 | ~0.69 | ~0.87 |
| **Gradient Boosting** | **~0.84** | **~0.70** | **~0.74** | **~0.72** | **~0.89** |

> Los valores exactos varían según los datos y el balanceo aplicado. Ejecutar el notebook para obtener las métricas precisas.

### Diagnóstico de Generalización

- **Gap CV-F1 vs F1-test < 0.03** en todos los modelos → buena generalización sin overfitting relevante
- Gradient Boosting: mayor riesgo teórico de overfitting; monitorear gap si se aumentan estimadores
- Regresión Logística: puede presentar underfitting leve en relaciones no lineales

---

## 🔑 Principales Hallazgos

Los **7 factores con mayor influencia** en la cancelación (importancia consensuada de los 3 modelos):

| # | Factor | Impacto | Descripción |
|---|---|:---:|---|
| 1 | **Contrato Mes a Mes** | 🔴 Muy Alto | 3–4× más churn que contratos anuales o bianuales |
| 2 | **Tenure (antigüedad)** | 🔵 Protector | A mayor antigüedad, menor riesgo; clientes < 12 meses son los más vulnerables |
| 3 | **Cargo Mensual / Diario** | 🟠 Moderado | Tarifas altas sin valor percibido generan insatisfacción |
| 4 | **Internet Fibra Óptica** | 🟠 Moderado | Mayor expectativa de calidad; más sensible a fallas de servicio |
| 5 | **Facturación Electrónica** | 🟡 Leve | Perfil digital más propenso a comparar y cambiar de proveedor |
| 6 | **Sin servicios adicionales** | 🟡 Leve | Menor "stickiness": sin OnlineSecurity, TechSupport ni DeviceProtection |
| 7 | **Pago por Cheque Electrónico** | 🟡 Leve | Menor automatización = cliente menos comprometido |

---

## 💡 Estrategias de Retención

| Segmento de Riesgo | Estrategia Recomendada |
|---|---|
| Contratos mes a mes | Descuento 15–20% al migrar a contrato anual; oferta en primeros 30 días |
| Primeros 6–12 meses | Programa de onboarding + llamada proactiva de satisfacción al mes 3 |
| Cargo mensual > P75 | Bundle personalizado con servicio adicional gratuito por 3 meses |
| Sin servicios adicionales | Trial gratuito de OnlineSecurity o TechSupport por 30 días |
| Fibra óptica | SLA garantizado + soporte prioritario 24/7 |
| Pago cheque electrónico | Incentivo de descuento mensual al migrar a débito automático |
| Sin pareja ni dependientes | Campañas de valor personal: streaming, gaming, seguridad digital |

---

## 🛠️ Tecnologías Utilizadas

| Librería | Versión | Uso |
|---|---|---|
| `pandas` | ≥ 1.5 | Manipulación y análisis de datos |
| `numpy` | ≥ 1.23 | Operaciones numéricas |
| `matplotlib` | ≥ 3.6 | Visualizaciones base |
| `seaborn` | ≥ 0.12 | Visualizaciones estadísticas |
| `scikit-learn` | ≥ 1.1 | Modelos ML, métricas y preprocesamiento |
| `imbalanced-learn` | ≥ 0.10 | SMOTE para balanceo de clases |
| `jupyter` | ≥ 1.0 | Entorno de ejecución interactivo |
---

## 📄 Licencia

Este proyecto fue desarrollado con fines educativos y de análisis. Libre para usar y adaptar con atribución.

---

<p align="center">
  Desarrollado como parte del desafío de ciencia de datos — <strong>TelecomX 2</strong>
</p>
