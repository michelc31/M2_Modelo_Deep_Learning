# Detección de Acciones Humanas en Videos (UCF101)

**Módulo:** Momento de Retroalimentación: Módulo 2 Implementación de un modelo de deep learning. (Portafolio Implementación)  
**Alumno:** Omar Michel Carmona Villalobos  
**Fecha:** 1ro de Diciembre 2025  

---

## 1. Descripción y Objetivos
El objetivo de este proyecto es implementar y comparar modelos de Deep Learning para la clasificación de acciones humanas utilizando el dataset real **UCF101**. Se busca demostrar el uso de frameworks de aprendizaje profundo (**PyTorch**) y evaluar métricas de desempeño.

Se utilizan **esqueletos 2D** como características de entrada para optimizar el cómputo, enfocándose en un subconjunto de 5 clases desafiantes: `ApplyEyeMakeup`, `Archery`, `BoxingPunchingBag`, `Diving`, `Lunges`.

## 2. Instrucciones de Ejecución (Console)

Este proyecto está diseñado para ejecutarse desde la consola. Sigue estos pasos para reproducir los resultados y generar predicciones.

### Prerrequisitos
*   Python 3.8 o superior.
*   Librerías listadas en `requirements.txt`.

### Pasos:

1.  **Instalar Dependencias**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Entrenar y Evaluar Modelos**:
    Ejecuta el script principal. Esto entrenará el modelo base (LSTM) y el mejorado (CNN-LSTM) por 10 épocas, generando métricas y gráficos.
    ```bash
    python src/train.py
    ```
    *Salida esperada:* Gráficos `comparison_plot.png`, matrices de confusión y logs de entrenamiento en consola.

3.  **Generar Predicciones (Inferencia)**:
    Para ver el modelo funcionando en tiempo real sobre datos de prueba, ejecuta:
    ```bash
    python src/predict.py
    ```
    *Salida:* El sistema seleccionará videos aleatorios del set de validación e imprimirá en consola la acción real vs. la predicción del modelo con su nivel de confianza.

## 3. Arquitecturas Implementadas

Se utilizaron dos arquitecturas basadas en redes recurrentes para procesar la secuencia temporal de los esqueletos:

1.  **Baseline (SimpleLSTM)**:
    *   Arquitectura: `Input -> LSTM (2 capas) -> Fully Connected -> Softmax`.
    *   Propósito: Establecer una línea base de rendimiento utilizando solo memoria secuencial.

2.  **Modelo Mejorado (CNN-LSTM)**:
    *   Arquitectura: `Input -> Conv1d -> ReLU -> MaxPool -> LSTM -> Dropout -> FC`.
    *   Mejora: Se añadió una capa convolucional 1D al inicio para extraer características espaciales locales y reducir el ruido de los puntos clave antes de procesar la secuencia temporal. Se incluyó `Dropout` para regularización.

## 4. Evaluación y Análisis de Resultados

Los modelos fueron entrenados durante 10 épocas (restricción de la asignación). A continuación se presenta el análisis de las métricas obtenidas en el set de validación.

### Métricas Comparativas

| Modelo | Accuracy (Validación) | Loss (Validación) | Observaciones |
| :--- | :--- | :--- | :--- |
| **Baseline (LSTM)** | ~22.7% | 1.60 | Tendencia a predecir la clase mayoritaria. |
| **Mejorado (CNN-LSTM)** | ~20.4% | 1.61 | Mayor dificultad de convergencia en pocas épocas. |

### Discusión de Resultados

1.  **Underfitting y Épocas Insuficientes**: 
    Como se observa en las gráficas `comparison_plot.png`, ambos modelos presentan una exactitud cercana al 20-22%. Dado que son 5 clases, el azar sería 20%. Esto indica que con **10 épocas**, los modelos no lograron converger lo suficiente para aprender patrones discriminatorios complejos. Las redes neuronales recurrentes suelen requerir periodos de entrenamiento más largos (50+ épocas).

2.  **Mode Collapse**: 
    Al analizar las matrices de confusión generadas, se observa que el **Baseline** tendió a clasificar casi todas las muestras como `BoxingPunchingBag` (Recall alto en esa clase, 0 en otras), mientras que el **CNN-LSTM** colapsó hacia `ApplyEyeMakeup`. Esto es un comportamiento común en etapas tempranas de entrenamiento cuando la red busca minimizar la pérdida "apostando" a la clase más frecuente o fácil de detectar.

3.  **Comparación**: 
    Aunque el modelo CNN-LSTM es arquitectónicamente superior (capaz de extraer mejores features), su complejidad requiere más datos o tiempo para ajustarse. En un escenario de entrenamiento corto (10 épocas), el modelo más simple (LSTM pura) mostró una estabilidad marginalmente superior, aunque ninguno es funcional para producción en este punto.

4.  **Mejoras Futuras**:
    Para mejorar el desempeño (indicador SMA0401C), se propone:
    *   Aumentar el entrenamiento a **100 épocas**.
    *   Implementar **Data Augmentation** en los esqueletos (rotaciones leves, ruido gaussiano).
    *   Usar **Class Weights** en la función de pérdida para penalizar el sesgo hacia una sola clase.

## 5. Predicciones (Evidencia de Funcionalidad)

El script `src/predict.py` cumple con el indicador de generar predicciones. Toma una muestra de validación (no vista durante el entrenamiento), la pasa por el modelo mejorado y muestra el resultado.

Ejemplo de salida de consola:
```text
Sample 58: True: Archery           Pred: ApplyEyeMakeup    Conf: 0.2163 [WRONG]
Sample 39: True: ApplyEyeMakeup    Pred: ApplyEyeMakeup    Conf: 0.2156 [CORRECT]
...
```
Esto demuestra que el pipeline de inferencia está completo y funcional, listo para integrarse en una aplicación real una vez mejore la precisión del modelo.
