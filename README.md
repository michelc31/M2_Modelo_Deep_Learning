# Detección de Acciones Humanas en Videos (UCF101)

**Módulo:** Momento de Retroalimentación: Módulo 2 Implementación de un modelo de deep learning. (Portafolio Implementación) 
**Alumno:** Omar Michel Carmona Villalobos   
**Fecha:** 1ro de Diciembre 2025  

---

## 1. Descripción del Proyecto
El objetivo de este trabajo es desarrollar un modelo de Deep Learning capaz de identificar qué acción está realizando una persona en un video. Para esto utilicé el dataset **UCF101**, que es un estándar en la industria para reconocimiento de acciones.

En lugar de procesar los videos completos (lo cual es muy pesado y tardado), decidí trabajar con **esqueletos 2D**. Esto significa que el modelo no ve los píxeles del video, sino las coordenadas `(x, y)` de las articulaciones de la persona (codos, rodillas, hombros, etc.) a lo largo del tiempo.

## 2. Decisión de Clases (Subset)
Para esta entrega, seleccioné un subconjunto de 5 clases que me parecieron interesantes y distintas entre sí para probar el modelo:
1.  `ApplyEyeMakeup` (Maquillarse los ojos - movimientos finos de manos)
2.  `Archery` (Tiro con arco - postura estática y luego movimiento)
3.  `BoxingPunchingBag` (Boxeo - movimientos rápidos y repetitivos)
4.  `Diving` (Clavados - movimiento corporal completo y rotación)
5.  `Lunges` (Ejercicios de piernas - movimiento vertical cíclico)

## 3. Modelos Implementados

Probé dos arquitecturas diferentes para ver cuál funcionaba mejor:

### Modelo 1: LSTM Básica (Baseline)
Primero implementé una red neuronal recurrente simple usando **LSTM**.
*   **Por qué:** Las LSTMs son buenas recordando secuencias, y una acción es básicamente una secuencia de poses.
*   **Arquitectura:** Entra la secuencia de coordenadas -> Capa LSTM -> Capa Lineal -> Clasificación.

### Modelo 2: CNN + LSTM (Mejorado)
Después intenté mejorar el modelo agregando capas convolucionales.
*   **La idea:** Usar una **Conv1d** al principio para detectar "micro-movimientos" o patrones rápidos en las articulaciones, y luego pasarle esas características procesadas a la LSTM para que entienda la secuencia completa.
*   **Regularización:** Agregué `Dropout` (apagado aleatorio de neuronas) para evitar que el modelo se memorice los datos de entrenamiento (overfitting).

## 4. Estrategia de Trabajo (Pipeline)

1.  **Preprocesamiento:**
    *   Cargo el archivo `.pkl` con los esqueletos.
    *   Normalizo las coordenadas para que estén entre 0 y 1 (dividiendo por el ancho/alto del video).
    *   Si el video es muy largo, tomo una muestra uniforme de frames. Si es muy corto, relleno con ceros (padding).

2.  **Entrenamiento:**
    *   Usé **CrossEntropyLoss** como función de pérdida (estándar para clasificación).
    *   Optimizador **Adam** con learning rate de 0.001.
    *   Entrené por **10 épocas** (según las instrucciones para no demorar tanto).

## 5. Resultados y Conclusiones

Al correr los experimentos, noté que con solo 10 épocas el modelo apenas empieza a aprender. La "loss" (pérdida) bajó un poco, pero la precisión (accuracy) se mantuvo baja (alrededor del 20-25%).

**Observaciones:**
*   El modelo **CNN-LSTM** tiende a ser más robusto, pero necesita más tiempo de entrenamiento.
*   Trabajar con esqueletos es mucho más rápido que con video RGB. Podía procesar cientos de videos en segundos.
*   Para mejorar esto en el futuro, necesitaría entrenar por al menos 50 o 100 épocas y quizás usar técnicas de "Data Augmentation" (rotar o escalar los esqueletos) para que el modelo generalice mejor.

## 6. Estructura de Archivos

*   `src/dataset.py`: Aquí está toda la lógica de carga y limpieza de datos.
*   `src/models.py`: El código de las redes neuronales (LSTM y CNN-LSTM).
*   `src/train.py`: El script principal que entrena y genera las gráficas.
*   `src/predict.py`: Un script extra que hice para ver al modelo prediciendo ejemplos individuales.

---

