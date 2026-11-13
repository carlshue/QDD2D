# QDD2D — Quadtree-based Dynamic Diffusion (2D)

> Prototipo de investigación: **modelos de difusión que generan imágenes sobre una representación jerárquica adaptativa (quadtree)** en lugar de sobre una rejilla fija de píxeles, como paso previo a la extensión 3D con **octrees**.

---

## 1. Objetivo

Los modelos de difusión actuales (DDPM, LDM, DiT…) gastan **el mismo cómputo en cada píxel / token**, aunque gran parte de la imagen sea uniforme (fondo negro en MNIST, cielo, paredes…). QDD2D investiga si se puede:

1. **Representar** cada imagen como un quadtree: los nodos se subdividen sólo donde hay detalle (varianza > umbral); las zonas homogéneas quedan como hojas grandes.
2. **Difundir sobre esa estructura**: el modelo de ruido/desruido opera sobre los nodos del árbol, no sobre la rejilla densa.
3. **Generar también la estructura** (de ahí *Dynamic*): el modelo no sólo predice los valores de las hojas, sino **si cada nodo debe subdividirse** (`quadtree_decoder.py`: *"predicting leaf values and subdivision probabilities"*). La topología del árbol emerge durante el muestreo, de grueso a fino.
4. **Medir el beneficio**: comparar contra un **U-Net denso** (`baseline_unet.py`) en calidad (FID, NLL/bpd) frente a coste (nº de nodos/tokens, FLOPs, memoria).
5. **Escalar a 3D**: el mismo esquema con octrees para formas/volúmenes, donde la dispersión (>95 % del espacio vacío) hace el ahorro mucho mayor.

### Decisiones de diseño tomadas
| Decisión | Elección | Motivo |
|---|---|---|
| Valores de las hojas | **Binarios** (lleno / vacío) | En 3D interesa la **ocupación** (vóxel lleno o vacío); la malla y los vértices se extraen después (p. ej. Marching Cubes). El 2D binario es el banco de pruebas directo. |
| Árbol fijo o generado | **Generado** | Con hojas binarias, la estructura del árbol *es* la imagen: un árbol fijo apenas dejaría nada que generar. |
| Tipo de difusión | **Discreta** (máscaras / D3PM), no gaussiana | Los datos son categóricos. |
| Primera arquitectura | **Transformer sobre nodos** | Acepta secuencias de longitud variable, es sencillo y es el estándar actual (DiT, OctGPT, BudgetPix). Más adelante se comparará con una CNN dispersa (O-CNN). |

**Formulación concreta.** Cada nodo toma un estado de **3 clases: `{vacío, lleno, subdividir}`**. Una imagen es el conjunto de estados de los nodos que existen, nivel a nivel. El modelo aprende a desenmascarar/desruidar esos estados de grueso a fino: los nodos `subdividir` del nivel *l* crean 4 hijos en el nivel *l+1*.

### Hipótesis de trabajo
- H1: un quadtree con umbral adecuado representa MNIST con una fracción pequeña de nodos (≈10–30 %) y error de reconstrucción despreciable.
- H2: un modelo de difusión discreta sobre nodos alcanza calidad comparable al baseline denso con menos cómputo.
- H3: predecir la subdivisión de forma generativa (coarse-to-fine) es viable y estable.

---

## 2. Estado actual del código

| Componente | Archivo | Estado |
|---|---|---|
| MNIST binario 32×32 (lee los IDX, sin torchvision) | `data/binary_mnist.py` | ✅ |
| Quadtree binario exacto, por lotes, serialización a tokens | `structures/binary_quadtree.py` | ✅ con tests |
| Estadísticas de la Fase 0 | `experiments/quadtree_stats.py` | ✅ |
| `Dataset` de árboles | `data/quadtree_dataset.py` | ✅ |
| Transformer sobre nodos | `models/quadtree_transformer.py` | ✅ |
| Difusión enmascarada nivel a nivel (entrenamiento y muestreo) | `models/adaptive_diffusion.py` | ✅ con tests |
| Entrenar / muestrear | `experiments/run_training.py`, `experiments/run_sampling.py` | ✅ (falta un entrenamiento largo) |
| Baseline sin árbol (difusión enmascarada sobre píxeles) | — | ⬜ Hito 2 |
| Métricas (FID, NLL, FLOPs) | `utils/metrics.py` | ⬜ Hito 3 |
| Prototipo inicial en escala de grises | `structures/node.py`, `structures/quadtree.py`, `main.py` | Heredado; lo sustituye el quadtree binario. Sus hijos se solapan con lados impares (28→7→3/4). |

Archivos aún vacíos del esqueleto original: `models/baseline_unet.py`, `models/quadtree_encoder.py`, `models/quadtree_decoder.py`, `diffusion/*`, `configs/*`, `data/quadtree_builder.py`.

### Uso
```bash
cd adaptive_diffusion
pip install torch numpy matplotlib pytest
python -m pytest -q tests
python -m experiments.quadtree_stats
python -m experiments.run_training --steps 20000 --out runs/qdd
python -m experiments.run_sampling --ckpt runs/qdd/model.pt --num 100
```

### Entrenar en GPU
Los scripts usan la GPU automáticamente si PyTorch la detecta (`--device auto`; también `cuda`, `cuda:0`, `mps`, `cpu`). En CUDA se activa la precisión mixta (`--no_amp` la desactiva). Se guarda un punto de control cada `--save_every` pasos.

```bash
python -m experiments.run_training --device cuda --batch 256 --lr 1e-3 --steps 30000 --out runs/qdd_gpu
```

En Google Colab (entorno de ejecución con GPU):
```bash
!git clone -b claude/keen-heisenberg-yew59l https://github.com/carlshue/QDD2D.git
%cd QDD2D/adaptive_diffusion
!python -m experiments.run_training --device cuda --batch 256 --lr 1e-3 --steps 30000 --out runs/qdd_gpu
!python -m experiments.run_sampling --ckpt runs/qdd_gpu/model.pt --num 100
```

**Experimento completo en un solo comando** (entrena, muestrea y deja todo en `results/<nombre>/` para subirlo a git):
```bash
python -m experiments.run_local --name gpu01
```
Guarda `train.log`, `sample.log`, `samples.png`, `model.pt` y `env.json` (GPU, versiones, duración).

> La ruta de GPU no se ha podido ejecutar todavía: el entorno donde se escribió el código no tiene GPU. En CPU el entrenamiento va a ~0,8 s por paso (lote de 64).

### Resultado de la Fase 0 (medido sobre los 60 000 dígitos de entrenamiento)
| | Árbol | Rejilla 32×32 |
|---|---|---|
| Tokens por imagen (media) | **216** (p5 137, p95 289, máx 397) | 1024 |
| Proporción | 21 % (27,5 % frente a los 784 píxeles de 28×28) | 100 % |
| Coste relativo de la atención (∝ L²) | **4,6 %** | 100 % |
| Reconstrucción | exacta en las 70 000 imágenes | — |

Nodos por nivel (media): 1 · 4 · 15,8 · 28,7 · 64 · 102,5. **La hipótesis H1 se cumple.**

### Resultado del prototipo (`results/gpu01`, RTX 3060, 75 min)
- 30 000 pasos, lote 256, 0,81 M de parámetros. La pérdida baja de 0,93 a **0,289** y sigue bajando despacio al final.
- Las 100 muestras condicionadas son dígitos reconocibles de la clase pedida en su gran mayoría (inspección visual, sin métrica todavía). Fallos visibles: algunos trazos rotos, píxeles sueltos y algún dígito deformado.
- Árboles generados: **208,5 nodos de media** (máx. 289), frente a 216 en los datos y 1024 de la rejilla. 35 evaluaciones de red por lote de muestras.
- Canonicidad (200 muestras nuevas): el 62 % de los árboles son canónicos; el resto lleva de media 2,5 nodos de más (cuatro hermanos iguales que deberían ser una hoja). Efecto pequeño.

**El hito H1 se cumple.** Falta el baseline (H2) y las métricas (H3) para saber si el árbol compensa.

---

## 3. Viabilidad, hitos y pasos

### Viabilidad
| Aspecto | Valoración | Motivo |
|---|---|---|
| Representación (H1) | **Demostrada** | 21 % de los tokens, sin pérdida (tabla anterior). |
| Técnica en 2D | **Alta** | Piezas conocidas: difusión enmascarada (MDLM/MaskGIT) + transformer. El prototipo ya entrena y muestrea árboles válidos. |
| Cómputo en MNIST | **Alta** | Modelo de ~1 M de parámetros; horas en una GPU modesta. En CPU sólo sirve para pruebas de humo. |
| Ganar al baseline en calidad (H2) | **Incierta: es la pregunta de investigación** | El árbol ahorra tokens por paso, pero generar nivel a nivel puede necesitar más pasos y acumular errores (un `subdividir` mal decidido no se corrige después). |
| Novedad | **Media-alta, con riesgo de solape** | BudgetPix, QuadTok y DVD (2026) están muy cerca; ninguno hace difusión discreta sobre la topología del árbol. El campo se mueve rápido. |
| Paso a 3D | **Media** | El código se generaliza (8 hijos en vez de 4), pero hacen falta GPU seria, datos (ShapeNet) y probablemente atención dispersa. |

**Riesgos principales y mitigación**
1. *Error acumulado entre niveles* → re-enmascarado (ReMDM), o variante B (todos los niveles a la vez).
2. *Más evaluaciones de red* (niveles × pasos) → pocos pasos en los niveles gruesos (tienen ≤16 nodos), medir coste total en FLOPs y no en pasos.
3. *MNIST es demasiado fácil para ver diferencias* → repetir en Fashion-MNIST binario y en siluetas a 64×64 o 128×128, donde el ahorro crece.
4. *Árboles no canónicos* (4 hermanos iguales que deberían ser una hoja) → medir su frecuencia; si molesta, penalizar o fusionar al decodificar.

### Hitos
| Hito | Contenido | Criterio de éxito | Estado |
|---|---|---|---|
| **H0. Representación** | Quadtree binario exacto, tokens, estadísticas, tests | Reconstrucción exacta; < 30 % de tokens | ✅ hecho |
| **H1. Prototipo** | Transformer + difusión enmascarada nivel a nivel; entrenar y muestrear | Los tests pasan; dígitos reconocibles tras un entrenamiento completo | ✅ hecho (`results/gpu01`) |
| **H2. Baseline justo** | Misma red y misma difusión sobre los 1024 píxeles, sin árbol | Baseline entrenado con el mismo presupuesto | ⬜ |
| **H3. Evaluación** | FID/KID con extractor entrenado en MNIST, NLL, tokens, FLOPs y tiempo por muestra | Tabla calidad frente a coste, árbol vs rejilla | ⬜ |
| **H4. Ablaciones** | Pasos por nivel, re-enmascarado, variante B, orden de nodos, árbol con profundidad limitada | Saber qué decisiones importan | ⬜ |
| **H5. Escalado 2D** | Fashion-MNIST binario; siluetas a 64–128 px | El ahorro relativo crece con la resolución | ⬜ |
| **H6. 3D** | Octree de ocupación en ShapeNet; comparar con OctFusion, OctGPT, DVD | Formas plausibles con menos tokens que la rejilla | ⬜ |

**Decisión tras H3:** si el árbol iguala al baseline en calidad con menos cómputo, seguir a H4–H6. Si pierde con claridad, probar la variante B y el re-enmascarado antes de descartar.

### Cómo funciona el prototipo (H1)
- **Tokens:** cada nodo existente es `(nivel, y, x, estado)`, en orden de nivel y, dentro de él, en orden Morton (los 4 hermanos van seguidos).
- **Entrenamiento:** se elige un nivel *l* al azar. Los niveles anteriores se dan limpios; los nodos del nivel *l* se enmascaran con probabilidad *t* ~ U(0,1). Pérdida: entropía cruzada sobre los enmascarados. Condicionado a la etiqueta del dígito (con *dropout* de etiqueta).
- **Muestreo:** se parte de la raíz enmascarada. En cada nivel se revelan los nodos en varios pasos (calendario coseno); los `subdividir` crean 4 hijos enmascarados. Nunca se procesan nodos que no existen.
- **Pendiente conocido:** la pérdida no lleva el peso 1/*t* de MDLM (es la versión tipo MaskGIT), así que todavía no da una cota de verosimilitud.

---

## 3.1 Resultados en MNIST 32×32 (noviembre de 2026)

2000 muestras por medida y 3 semillas de muestreo (rangos mínimo–máximo). Imágenes reales: 98,7–99,0 % de
acierto de clase y Fréchet 3,0–4,5. Los datos completos están en `adaptive_diffusion/results/analysis/`.

| Caso | Temperatura | Acierto | Fréchet | Tokens por muestra |
|---|---|---|---|---|
| `grow` (sin unión), 8 pasos por nivel | 1,0 | 93,9–94,7 % | 6,9–9,2 | 3 030 |
| `grow` (sin unión), 10 pasos por nivel | 0,9 | 96,5–97,1 % | 4,7–5,7 | 3 500 |
| `grow` + reenmascarado por confianza 0,05 | 1,0 | 96,9–97,2 % | 5,7–6,0 | 3 140 |
| `grow_merge` semilla 0, con unión, 8 pasos | 0,9 | 97,9–98,8 % | 8,6–10,5 | 3 650 |
| `grow_merge` semilla 0, sin unión, 8 pasos | 0,9 | 97,5–98,0 % | 7,8–9,5 | 3 190 |
| `grow_merge` semilla 1, con unión, 8 pasos | 0,9 | 97,8–98,2 % | 7,3–7,9 | 3 640 |
| `grow_merge` semilla 1, sin unión, 8 pasos | 0,9 | 96,7–97,6 % | 7,8–8,6 | 3 160 |
| `edit cell`, 32 pasos | 0,7 | 96,8–97,2 % | 8,0–9,3 | 7 190 |
| `edit cell` + resolve, 32 pasos | 0,8 | 97,7–98,9 % | 5,0–5,5 | 23 260 |

Lectura:

- **La temperatura de muestreo pesa más que cualquier técnica probada.** `grow` pasa de Fréchet 7–9 a 5–6
  solo con bajarla a 0,85–0,9. Toda comparación debe barrerla en cada método (`experiments.temperature_sweep`).
- **`grow` con temperatura ajustada es la referencia:** empata en Fréchet con `edit cell` + resolve con 7
  veces menos tokens.
- **La unión entrenada (`grow_merge`) no mejora a `grow` en MNIST.** Sube el acierto (hasta 98,8 %) pero el
  Fréchet es peor (7–10 frente a 5–6): muestras más reconocibles y menos variadas. La revisión actúa poco
  (unas 2,4 revisiones y 1,6 uniones por imagen) y no poda: el árbol final tiene 215–220 nodos frente a 211.
- **La varianza entre semillas de entrenamiento es grande:** las dos semillas de `grow_merge` difieren en
  unos 2 puntos de Fréchet, más que el efecto de activar la unión. Con una semilla por caso no se
  distinguen diferencias pequeñas.
- **El reenmascarado estilo ReMDM sobre `grow`** (sin reentrenar, solo el nivel en curso) mejora por
  confianza y no hace nada al azar; no supera al control de temperatura.
- **El prior del árbol aleatorio decide el resultado de `edit`:** `cell` ≫ `level` ≫ `uniform`.

## 3.2 Unión entrenada con los errores propios (noviembre de 2026)

`grow_merge` se entrenaba estropeando los nodos ya decididos con ruido del prior `cell`. Ahora puede
estropearlos con los errores de la propia red (`--corrupt self`) o con ambas cosas (`--corrupt both`):
en cada lote se tapa parte de un nivel anterior, se sortea la predicción de la red y se deja ese
resultado como si estuviera decidido. Dos semillas de entrenamiento por variante, 30 000 pasos.

**Medida directa** (`experiments.correction_analysis`, 5000 imágenes de prueba, sin generar nada): de los
errores que comete la red al decidir un nivel, cuántos señala después (probabilidad del estado actual
menor que 0,5) y cuántos arreglaría.

| Entrenado con | Detecta | Arreglaría | Subdivisión de más | Subdivisión de menos | Falsa alarma |
|---|---|---|---|---|---|
| ruido del prior (`prior`) | 35–37 % | 34–36 % | 51–55 % | 18 % | 0,1 % |
| errores propios (`self`) | 59–63 % | 57–60 % | 74–76 % | 43–48 % | 0,3 % |
| ambos (`both`) | 60–61 % | 58–59 % | 74–75 % | 44–46 % | 0,3 % |

**Calidad de las muestras** (2000 muestras, 3 semillas de muestreo; cada celda da el rango de la semilla
de entrenamiento 0 y el de la 1, con la mejor temperatura de cada una):

| Entrenado con | Fréchet con unión | Fréchet sin unión | Revisiones por imagen | Nodos finales |
|---|---|---|---|---|
| `prior` | 8,6–10,5 · 6,4–8,0 | 7,0–8,2 · 7,2–8,4 | 1,9–2,8 | 214–218 |
| `self` | 7,9–9,4 · 10,5–11,1 | 10,5–12,4 · 11,5–13,5 | 2,0–4,6 | 207–210 |
| `both` | 6,6–7,4 · 5,6–6,8 | 7,4–8,7 · 6,1–6,8 | 3,1–4,9 | 212–216 |
| `grow` sin unión (referencia, 1 semilla) | — | 4,7–5,7 | — | 211 |

Lectura:

- **La red aprende a revisar:** detecta y arreglaría casi el doble de sus errores (del 35 % al 60 %), en
  las dos semillas y también sobre un conjunto de errores común.
- **`both` es la mejor variante con unión** y mejora a `prior` en las dos semillas. `self` sola empeora
  las muestras aunque revisa bien; con temperatura 1 y sin revisión cae a Fréchet 23–29.
- **Con temperatura 1 la revisión ahora importa mucho** (`both`: de 11–14 sin unión a 6–9 con ella). Con
  la temperatura ajustada la diferencia se reduce a un punto o menos.
- **Sigue sin superar a `grow` con temperatura ajustada** (4,7–5,7), aunque `both` semilla 1 se queda cerca
  (5,6–6,8). `grow` solo tiene una semilla de entrenamiento, así que la comparación no es concluyente.
- **Poda:** `self` deja el árbol en 207–210 nodos (el mínimo para estas imágenes es ~209); las demás, 212–218.

## 4. Estado del arte

### 4.1 Fundamentos de difusión
- Ho, Jain, Abbeel. *Denoising Diffusion Probabilistic Models*. NeurIPS 2020. arXiv:2006.11239
- Song et al. *Score-Based Generative Modeling through SDEs*. ICLR 2021. arXiv:2011.13456
- Nichol, Dhariwal. *Improved DDPM*. ICML 2021. arXiv:2102.09672
- Song, Meng, Ermon. *DDIM*. ICLR 2021. arXiv:2010.02502
- Karras et al. *Elucidating the Design Space of Diffusion Models (EDM)*. NeurIPS 2022. arXiv:2206.00364
- Lipman et al. *Flow Matching for Generative Modeling*. ICLR 2023. arXiv:2210.02747 · Liu et al. *Rectified Flow*. arXiv:2209.03003
- Austin et al. *Structured Denoising Diffusion in Discrete State-Spaces (D3PM)*. NeurIPS 2021. arXiv:2107.03006 — relevante para difundir los bits de subdivisión.

### 4.2 Difusión multiescala / coarse-to-fine (lo más cercano en 2D)
- Ho et al. *Cascaded Diffusion Models*. JMLR 2022. arXiv:2106.15282
- Rombach et al. *Latent Diffusion Models*. CVPR 2022. arXiv:2112.10752
- Gu et al. *Matryoshka Diffusion Models*. ICLR 2024. arXiv:2310.15111
- Teng et al. *Relay Diffusion*. ICLR 2024. arXiv:2309.03350
- Gu et al. *f-DM: multi-stage diffusion via progressive signal transformation*. ICLR 2023. arXiv:2210.04955
- Hoogeboom et al. *simple diffusion*. ICML 2023. arXiv:2301.11093
- Crowson et al. *Hourglass Diffusion Transformers (HDiT)*. ICML 2024. arXiv:2401.11605
- Phung et al. *Wavelet Diffusion Models*. CVPR 2023. arXiv:2211.16152 — los wavelets Haar son el análogo "denso" de un quadtree.
- Tian et al. *Visual Autoregressive Modeling (VAR): next-scale prediction*. NeurIPS 2024. arXiv:2404.02905 — generación por escalas, inspiración directa para la Fase 3.

### 4.3 Tokens adaptativos / cómputo variable
- Peebles, Xie. *Scalable Diffusion Models with Transformers (DiT)*. ICCV 2023. arXiv:2212.09748
- Ronen et al. *Vision Transformers with Mixed-Resolution Tokenization (Quadformer)*. CVPRW 2023. arXiv:2304.00287 — **tokenización por quadtree**, muy cercano.
- Tang et al. *QuadTree Attention for Vision Transformers*. ICLR 2022. arXiv:2201.02767
- Duggal et al. *Adaptive Length Image Tokenization via Recurrent Allocation (ALIT)*. ICLR 2025. arXiv:2411.02393
- Bachmann et al. *FlexTok: resampling images into 1D token sequences of flexible length*. ICML 2025. arXiv:2502.13967
- Nash et al. *Generating Images with Sparse Representations (DCTransformer)*. ICML 2021. arXiv:2103.03841
- Wang et al. *Patch Diffusion*. NeurIPS 2023. arXiv:2304.12526
- Bolya, Hoffman. *Token Merging for Fast Stable Diffusion (ToMe-SD)*. CVPRW 2023. arXiv:2303.17604

### 4.4 Generación jerárquica 3D (objetivo final)
- Wang et al. *O-CNN: Octree-based CNNs*. SIGGRAPH 2017 · *Adaptive O-CNN*. SIGGRAPH Asia 2018.
- Tatarchenko et al. *Octree Generating Networks*. ICCV 2017. arXiv:1703.09438
- Ibing et al. *Octree Transformer: autoregressive 3D shape generation*. CVPR 2023. arXiv:2111.12480
- Xiong et al. *OctFusion: Octree-based Diffusion Models for 3D Shape Generation*. 2024. arXiv:2408.14732
- Ren et al. *XCube: Large-Scale 3D Generative Modeling using Sparse Voxel Hierarchies*. CVPR 2024. arXiv:2312.03806
- Wei et al. *OctGPT: Octree-based Multiscale Autoregressive Models for 3D Shape Generation*. SIGGRAPH 2025. arXiv:2504.09975
- Hui et al. *Neural Wavelet-domain Diffusion for 3D Shape Generation*. SIGGRAPH Asia 2022. arXiv:2209.08725
- Zheng et al. *LAS-Diffusion*. SIGGRAPH 2023. arXiv:2305.04461

### 4.5 Difusión discreta / enmascarada (clave con datos binarios)
- Chang et al. *MaskGIT: Masked Generative Image Transformer*. CVPR 2022. arXiv:2202.04200
- Sahoo et al. *Simple and Effective Masked Diffusion Language Models (MDLM)*. NeurIPS 2024. arXiv:2406.07524 · código: `kuleshov-group/mdlm`
- Shi et al. *Simplified and Generalized Masked Diffusion for Discrete Data (MD4)*. NeurIPS 2024. arXiv:2406.04329
- Lou et al. *Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution (SEDD)*. ICML 2024. arXiv:2310.16834
- Wang et al. *Remasking Discrete Diffusion Models with Inference-Time Scaling (ReMDM)*. NeurIPS 2025. arXiv:2503.00307 — permite corregir tokens ya decididos al muestrear.
- Sahoo et al. *Scaling Beyond Masked Diffusion Language Models*. 2026. arXiv:2602.15014 — compara difusión enmascarada, uniforme e interpolante a gran escala.

### 4.6 Novedades 2025–2026 directamente relacionadas
- Kara et al. **BudgetPix: Compute-Adaptive Tokenization for Pixel-Space Image Diffusion**. 2026. arXiv:2610.12307 — quadtree guiado por entropía y tokens de longitud variable para difusión en píxeles; iguala a PixelDiT a 1024² con ~25 % del cómputo. **Es la evidencia más directa de que tu intuición funciona** (en continuo).
- Mao et al. **QuadTok: Quadtree Visual Tokenizer for Autoregressive Image Generation**. 2026. arXiv:2610.10497 — tokenizador por quadtree para un generador **autorregresivo** (no difusión). Ahorra ~10 % de tokens; su mejor gFID (2,08, ImageNet 256) usa una topología de quadtree **dada de antemano**. Código: `myc634/QuadTok`.
- Zamfir et al. **Elastic Token Compression for Pixel-Space Diffusion Transformers**. 2026. arXiv:2608.29281 — regiones adaptativas no diádicas (discute las limitaciones del quadtree).
- Shen et al. **CAT: Content-Adaptive Image Tokenization**. 2025. arXiv:2501.03120
- **HierOctFusion**. 2025. arXiv:2508.11106 — octree diffusion con jerarquía de partes.
- **Octree Diffusion for Semantic Scene Generation and Completion**. 2025. arXiv:2509.16483
- Xiang et al. **DVD: Discrete Voxel Diffusion for 3D Generation and Editing**. ICML 2026. arXiv:2605.07971 — **difusión discreta sobre ocupación binaria de vóxeles**, el equivalente 3D de este proyecto, pero sin jerarquía. Código: `TeCai/DVD`.
- Encontrados y aún sin revisar: *MOSAIK: Multi-Patch Content-Aware Spatial Allocation of Image Tokens* (arXiv:2608.05450) y *Dynamic Chunking Diffusion Transformer, DC-DiT* (arXiv:2603.06351).
- OctFusion fue aceptado en SGP 2025 (v2 de arXiv:2408.14732).

> **Estado de la verificación (nov. 2026).** Título, autores e identificador de las referencias de §4.5–4.6 de 2025–2026 se han contrastado por búsqueda web. Las de §4.1–4.4 (2017–2025) no se han podido contrastar una a una porque arxiv.org no es accesible desde el entorno de trabajo; revisar los identificadores antes de citarlas. Las cifras de los preprints de 2026 son las que declaran sus autores.

### 4.7 Posicionamiento
- **BudgetPix** (difusión) y **QuadTok** (autorregresivo) muestran que los tokens por quadtree ahorran cómputo, pero con **valores continuos** y un **árbol calculado o dado de antemano**, no generado.
- **DVD** hace difusión **discreta de ocupación**, pero sobre una rejilla **plana**.
- **OctGPT / XCube / OctFusion** son jerárquicos en 3D, pero autorregresivos o con difusión **continua** en latente.

**Hueco de QDD2D:** **difusión discreta (enmascarada) sobre los estados `{vacío, lleno, subdividir}` de un quadtree, generando la topología de grueso a fino**. Une la jerarquía de OctGPT/XCube con la difusión discreta de ocupación de DVD, validada primero en 2D (MNIST) y extensible directamente a octrees.

---

## 5. Recursos útiles
- Datasets: MNIST → Fashion-MNIST → CIFAR-10 / CelebA 64; 3D: ShapeNet, Objaverse.
- Métricas: `torchmetrics` (FID/KID), `clean-fid`.
- Código de referencia: `lucidrains/denoising-diffusion-pytorch`, `facebookresearch/DiT`, `octree-nn/ocnn-pytorch`, `octree-nn/octfusion`, `nv-tlabs/XCube`.
- Orden Morton (Z-order) para serializar nodos y hacer batching eficiente.

---

## 6. Preguntas
**Resueltas:** valores binarios · árbol generado · difusión discreta · transformer como primera arquitectura (ver §1).

**Abiertas:**
1. ¿Objetivo final: publicación, TFG/TFM o prototipo para 3D? Determina el alcance de H3–H6.
2. ¿Hay GPU disponible para los entrenamientos de H1–H3?
3. Variante A frente a B y árbol exacto frente a aleatorizado: se resuelven experimentalmente en H4.

### Glosario rápido de arquitecturas
- **Transformer sobre nodos**: cada nodo es un token con su posición y nivel; la atención relaciona todos con todos. Flexible; coste cuadrático.
- **GNN**: los nodos intercambian mensajes por aristas (padre-hijo y vecinos). Respeta la estructura, pero la información viaja despacio.
- **CNN dispersa (O-CNN)**: convoluciones sólo sobre las celdas existentes, nivel a nivel. Muy eficiente en 3D; necesita librerías específicas.
