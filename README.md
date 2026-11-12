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

### Resultado de la Fase 0 (medido sobre los 60 000 dígitos de entrenamiento)
| | Árbol | Rejilla 32×32 |
|---|---|---|
| Tokens por imagen (media) | **216** (p5 137, p95 289, máx 397) | 1024 |
| Proporción | 21 % (27,5 % frente a los 784 píxeles de 28×28) | 100 % |
| Coste relativo de la atención (∝ L²) | **4,6 %** | 100 % |
| Reconstrucción | exacta en las 70 000 imágenes | — |

Nodos por nivel (media): 1 · 4 · 15,8 · 28,7 · 64 · 102,5. **La hipótesis H1 se cumple.**

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
| **H1. Prototipo** | Transformer + difusión enmascarada nivel a nivel; entrenar y muestrear | Los tests pasan; dígitos reconocibles tras un entrenamiento completo | 🟡 código hecho; falta entrenamiento largo en GPU |
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
