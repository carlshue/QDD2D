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
| Carga MNIST normalizada a [-1,1] | `data/mnist_loader.py` | ✅ |
| Nodo y subdivisión recursiva por varianza | `structures/node.py` | ✅ (ver problemas) |
| Quadtree, reconstrucción continua y binaria | `structures/quadtree.py` | ✅ |
| Visualización (original / binaria / overlay) | `utils/visualization.py` | ✅ |
| Script de inspección / logging | `main.py`, `experiments/log_quadtree.py` | ✅ |
| Constructor y `Dataset` de quadtrees | `data/quadtree_builder.py`, `data/quadtree_dataset.py` | ⬜ sólo cabecera |
| Encoder / decoder de nodos | `models/quadtree_encoder.py`, `models/quadtree_decoder.py` | ⬜ |
| Proceso de difusión jerárquico | `models/adaptive_diffusion.py` | ⬜ |
| Baseline U-Net | `models/baseline_unet.py` | ⬜ vacío |
| Ruido y scheduler | `diffusion/noise.py`, `diffusion/scheduler.py` | ⬜ vacío |
| Configs, métricas, train/sample | `configs/*.yaml`, `utils/metrics.py`, `experiments/run_*.py` | ⬜ |

### Problemas conocidos
- **Tamaño impar (28→14→7→3/4)**: en `QuadtreeNode.subdivide` los hijos son cuadrados de lado `max(w,h)`, por lo que se **solapan** cuando el tamaño es impar. Opciones: rellenar MNIST a 32×32 (recomendado, potencia de 2) o usar nodos rectangulares `(w,h)`.
- `threshold=0.0` sobre datos en [-1,1] subdivide casi hasta el píxel en los bordes del dígito; hay que barrer el umbral y fijar `max_depth`.
- Hay `__pycache__/` y los datos MNIST en `data/MNIST/raw` versionados; deberían ir en `.gitignore`.
- Los imports son relativos a `adaptive_diffusion/` (ejecutar desde esa carpeta).

### Uso
```bash
cd adaptive_diffusion
pip install torch torchvision numpy matplotlib
python main.py
```

---

## 3. Diseño propuesto (hoja de ruta)

**Fase 0 — Representación binaria.** Pad a 32×32 (5 niveles: 32→16→8→4→2→1). Binarizar MNIST (umbral 0 sobre [-1,1]). Construir el árbol **exacto**: un nodo es hoja `vacío`/`lleno` si su bloque es uniforme; si no, `subdividir`. Estadísticas: nº de nodos frente a 1024 píxeles e histograma de profundidades. Serializar a tensores `(nivel, x, y, estado)` en orden BFS + Morton (Z-order) dentro de cada nivel, con padding y máscara para batching.

**Fase 1 — Baselines.**
- (a) Difusión discreta enmascarada (MDLM / MaskGIT) sobre los **1024 píxeles binarios** con un transformer. Es la comparación justa: mismo tipo de difusión, sin árbol.
- (b) Opcional: DDPM con U-Net denso como referencia clásica.

**Fase 2 — Difusión discreta sobre el árbol (núcleo del proyecto).** Dos variantes:
- *Variante A, nivel a nivel (cascada, tipo VAR/OctGPT, pero con difusión)*: para el nivel *l*, se parte de todos los nodos enmascarados y se desenmascaran en *T* pasos, condicionando en los niveles anteriores (ya fijados). Los nodos `subdividir` generan los hijos del nivel siguiente. Un único transformer compartido entre niveles, con embedding de nivel.
- *Variante B, árbol completo enmascarado*: se considera el árbol completo de 32×32 (1365 nodos posibles) y se enmascaran estados con una máscara consistente (un hijo sólo existe si el padre es `subdividir`). Difusión conjunta de todos los niveles.
- Red: transformer con embeddings de `(x, y, nivel, estado)`, atención entre todos los nodos activos (o restringida a padre/vecinos si escala mal).
- Pérdida: entropía cruzada sobre los nodos enmascarados (objetivo MDLM).

**Fase 3 — Variantes de árbol.** Comparar árbol exacto, árbol con profundidad máxima (hojas con mayoría lleno/vacío) y un orden de subdivisión aleatorio como *augmentation* (es la duda de "árbol fijo o aleatorio").

**Fase 4 — Evaluación.** FID/KID con un extractor entrenado en MNIST, precisión/recall, NLL (bits por imagen), nº medio de tokens, FLOPs y tiempo por muestra frente a los baselines. Ablaciones: nº de pasos *T*, orden de serialización, variante A frente a B, transformer frente a CNN dispersa.

**Fase 5 — 3D.** Octrees sobre ShapeNet (SDF/ocupación), comparando con OctFusion, XCube, OctGPT.

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
- *Scaling study de difusión discreta* (enmascarada frente a uniforme e interpolante). 2026. arXiv:2602.15014

### 4.6 Novedades 2025–2026 directamente relacionadas
- **BudgetPix: Compute-Adaptive Tokenization for Pixel-Space Image Diffusion**. 2026. arXiv:2610.12307 — quadtree guiado por entropía y tokens de longitud variable para difusión en píxeles; iguala a PixelDiT a 1024² con ~25 % del cómputo. **Es la evidencia más directa de que tu intuición funciona** (en continuo).
- **QuadTok: Quadtree Visual Tokenizer**. 2026. arXiv:2610.10497 — tokenizador por quadtree; generación controlable a partir de *layouts* de quadtree.
- **Elastic Token Compression for Pixel-Space DiTs**. 2026. arXiv:2608.29281 — regiones adaptativas no diádicas (discute las limitaciones del quadtree).
- **CAT: Content-Adaptive Image Tokenization**. 2025. arXiv:2501.03120
- **HierOctFusion**. 2025. arXiv:2508.11106 — octree diffusion con jerarquía de partes.
- **Octree Diffusion for Semantic Scene Generation and Completion**. 2025. arXiv:2509.16483
- **DVD: Discrete Voxel Diffusion for 3D Generation and Editing**. ICML 2026. arXiv:2605.07971 — **difusión discreta sobre ocupación binaria de vóxeles**, el equivalente 3D de este proyecto, pero sin jerarquía.
- OctFusion fue aceptado en SGP 2025 (v2 de arXiv:2408.14732).

> Las referencias de 2026 son preprints; las cifras salen del resumen y conviene verificarlas leyendo los artículos.

### 4.7 Posicionamiento
- **BudgetPix / QuadTok** ya demuestran que los tokens por quadtree ahorran cómputo en difusión, pero con **valores continuos** y un **árbol calculado de antemano** a partir de la imagen.
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
1. ¿Variante A (nivel a nivel) o B (árbol completo enmascarado)? Propuesta: empezar por A, que es más simple y escala mejor a 3D.
2. ¿Árbol exacto o aleatorizado? Se resolverá experimentalmente en la Fase 3.
3. ¿Objetivo final: publicación, TFG/TFM o prototipo para 3D? Determina el alcance de la evaluación.

### Glosario rápido de arquitecturas
- **Transformer sobre nodos**: cada nodo es un token con su posición y nivel; la atención relaciona todos con todos. Flexible; coste cuadrático.
- **GNN**: los nodos intercambian mensajes por aristas (padre-hijo y vecinos). Respeta la estructura, pero la información viaja despacio.
- **CNN dispersa (O-CNN)**: convoluciones sólo sobre las celdas existentes, nivel a nivel. Muy eficiente en 3D; necesita librerías específicas.
