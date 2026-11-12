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

### Hipótesis de trabajo
- H1: un quadtree con umbral adecuado representa MNIST con una fracción pequeña de nodos (≈10–30 %) y error de reconstrucción despreciable.
- H2: un modelo de difusión sobre nodos alcanza calidad comparable al baseline denso con menos cómputo.
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

**Fase 0 — Representación.** Pad a 32×32, barrido de umbral/profundidad, estadísticas (nº hojas, MSE, histograma de profundidades). Serializar el árbol a tensores: lista de nodos con `(nivel, x, y, valor, máscara_split)` en orden BFS/Morton (Z-order), con padding y máscara para batching.

**Fase 1 — Baseline denso.** DDPM con U-Net pequeño en MNIST 32×32 (referencia de FID y coste).

**Fase 2 — Difusión sobre estructura fija.** Árbol dado (del dato real), difusión gaussiana sólo sobre los valores de las hojas. Red: transformer sobre tokens de nodo con embedding de posición (x, y, nivel) — tipo DiT con tokens de tamaño variable. Valida H2 de forma aislada.

**Fase 3 — Estructura generativa (dinámica).** Generar el árbol nivel a nivel:
- *Opción A (coarse-to-fine en cascada)*: en cada nivel, un modelo predice valores (difusión continua) y bits de subdivisión (clasificación / difusión discreta tipo D3PM) condicionado al nivel anterior.
- *Opción B (estructura en el estado)*: difusión conjunta sobre valores + bits de *split*, con un árbol "completo" enmascarado.
- *Opción C (latente)*: autoencoder árbol→latente fijo y difusión en latente (estilo LDM / XCube).

**Fase 4 — Evaluación.** FID/KID (con clasificador MNIST), precisión/recall, bpd, nº medio de tokens, FLOPs y tiempo por muestra vs baseline. Ablaciones: umbral, profundidad máx., orden de serialización.

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

### 4.5 Posicionamiento
Ningún trabajo revisado combina las tres cosas que QDD2D busca: **(i)** difusión **(ii)** sobre tokens de quadtree de tamaño variable **(iii)** con la **topología generada dinámicamente** en 2D. OctFusion/XCube/OctGPT lo hacen en 3D (con octrees y ocupación), Quadformer usa quadtrees sólo para clasificación, y Matryoshka/VAR/Relay son multiescala pero densos. El nicho es un estudio limpio y controlado en 2D que sirva de banco de pruebas para el 3D.

---

## 5. Recursos útiles
- Datasets: MNIST → Fashion-MNIST → CIFAR-10 / CelebA 64; 3D: ShapeNet, Objaverse.
- Métricas: `torchmetrics` (FID/KID), `clean-fid`.
- Código de referencia: `lucidrains/denoising-diffusion-pytorch`, `facebookresearch/DiT`, `octree-nn/ocnn-pytorch`, `octree-nn/octfusion`, `nv-tlabs/XCube`.
- Orden Morton (Z-order) para serializar nodos y hacer batching eficiente.

---

## 6. Preguntas abiertas (a decidir)
1. ¿Los valores de las hojas son **continuos** (escala de grises) o **binarios** (como `to_binary_image`)? Cambia el tipo de difusión (gaussiana vs discreta).
2. ¿La estructura del árbol se **genera** (Fase 3) o basta con difundir sobre árboles fijos?
3. ¿Arquitectura preferida: transformer sobre nodos, GNN sobre el árbol, o CNN dispersa tipo O-CNN?
4. ¿Objetivo final: publicación, TFG/TFM, o prototipo para 3D? Determina el alcance de la evaluación.
