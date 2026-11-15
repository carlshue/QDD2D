# QDD2D

Difusión discreta sobre quadtrees binarios en 2D (MNIST) como banco de pruebas barato de ideas que
luego se quieren llevar a octrees en 3D. La referencia a superar o complementar es OctGPT.

## Qué se quiere conseguir

- **Meta de fondo:** aportar algo sobre OctGPT (generación autorregresiva multiescala sobre octrees
  serializados). OctGPT decide cada subdivisión una vez y no puede deshacerla.
- **Hipótesis principal:** permitir **unir** nodos (deshacer una subdivisión) durante la generación
  mejora el resultado y actúa como una "poda" del árbol. En 3D un error de subdivisión cuesta ×8
  tokens (×4 en 2D), así que ahí debería pesar más.
- **Idea en exploración:** atención por ventanas combinada con un "pseudo temple simulado": llevar
  cuenta en tiempo de ejecución de zonas frías y calientes y recalcular solo las ventanas con nodos
  calientes para ahorrar cómputo.
- MNIST 2D no es el objetivo final; sirve para comparar variantes en minutos.

## Representación

- MNIST binarizado a 32×32, quadtree de profundidad 5. Unos 216 nodos por imagen frente a 1024 píxeles.
- Estados de nodo: `EMPTY=0`, `FULL=1`, `SPLIT=2`. Ids de entrada extra: `MASK=3`, `PAD=4`.
- `structures/pyramid.py` (`Pyramid`): representación densa de todos los niveles (1365 celdas) con
  `exists`, `to_sequence`, `to_image`, `sample_states`, priors.

## Métodos (en `adaptive_diffusion/models/`)

| Método | Fichero | Qué hace |
|---|---|---|
| `grow` | `adaptive_diffusion.py` | Desenmascarado nivel a nivel, 8 pasos por nivel, sin revisión. Es el esquema más parecido a OctGPT. |
| `edit` | `edit_diffusion.py` | Parte de un árbol aleatorio (prior `uniform`, `level` o `cell`) y lo edita; `SPLIT`→hoja es una unión. |
| `grow_merge` | `grow_merge.py` | Crece desde la raíz como `grow`, pero los nodos ya decididos se pueden revisar (subdividir tarde, unir, cambiar relleno). `--revise 0` lo deja en crecimiento puro. |

- `resolve_fresh` (`--resolve`) en `edit`: los nodos recién nacidos se resuelven con la red en el
  mismo paso en vez de quedarse como ruido del prior. Es lo que más mejora `edit`.
- `models/heat.py` (`HeatTrace`): medición **pasiva** de nodos calientes (enmascarados o con
  probabilidad del estado actual bajo un umbral) y de cuántos tokens habría que recalcular con
  ventanas. No altera las muestras.

## Estado de los resultados

Todo con 2000 muestras, **un solo modelo entrenado por configuración** y 3 semillas de muestreo, así
que las diferencias pequeñas no son concluyentes. Datos reales: 98,7–99,0 % de acierto, Fréchet 3,0–4,5.

| Caso | Acierto de clase | Fréchet | Tokens por muestra |
|---|---|---|---|
| `grow` | 94,5–95,1 % | 6,7–8,6 | ~3 000 |
| `edit cell` | 94,2–94,8 % | 12,1–13,5 | ~7 250 |
| `edit cell` + resolve, 32 pasos | 97,2–97,8 % | 5,6–6,4 | ~24 800 |
| `edit cell` + resolve, 16 pasos | 97,65 % | 6,56 | ~12 700 |
| `edit level` | 82,2–84,2 % | 60,8–65,0 | — |
| `edit uniform` | 60,3–64,4 % | 137–144 | — |

- El árbol aleatorio inicial pesa mucho: un modelo entrenado con `uniform` y arrancado desde un
  árbol `cell` sube a 89 %; al revés baja a 61 %.
- La pérdida de entrenamiento engaña: `uniform` tiene la pérdida más baja y las peores muestras.
- Más pasos sin resolve no ayudan a `edit`.
- **La temperatura importa** (barrido en `results/analysis/temperature_*.json`, tabla de arriba es con
  temperatura 1): `grow` a 0,85–0,9 da 96,4–97,3 % y Fréchet 4,7–6,2 con ~3 000–3 500 tokens;
  `edit cell` + resolve 32 pasos a 0,8 da 97,7–98,9 % / 5,0–5,5 con ~23 300 tokens; `edit cell` sin
  resolve a 0,7 da 96,8–97,2 % / 8,0–9,3. `grow` con temperatura ajustada es la referencia a batir.
- Reenmascarado estilo ReMDM sobre `grow` (`--remask`, solo el nivel en curso): por confianza 0,05 da
  96,9–97,2 % / 5,7–6,0, pero no supera al control de temperatura; uniforme no hace nada; no poda.
- Dato preliminar (500 muestras, `edit cell`): 20–43 % de nodos calientes, pero con ventanas de ~12
  tokens habría que recalcular el 86–92 %. En 32×32 las ventanas parecen demasiado gruesas.

### `grow_merge` entrenado (2 semillas, 30 000 pasos, 2026-11-13)

- Mejor caso (semilla 1, temperatura 0,9, con unión): 97,8–98,2 % / Fréchet 7,3–7,9. Semilla 0: 8,6–10,5.
- No supera a `grow` con temperatura ajustada (Fréchet 4,7–6,2): más acierto, menos variedad.
- La revisión actúa poco (~2,4 revisiones, ~1,6 uniones por imagen) y no poda (215–220 nodos).
- La varianza entre semillas de entrenamiento (~2 puntos de Fréchet) supera el efecto de la unión.
- Tabla completa en la sección 3.1 del README.

### `grow_merge` con errores propios (`--corrupt self|both`, 2 semillas cada uno, 2026-11-13)

- Medida directa (`experiments.correction_analysis`): detección de errores propios 35 % → 60 %,
  arreglo 34 % → 58–60 %, en ambas semillas.
- `both` (errores propios + ruido) es la mejor variante con unión: Fréchet 6,6–7,4 y 5,6–6,8.
  `self` sola empeora las muestras (8–11) aunque poda hasta ~208 nodos.
- Con temperatura 1 la revisión mejora mucho a `both`; con temperatura ajustada, un punto o menos.
- `grow` con 3 semillas de entrenamiento da 4,7–5,7, 8,1–9,0 y 5,2–6,0 (nodos 211, 227, 216): no se
  distingue de `both`. La varianza entre entrenamientos manda; MNIST 32×32 no decide. Tablas en README 3.2.

## 64×64: Quick, Draw! de trazo fino (evaluado 2026-11-14)

- Datos: `data/quickdraw.py`, 10 clases, 60 000 + 10 000 dibujos, trazo de 1 píxel, profundidad 6.
  Árboles de ~930 nodos de media (máx. ~1750) frente a 4096 píxeles. `--dataset quickdraw`.
- Clasificador de métricas: `results/classifier_quickdraw.pt`, 96,5 % en prueba; Fréchet de imágenes
  reales (entrenamiento frente a prueba) 15–18, ese es el suelo.
- Tiempos reales por entrenamiento de 30 000 pasos: `grow_merge` 112–130 min, `grow` ~135 min con
  `--workers 8`, `edit` ~125 min. La gráfica 1 va ~15 % más lenta. El muestreo es barato (~0,01 s por
  muestra): un barrido de 7 temperaturas × 3 semillas tarda pocos minutos.
- Mejor Fréchet por modelo (semilla 0 · 1), temperatura barrida: `grow` 78–81 (T 0,7) · 128–135;
  `both` con unión 137–150 · 97–106 (T 0,8); `both` sin unión 175–189 · 125–133; `edit cell` + resolve
  ~200 con ~119 000 tokens; `edit cell` 267–295.
- Dentro del mismo modelo la unión ayuda mucho (a T=1 divide el Fréchet por tres) y actúa 11–17 veces
  por imagen; detección de errores propios 68–71 %, arreglo 68–71 %, falsa alarma 0,3–0,4 %.
- Frente a `grow` sigue sin haber ganador: las semillas se cruzan y la varianza entre entrenamientos
  es de ~50 puntos. `grow` genera árboles grandes (1 020–1 160 nodos), la unión algo cortos (~890).
- `edit` queda descartado a 64×64. Todo está lejos del suelo: el modelo (dim 128, 4 capas) parece
  quedarse corto. Tablas en README 3.3.

### Unión añadida a un `grow` ya entrenado (2026-11-14)

- `run_training --init ckpt` afina desde otro checkpoint (también de `grow` a `grow_merge`).
- Mismo `grow` de partida + 5 000 pasos a lr 3e-4: control `grow` 50–53 · 65–71 de Fréchet; afinado con
  unión (`both`) 92–97 · 115–119 con un 20–25 % más de tokens. **La unión pierde en las dos semillas.**
- La unión mejora a su propio modelo, pero entrenar para revisar empeora al generador más de lo que
  la revisión recupera. Tal como está planteada, la hipótesis principal no se sostiene en 2D.
- Lateral: el lr constante 1e-3 dejaba los modelos a medio converger; bajar el lr al final da la
  mayor mejora vista (78–81 → 50–53). Usar decaimiento de lr en cualquier entrenamiento futuro.
- Los modelos tienen 0,86 M de parámetros (dim 128, 4 capas).
- Tabla en README 3.4; comparativa visual en `results/analysis/comparativa_quickdraw.png`.

### Corrector aparte sobre `grow`, a igual cómputo (2026-11-14)

- Criterio del usuario: una componente de revisión sólo vale si mejora **a igual coste computacional**;
  si no, ese coste se gasta en un `grow` mayor. Comparar siempre a tokens por muestra iguales.
- `experiments.train_corrector --gen <grow> --out <dir>`: red aparte del mismo tamaño, generador
  intacto; `temperature_sweep --corrector <ckpt> [--corr_thr --corr_rounds --tag]`. Actúa al acabar
  cada nivel, antes de crear los hijos; cada pasada cuenta como una evaluación.
- Resultado (Fréchet medio, s0 · s1, mismo presupuesto de tokens): `grow` solo 51–52 · 67–70;
  `grow` + 5 000 pasos más 50–51 · 63–64; `grow` + corrector 44–46 · 54–58. Gana en las dos semillas
  y en todos los presupuestos (10 000–17 000 tokens). Primera variante de revisión que gana a igual coste.
- No poda (3–5 uniones por imagen, 980–1 000 nodos) y sólo detecta ~19 % de los errores. Memoria no
  igualada (dos redes). Tabla en README 3.5.

### Cierre de la línea de unión/revisión (2026-11-14)

- `grow` de 8 capas (`qd_grow_L8_ft*`, doble de parámetros, doble de coste por token) frente al
  corrector a igual cómputo (tokens × capas/4): empate en ~10 000; a ~17 000 gana el grande
  (38 · 46 frente a 45 · 54) y sigue mejorando (32 · 38 con 8 pasos). Tabla en README 3.6.
- **Línea de unión/revisión cerrada por decisión del usuario**: no compensa frente a escalar `grow`.
  No retomarla salvo que lo pida.
- `run_training --accum N` acumula gradiente (la gráfica de 16 GB no aguanta 8 capas con lote 256;
  con `CUDA_VISIBLE_DEVICES=1` PyTorch ve la de 16 GB, al revés que `nvidia-smi`).
- Tiempo del `grow` de 8 capas: ~4 h 30 min los 30 000 pasos + ~45 min el afinado.
- Medida de la estructura (3 000 dibujos): 963 nodos para 234 píxeles llenos; 51 % hojas vacías; 81 %
  de los nodos en los dos últimos niveles; `FULL` casi no existe fuera del nivel de píxel; 131 bloques
  2×2 no vacíos con sólo 15 patrones. Propuesta siguiente: un token por nodo `SPLIT` con el patrón de
  ocupación de sus 4 hijos (~241 tokens en vez de 963; en 3D, un byte por nodo).
- Al usuario le interesan sobre todo mejoras de estructura de datos.

### Tokens de ocupación `tile` (2026-11-14) — línea activa

- `models/tile_grow.py`, `--method tile`: un token por nodo con contenido; valor = patrón de ocupación
  de sus 4 hijos (16 símbolos, `T_MASK=16`, `T_PAD=17`). ~246 tokens por dibujo frente a ~963.
  `QuadtreeTransformer(num_states=16)`. OctGPT no hace esto: usa un token binario por nodo
  (`split_size = 2`) y códigos VQ en las hojas.
- Resultado (4 capas, misma receta que `grow`): Fréchet 32,3 · 33,4 a T 0,9 con ~3 170 tokens por
  muestra. Iguala al `grow` de 8 capas (32 · 38 con 13 450 tokens ×2) con ~8 veces menos cómputo.
  Entrena en ~40 min. 8 pasos por nivel bastan. Tabla en README 3.7.
- `tile` de 8 capas (`qd_tile_L8_ft*`): Fréchet 22,2 · 24,9 a T 0,9 con ~3 170 tokens ×2 (acierto
  94–95 %). Óptimo de temperatura estrecho. Entrena en ~1 h 10 min + 8 min. Es la mejor marca.
- **Novedad (revisado 2026-11-14): el token de ocupación NO es nuevo.** OctSqueeze (2020, compresión
  LiDAR) ya usa el símbolo de 8 bits por nodo con softmax de 256; Uni-3DAR (arXiv 2503.16278) lo usa
  para generar ("2-level subtree compression", ~8× menos tokens), pero autorregresivo causal. No
  verificados (sólo resumen de búsqueda): OAT/OctreeGPT 2504.02817, Nexus 2607.13563, MeshOctave
  2609.38985 (difusión discreta enmascarada por escala con ocupación de 8 bits, en mallas). Lo que
  queda por diferenciar es la combinación con desenmascarado paralelo por nivel y el estudio de diccionario.
- **JoVox (arXiv 2611.10690, 7 nov 2026, KAUST) cubre casi toda nuestra línea en 3D** (leído): un token
  por celda ocupada, códigos de ocupación de 256 valores, predicción a dos niveles (4×4×4 como 8
  códigos de 256), flujo discreto enmascarado **conjunto en todos los niveles** (no nivel a nivel) y
  reenmascarado de los códigos más antiguos, que puede deshacer decisiones gruesas. Atención global
  con RoPE, 1,5 mil M de parámetros, 64³ (256³ en apéndice). Leídos también: OAT (código estructural de
  8 bits + VQ, autorregresivo BFS), Nexus (patrón de 8 como un token, difusión continua por niveles),
  MeshOctave (difusión enmascarada por escala, sobre mallas).
- Hueco que queda (2026-11-14): nadie de estos usa atención local ni recalcula sólo lo que cambia
  (JoVox reevalúa todos los tokens en ~80 pasos con atención global), y ninguno compara la revisión a
  igual cómputo. Para un paper haría falta 3D (ShapeNet 64³ como mínimo) con esas líneas base.
- Plan acordado: (1b) hecho; (2) atención por ventanas a 64×64 (la
  atención es global y no escala; OctGPT usa ventanas en orden Morton con dilatación); (3) subir a
  128×128 o 3D. El usuario quiere estudiar además el tamaño de diccionario (bloques 2×2 → 4×4).
- El usuario no quiere tandas de muchas horas: dar estimación de tiempo antes de lanzar.

### Zonas calientes, generación conjunta y atención local (2026-11-15) — línea activa

- Dirección elegida por el usuario: zonas calientes ("temple simulado") para no reevaluar todo en cada paso.
- `models/tile_joint.py`: representación densa (`Grid`), `CachedNet` (evalúa sólo las celdas pedidas con
  las demás como contexto guardado), `sample` (conjunto, estilo JoVox) y `sample_levels` (nivel a nivel
  sobre modelos `tile`, con `heat`, `lazy`, `order`). Barridos: `experiments/joint_sweep.py` y
  `experiments/levels_sweep.py`. No hace falta reentrenar para usar zonas calientes.
- Resultado: calor 1 = mitad de celdas sin pérdida; + perezosa = un tercio (4 capas sin pérdida, 8 capas
  −2,5 puntos). Calor 0 rompe. Orden o reenmascarado por confianza: descartados (hunden la variedad).
- Generación conjunta (`train_joint.py`, `qd_joint*`): pierde frente a nivel a nivel a igual cómputo
  (40 · 36,5 con ~3 300 celdas frente a 32–33). Reenmascarar lo antiguo aporta 0–4 puntos.
- Atención local (`--local 2 1`, `qd_tile_loc*_ft`): 36,3 · 32,0 con 8,5× menos pares de atención; con
  calor 2 + perezosa 32,6 · 32,9 con ~19× menos pares. Tabla en README 3.8.
- 128×128: `--dataset quickdraw128` (~519 tokens por dibujo, máx. >2 000), clasificador
  `results/classifier_quickdraw128.pt` (96,3 %), suelo de Fréchet 19,5–22,4. Con 128×128 hace falta
  `--batch 128 --accum 2` (memoria). La gráfica de 16 GB no aguanta entrenar y muestrear a la vez.

- 128×128, 4 capas (semilla 0 · 1): global 74,2 · 77,1 con 2,9 M de pares; local 76,1 · 82,3 con 161 000
  (18× menos; a 64×64 eran 8,5×); las zonas calientes no cuestan calidad. Todo lejos del suelo (~20).
- Temperatura por nivel: los niveles gruesos mandan (0,9); en los dos finos 0,8 da ~1–1,5 puntos.
- Tiempo real: zonas calientes 1,4–1,9× más rápido (no 2–3×); la atención local no ahorra tiempo en
  esta implementación (máscara sobre atención densa). Falta una implementación dispersa.
- `--ctx`: estado del padre + vecinas existentes como entrada (`qd_tile_ctx*`, `qd_tile_locctx*`).

## Pendiente

1. A 64×64: más semillas y/o modelo mayor o más pasos para bajar la varianza entre entrenamientos
   antes de comparar `grow` con la unión.
2. Cabeza de calidad por nodo (ingrediente 3) si la detección se queda corta.
3. Caché por token / zonas frías y calientes (ingrediente 4) sobre árboles de 64×64.
4. `sampling_analysis` y `heat_analysis` completos siguen sin ejecutarse dentro del repo.
5. Al comparar métodos, barrer siempre la temperatura (`experiments.temperature_sweep`).

## Cómo se ejecuta

Desde `adaptive_diffusion/`, con `/opt/conda/bin/python3`:

```bash
python -m pytest -q tests
python -m experiments.run_local --name grow_merge --method grow_merge
python -m experiments.evaluate --ckpt results/edit_cell/model.pt --resolve
python -m experiments.heat_analysis --ckpt results/edit_cell/model.pt --resolve
python -m experiments.sampling_analysis
```

- Resultados en `adaptive_diffusion/results/<nombre>/`; análisis en `results/analysis/`.
- `evaluate` escribe `metrics{_resolve}{_sin_union}.json` para no pisar `metrics.json`.
- `octGPT/` (paper y código de referencia) está en `.gitignore`.

## Máquina

- GPU0: RTX 5070 Ti 16 GB (algo más rápida). GPU1: RTX PRO 4500 32 GB.
- La gráfica **es** el cuello de botella al entrenar: varios trabajos en la misma gráfica o un lote
  mayor no aumentan el rendimiento. Un trabajo por gráfica; AMP activado (2× más rápido).
- `grow_merge`: ~30–37 ms por paso, 30 000 pasos ≈ 15–18 min (extrapolado de 300 pasos).

## Normas de trabajo

- Entrenamientos y análisis: el 2026-11-13 el usuario dio vía libre para lanzarlos cuando hagan falta
  ("tienes vía libre para entrenar cuando y como lo necesites"). Antes pedía esperar a su señal; si las
  gráficas pueden estar ocupadas por otra cosa, comprobar con `nvidia-smi` antes de lanzar.
- No hacer push sin permiso (el del 2026-11-13 fue autorizado para esa tanda). `origin` es HTTPS sin credenciales; el push funciona por SSH
  (`git@github.com:carlshue/QDD2D.git`).
- No usar `pkill -f` con un patrón que coincida con la propia orden; usar `pgrep -af "[p]atrón"`.
- El usuario escribe en español; responder en español.
- Documento explicativo (conceptos, ingredientes, orden de experimentos):
  https://claude.ai/artifact/99Jrjo3TTaf3263b4dFZx1
