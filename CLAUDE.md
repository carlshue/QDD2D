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
- Sigue sin superar a `grow` a 0,9 (4,7–5,7, una sola semilla de entrenamiento). Tablas en README 3.2.

## Pendiente

1. Más semillas de `grow` (solo hay una) para saber si `both` lo iguala o no.
2. Pasar a un problema mayor (64×64 de trazo fino o vóxeles 3D), con `grow`, `grow_merge both` y
   `edit cell` + resolve; decidir el conjunto de datos con el usuario.
3. Cabeza de calidad por nodo (ingrediente 3) si la detección del 60 % se queda corta.
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
