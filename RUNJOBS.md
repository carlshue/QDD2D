# RUNJOBS — banco 3D: octree frente a omnitree

Instrucciones para ejecutar en la máquina local (2 GPU) la primera comparación 3D y devolver los
resultados. Están escritas para seguirse **al pie de la letra, en orden, sin improvisar**.

## 0. Reglas

1. **No modifiques código.** Si algo falla y este documento no dice qué hacer, para y cuéntaselo al usuario
   con el comando, las últimas 30 líneas de la salida y el paso en que estabas.
2. **Un trabajo por GPU.** Nunca dos entrenamientos en la misma gráfica.
3. **No hagas `git push` sin que el usuario lo autorice** en la conversación (paso 9).
4. Todos los comandos se lanzan desde `adaptive_diffusion/`, con `/opt/conda/bin/python3`.
   En este documento `PY` significa `/opt/conda/bin/python3`.
5. No uses `pkill -f` con un patrón que coincida con tu propia orden; usa `pgrep -af "[p]atrón"`.
6. Responde al usuario en español.

## 1. Traer el código

Desde la raíz del repositorio:

```bash
git status --short
```

Si sale algo que no sea de `results/`, **para y pregunta**. Si está limpio:

```bash
git fetch git@github.com:carlshue/QDD2D.git claude/keen-heisenberg-yew59l
git merge --ff-only FETCH_HEAD
```

Si `--ff-only` falla, **para y pregunta** (no hagas un merge normal ni un rebase).

Comprobación: deben existir estos ficheros.

```bash
ls adaptive_diffusion/experiments/run_box.py adaptive_diffusion/data/abo/abo32.npz adaptive_diffusion/structures/boxtree.py
```

## 2. Identificar las gráficas

```bash
cd adaptive_diffusion
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
/opt/conda/bin/python3 -c "import torch; [print(i, torch.cuda.get_device_name(i), round(torch.cuda.get_device_properties(i).total_memory / 2**30), 'GB') for i in range(torch.cuda.device_count())]"
```

- El segundo comando da los índices **tal como los ve PyTorch**, que son los que usa `CUDA_VISIBLE_DEVICES`
  (pueden ir al revés que en `nvidia-smi`).
- Apunta dos números: `G32` = índice de la gráfica de 32 GB, `G16` = índice de la de 16 GB.
- Si alguna gráfica tiene más de 1 GB de memoria usada, hay otro trabajo corriendo: **para y pregunta**.

En el resto del documento sustituye `G32` y `G16` por esos números.

## 3. Tests (menos de 1 minuto)

```bash
/opt/conda/bin/python3 -m pytest -q tests/test_boxtree.py tests/test_box_model.py
```

Debe terminar con `10 passed`. Si no, para.

## 4. Prueba rápida de `omni` (unos 5 minutos)

```bash
CUDA_VISIBLE_DEVICES=G16 /opt/conda/bin/python3 -m experiments.run_box --name prueba3d_omni --tree omni --steps 300 --limit 500 --num 200
```

Tiene que cumplirse todo esto:

- La primera línea dice `GPU detectada: ...` (no el aviso de que no hay GPU).
- Aparece `precisión mixta sí`.
- Aparece `entrenando el clasificador 3D de evaluación (una sola vez)...`: se guarda en
  `results/classifier_abo32.pt` y ya no se vuelve a entrenar.
- Termina con `Listo. Resultados en results/prueba3d_omni`.
- Existe `results/prueba3d_omni/metrics_box.json`.

Apunta `T_OMNI` = los segundos de la línea `paso    300 | pérdida ... | NNNs | ...` de
`results/prueba3d_omni/train.log`.

**Hazla antes que la de `tile` y no las lances a la vez**: las dos entrenarían el clasificador al mismo tiempo.

## 5. Prueba rápida de `tile` y ajuste de memoria (unos 5 minutos)

Se hace en la gráfica de 16 GB, que es la que tiene menos memoria:

```bash
CUDA_VISIBLE_DEVICES=G16 /opt/conda/bin/python3 -m experiments.run_box --name prueba3d_tile --tree tile --steps 300 --num 200
```

(Sin `--limit`: usa todas las formas para que aparezcan las secuencias largas. La primera vez calcula los
árboles `tile`, que tarda menos de un minuto.)

- Si termina bien: `LOTE_TILE` = `--batch 64 --accum 2`.
- Si falla con `out of memory`: repite el mismo comando añadiendo `--batch 32 --accum 4`. Si ahora termina:
  `LOTE_TILE` = `--batch 32 --accum 4`.
- Si vuelve a fallar: repite con `--batch 16 --accum 8`. Si termina: `LOTE_TILE` = `--batch 16 --accum 8`.
- Si falla también así, para y pregunta.

Los tres ajustes tienen el mismo lote efectivo (128), así que la comparación sigue siendo justa.

Apunta `T_TILE` = los segundos de la línea `paso    300` de `results/prueba3d_tile/train.log` (de la
ejecución que terminó bien).

## 5b. Preparar los árboles `omni` (3–8 minutos, sólo procesador)

Los dos entrenamientos `omni` usan el mismo fichero de árboles. Se calcula ahora, una sola vez:

```bash
/opt/conda/bin/python3 -c "from experiments.box_common import make_dataset; d = make_dataset('omni', 32, True, 8, 2048, workers=8); print(len(d), 'árboles')"
```

Debe imprimir `45392 árboles` (5 674 formas × 8) y dejar un fichero `omni_32_train_a8_m2048_n5674.npz` en
`data/abo/cache/`. Si el número es otro, apúntalo en el informe y sigue.

## 6. Decidir el número de pasos

Calcula la duración prevista de un entrenamiento completo de `tile`:

```
MIN_TILE = T_TILE / 300 × 20000 / 60        (minutos)
```

- Si `MIN_TILE` ≤ 120: `PASOS` = 20000.
- Si 120 < `MIN_TILE` ≤ 240: `PASOS` = 10000.
- Si `MIN_TILE` > 240: **para y pregunta** al usuario, dándole `T_OMNI`, `T_TILE` y `MIN_TILE`.

`PASOS` es el mismo para los cuatro entrenamientos. Dile al usuario qué valores han salido antes de seguir.

## 7. Los cuatro entrenamientos

Dos cadenas en paralelo, una por gráfica. Cada cadena hace primero `tile` y luego `omni`. Sustituye `G32`,
`G16`, `PASOS` y `LOTE_TILE`.

Cadena de la gráfica de 32 GB:

```bash
CUDA_VISIBLE_DEVICES=G32 nohup sh -c "/opt/conda/bin/python3 -m experiments.run_box --name box_tile_s0 --tree tile --seed 0 --steps PASOS LOTE_TILE && /opt/conda/bin/python3 -m experiments.run_box --name box_omni_s0 --tree omni --seed 0 --steps PASOS" > results/cadena_g32.out 2>&1 &
```

Cadena de la gráfica de 16 GB:

```bash
CUDA_VISIBLE_DEVICES=G16 nohup sh -c "/opt/conda/bin/python3 -m experiments.run_box --name box_tile_s1 --tree tile --seed 1 --steps PASOS LOTE_TILE && /opt/conda/bin/python3 -m experiments.run_box --name box_omni_s1 --tree omni --seed 1 --steps PASOS" > results/cadena_g16.out 2>&1 &
```

Para ver cómo van:

```bash
tail -n 3 results/cadena_g32.out results/cadena_g16.out
```

- Cada 100 pasos sale una línea con `faltan ~X min`.
- Cada 1000 pasos sale `pérdida en prueba`.
- No las mires más de una vez cada 10 minutos; no hace falta hacer nada mientras corren.
- Si una cadena se para con un error, **no relances nada**: espera a que acabe la otra y cuéntalo en el informe.

## 8. Comprobar que ha terminado

Las dos cadenas han acabado cuando sus `.out` terminan en `Listo. Resultados en results/box_omni_s0` y
`...box_omni_s1`. Entonces:

```bash
ls results/box_tile_s0 results/box_tile_s1 results/box_omni_s0 results/box_omni_s1
```

En cada carpeta deben estar: `train.log`, `eval.log`, `metrics_box.json`, `model.pt`, `env.json`.

Tabla resumen (cópiala en el informe):

```bash
/opt/conda/bin/python3 -c "
import json, glob
for p in sorted(glob.glob('results/box_*/metrics_box.json')):
    for r in json.load(open(p)):
        print(p.split('/')[1], 'T', r['temperatura'], '| acierto', r['acierto_clase_pct'], '| Frechet', r['frechet'], '| 1-NNA', r['nna_pct'], '| cobertura', r['cobertura_pct'], '| copia', r['copia'], '/', r['copia_prueba'], '| tokens', r['tokens_por_muestra'], '| nodos', r['nodos'], '| truncados', r['truncados'], '| vacias', r['vacias'])
"
```

## 9. Subir los resultados

**Pregunta primero al usuario** si puedes hacer commit y push. Si dice que sí, desde la raíz del repositorio:

```bash
git add adaptive_diffusion/results/box_tile_s0 adaptive_diffusion/results/box_tile_s1 adaptive_diffusion/results/box_omni_s0 adaptive_diffusion/results/box_omni_s1 adaptive_diffusion/results/prueba3d_omni adaptive_diffusion/results/prueba3d_tile adaptive_diffusion/results/classifier_abo32.pt adaptive_diffusion/results/cadena_g32.out adaptive_diffusion/results/cadena_g16.out
git commit -m "Resultados 3D: tile y omni, 2 semillas (ABO 32^3)"
git push git@github.com:carlshue/QDD2D.git HEAD:main
```

No añadas `adaptive_diffusion/data/abo/cache/` (está en `.gitignore`; son cientos de MB).

## 10. Informe al usuario

Un mensaje con:

1. `G32`, `G16`, `T_OMNI`, `T_TILE`, `MIN_TILE`, `PASOS` y `LOTE_TILE`.
2. Duración real de cada entrenamiento (`minutos_entrenamiento` de cada `env.json`).
3. La tabla del paso 8.
4. La última línea `pérdida en prueba` de cada `train.log`, junto a la última línea `pérdida` de entrenamiento.
5. Cualquier error, aviso o cosa rara. No interpretes los resultados ni saques conclusiones: sólo los datos.

## Tiempos previstos

Son **estimaciones sin medir**: este código no se ha ejecutado nunca en GPU. Salen de extrapolar los tiempos
de los entrenamientos 2D del proyecto. El error puede ser de un factor 2 en cualquier sentido; las pruebas
rápidas (pasos 4 y 5) dan la cifra real.

| Paso | Estimación |
|---|---|
| 1–3. Código, gráficas, tests | 5 min |
| 4. Prueba rápida `omni` (incluye entrenar el clasificador) | 5 min |
| 5. Prueba rápida `tile` | 5 min (10–15 si hay que repetir por memoria) |
| 5b. Árboles `omni` | 3–8 min |
| 7. Un entrenamiento `tile` + su evaluación | 70–120 min |
| 7. Un entrenamiento `omni` + su evaluación | 40–60 min |
| **7. Las dos cadenas en paralelo** | **2–3 horas** |
| 8–10. Comprobar, subir, informe | 10 min |
| **Total** | **unas 2 h 30 min – 3 h 40 min** |

- Las dos gráficas trabajan a la vez, así que el total lo marca una cadena (`tile` + `omni`), no la suma de las cuatro.
- La de 16 GB es algo más rápida; la de 32 GB tiene más margen de memoria. Por eso el ajuste de memoria se
  hace en la de 16 GB y se usa el mismo en las dos.
- Con una sola gráfica serían entre 4 y 6 horas.
- Si en el paso 6 sale `PASOS` = 10000, los entrenamientos duran la mitad.
