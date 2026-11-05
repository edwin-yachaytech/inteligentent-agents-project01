# Proyecto 01 — Agentes Inteligentes

**Autor:** edwin-yachaytech. Q-learning tabular y DQN implementados desde sus componentes con Gymnasium, NumPy y PyTorch. Entrega individual: Tier 1 en CartPole-v1 y Tier 2 en Acrobot-v1. Tier 3 es opcional y no se implementó.

El repositorio contiene código, configuraciones, dependencias y resultados. El informe y las diapositivas se entregan por separado.

## Instalación

Versión probada: Python 3.12.14, CPU. Desde la raíz del repositorio:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

En Windows, activar con `.venv\Scripts\activate`. Los comandos con variables de entorno usan sintaxis Linux/macOS; Google Colab también permite ejecutarlos en Linux.

## Reproducir los experimentos

Las configuraciones exactas están en [experiments/configs.json](experiments/configs.json). Cada configuración se ejecuta con tres semillas independientes (0, 1 y 2) y 200.000 pasos por entrenamiento.

**Tier 1: CartPole con Q-learning (6, 10 y 20 bins por dimensión) y DQN.**

```bash
python -m experiments.suite --group tier1 --workers 2
```

**Tier 2: Acrobot, estudios de tasa de aprendizaje y exploración, y ablación sin red objetivo.**

```bash
python -m experiments.suite --group tier2 --workers 2
```

**Toda la matriz de 30 entrenamientos:**

```bash
python -m experiments.suite --group all --workers 2
```

**Una ejecución concreta:**

```bash
python -m experiments.train --config acrobot_default --seed 0
```

Los resultados completos ya están incluidos. Se omiten ejecuciones terminadas con `summary.json` y la misma configuración. La reanudación es por ejecución completa, no por paso. Para repetir una ejecución desde cero sin reemplazar resultados publicados, usar otra carpeta:

```bash
python -m experiments.train --config acrobot_default --seed 0 --root results/repeated
```

Para repetir toda la matriz desde cero, usar una copia sin las carpetas anteriores de las diez configuraciones. Si se cambia una configuración conservando su nombre, ejecutar `experiments.train` con un `--root` nuevo; el programa rechaza reutilizar resultados incompatibles.

## Regenerar figuras y GIF

Una vez completados los grupos, este comando reproduce todas las figuras desde los CSV:

```bash
MPLCONFIGDIR=/tmp/rl_mpl python -m experiments.analyze
```

| Figura | Comando del experimento |
|---|---|
| [tier1_evaluation.png](results/figures/tier1_evaluation.png) | `python -m experiments.suite --group tier1 --workers 2` |
| [tier1_training.png](results/figures/tier1_training.png) | `python -m experiments.suite --group tier1 --workers 2` |
| [tier2_learning_rate.png](results/figures/tier2_learning_rate.png) | `python -m experiments.suite --group tier2 --workers 2` |
| [tier2_exploration.png](results/figures/tier2_exploration.png) | `python -m experiments.suite --group tier2 --workers 2` |
| [tier2_ablation.png](results/figures/tier2_ablation.png) | `python -m experiments.suite --group tier2 --workers 2` |
| [tier2_training.png](results/figures/tier2_training.png) | `python -m experiments.suite --group tier2 --workers 2` |

Para regenerar los GIF sin pantalla:

```bash
SDL_VIDEODRIVER=dummy python -m experiments.video
```

Ejemplos: [CartPole Q-learning](results/videos/cartpole_q_10.gif), [CartPole DQN](results/videos/cartpole_dqn.gif) y [Acrobot DQN](results/videos/acrobot_default.gif). Se elige la semilla de entrenamiento cercana a la mediana de las medias finales y un episodio fijo independiente. La selección está documentada en [metadata.json](results/videos/metadata.json). Los videos son ilustrativos; las conclusiones usan las tres semillas.

## Protocolo

- Evaluación al inicio y cada 10.000 pasos: 10 episodios greedy, en otro entorno, sin aprendizaje ni consumo del RNG de entrenamiento.
- Evaluación final al paso 200.000: 100 episodios nuevos por semilla. Se evalúa la política final; no se selecciona el mejor checkpoint.
- Entrenamiento y evaluación se guardan por separado. El eje X es pasos del entorno.
- Las bandas son **una desviación estándar entre las medias de tres semillas**, no intervalos de confianza ni desviaciones entre episodios.
- Acrobot: tasa de aprendizaje 0.0003/0.001/0.003 y fracción de decaimiento de epsilon 0.25/0.50/0.80, cambiando un parámetro a la vez y compartiendo el punto base. La ablación retira la red objetivo y conserva replay.
- `first_validation_threshold_step` registra cruces en evaluaciones cortas de 10 episodios. El éxito final sobre 100 episodios se informa por separado.
- Solo `terminated` elimina bootstrap; una truncación por tiempo conserva el valor de la observación final, guardada antes del reinicio.
- La evaluación no cuenta en el presupuesto de pasos de entrenamiento, pero sí consume tiempo.
- Dos procesos comparten aproximadamente dos CPU y 8 GiB, con un hilo PyTorch por proceso. Los tiempos por ejecución incluyen evaluación y guardado; no son benchmarks universales.

## Resultados medidos

Media final ± desviación estándar entre tres semillas; cada semilla se evalúa en 100 episodios:

| Configuración | Retorno final | Éxito final |
|---|---:|---:|
| CartPole Q-learning, 6 bins | 227.15 ± 162.55 | 0/3 |
| CartPole Q-learning, 10 bins | 292.10 ± 179.28 | 1/3 |
| CartPole Q-learning, 20 bins | 327.40 ± 151.12 | 0/3 |
| CartPole DQN | 84.39 ± 12.68 | 0/3 |
| Acrobot DQN base | −88.70 ± 5.88 | 3/3 |
| Acrobot sin red objetivo | −500.00 ± 0.00 | 0/3 |

Umbrales de la consigna: 475 para CartPole y −100 para Acrobot. Se conservan los resultados negativos. Las métricas de todas las configuraciones, incluidos los barridos, varianzas, tiempos y cruces, están en [aggregate.csv](results/aggregate.csv) y [aggregate.json](results/aggregate.json).

DQN mejoró antes en CartPole, pero no conservó ese avance; las tablas sostuvieron mayor rendimiento durante el presupuesto y mostraron alta variabilidad. En Acrobot, retirar la red objetivo deterioró fuertemente el resultado final. Tres semillas y una arquitectura limitan la generalización.

## Estructura y comprobaciones

```text
agents/          Q-learning, discretización y DQN con replay y red objetivo
experiments/     configuraciones, entrenamiento, evaluación, figuras y GIF
tests/           pruebas de Bellman, replay y sincronización
results/         registros, métricas, pesos, curvas y GIF de los 30 entrenamientos
README.md        instalación y comandos de reproducción
requirements.txt dependencias probadas
```

Cada `results/<config>/seed_<n>/` conserva configuración, CSV de entrenamiento y evaluación, 100 retornos finales, resumen y pesos o tabla Q. El resumen incluye hardware y versiones.

Verificar los 30 entrenamientos, sus 21 checkpoints por ejecución, evaluaciones finales, modelos y correspondencia de las métricas:

```bash
python -m experiments.check_results
```

También se comprobó una repetición tabular exacta con la misma semilla: [reproducibility_check.json](results/reproducibility_check.json). Las huellas de los artefactos están en [integrity_sha256.json](results/integrity_sha256.json).

## Uso de IA

Se utilizó ChatGPT/Codex para implementación, pruebas, ejecución y documentación. El estudiante debe comprender y poder defender el código. No se usan agentes prefabricados de Stable-Baselines3 u otras bibliotecas de agentes.
