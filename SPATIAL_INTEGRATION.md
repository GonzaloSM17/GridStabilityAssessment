# Mapas espaciales de controladores PLL

El ZIP de referencia se uso como guia de los mapas; este modulo adapta el flujo a las clases reales de GridStabilityAssessment. No requiere MonteCarloAnalyzer, CSV de casos anterior, net_variations.npy ni etiquetas unstable_peak/slope/rapid.

## Ejecucion con selectores de archivos

Desde la raiz del repositorio:

```powershell
python spatial_playground.py
```

Solicita una carpeta de corrida, el Excel de activos y la carpeta de salida mediante selectores. La carpeta contiene `parameters.csv` separado por punto y coma y un unico `.parquet`. Para un estudio completo:

```powershell
python -m montecarlo.spatial_stability --mode study
```

Solicita la carpeta con `manifest.json` y `runs/run_XXXXXX/`, el Excel y la carpeta de salida. Tambien admite rutas explicitas para ejecucion sin dialogos:

```powershell
python -m montecarlo.spatial_stability --run "C:/ruta/run_000001" --assets "C:/ruta/Assets.xlsx" --output "C:/ruta/salida"
python -m montecarlo.spatial_stability --mode study --study "C:/ruta/estudio" --assets "C:/ruta/Assets.xlsx"
```

`--fault-time` y `--clearance-time` configuran la contingencia (por defecto 1.0 y 1.12 s). Las dependencias del repositorio estan en `requirements.txt`.

## Integracion con los workbenches

```python
result = workbench.create_spatial_stability()
```

Existe en `SingleRunWorkbench` y `MonteCarloWorkbench`. El primero solicita Assets si falta. En `single_run_playground.py` ahora se elige el Excel al inicio y se reutilizan los calculos ya realizados:

```python
spatial_result = workbench.create_spatial_stability(
    voltage=voltage_result.analysis,
    damping=damping_result.analysis,
    angular=angular_result.analysis,
)
```

Los playgrounds solicitan una carpeta de salida antes de analizar. En `single_run_playground.py`, todos los resultados van a la carpeta elegida y los mapas a `04_spatial_stability/`; en `playground.py`, los mapas van a `spatial_stability/`. El modo espacial independiente tambien crea `04_spatial_stability/` dentro de la carpeta seleccionada o indicada por `--output`. Las llamadas directas al workbench conservan la ruta anterior `analysis_outputs` si no se selecciona otra mediante `select_output_directory()`.

## Datos y criterio

- `parameters.csv`: GeneratorName, InService, Kp, Ki. Se incluyen las filas activas con ambos parametros, aunque sean constantes entre corridas. UnitId/UnitID desambigua controladores repetidos. Sin identidad unica se informa un error.
- Hoja `Unit Manager`: emtp_name, App. Power [MVA], Inertia [s] (requisito de Assets existente), posX y posY. La inercia de las maquinas sincronas se utiliza en el calculo angular.
- La equivalencia confirmada PFV_POZOALMONTE_HUAYCA -> PFV_LA_HUAYCA_II se aplica para buscar coordenadas y potencia; nombres exactos tienen prioridad. No hay coincidencias difusas. Los nombres normalizados ambiguos se rechazan.
- xi = Kp/(2*sqrt(Ki)); fn [Hz] = sqrt(Ki)/(2*pi), igual que data.run.read_active_controller_parameters.
- La etiqueta del caso usa SingleRunVoltageStability, SingleRunDampingStability y SingleRunAngularStability, y finalmente stability.dynamic_stability.assess_dynamic_stability. Se conserva la precedencia y los estados del codigo existente. Datos insuficientes se registran en diagnostics y pueden dar Not evaluated; nunca se convierten silenciosamente en Stable.
- Este resultado espacial incluye el criterio angular tambien en el estudio completo. El informe Monte Carlo preexistente conserva su comportamiento anterior de tension + amortiguamiento: no se sobrescribe su decision ni se reutiliza como si incluyera angulo.

## Resultados

```
xi_maps/by_case/xi_map_run_000001.png
fn_maps/by_case/fn_map_run_000001.png
xi_maps/by_stability/stable/xi_mean_map.png
fn_maps/by_stability/unstable/fn_mean_map.png
```

Solo PNG, sin subcarpetas por caso, tablas laterales o identificadores. Color por xi o bandas de fn [0,2), [2,5), [5,10), [10,20), >=20 Hz. Area del punto proporcional a MVA. Las escalas son comunes a todo el estudio. No se fuerzan distancias geograficas: posX/posY son posiciones del esquema.

Tambien hay grupos not_evaluated, separados de Stable/Unstable. Las clases sin casos se muestran sin valores. Los promedios por unidad dan igual peso a cada corrida en la que la unidad esta activa. Una unidad ausente no vale cero; n_cases y n_cases_in_class permiten revisar la cobertura. Los controladores que comparten un activo se reparten su potencia nominal entre las filas activas. El promedio del conjunto es ponderado por MVA dentro de cada caso y luego promediado entre los casos del grupo.

Los CSV device_case_parameters, case_assessments, excluded_devices y stability_group_summary, mas spatial_metadata.json, permiten revisar valores, exclusiones, diagnosticos, rutas y criterios. Si faltan coordenadas, se informa cada exclusion; si ninguna unidad puede representarse, el analisis falla con un mensaje claro.

## Verificacion

```powershell
python -B -m unittest discover -q
```

Incluye pruebas de conversion Hz/xi, equivalencias de nombres, unidades repetidas, limites de bandas, muestras no evaluables, reutilizacion de analisis, exportacion PNG y evaluacion nativa de Parquet de prueba. No es necesario iniciar los selectores para ejecutar las pruebas.

## Playground para multiples simulaciones

`python playground.py` ejecuta los analisis habituales y despues `workbench.create_spatial_stability()`. Guarda los archivos sin abrir ventanas de Matplotlib. Las figuras se cierran despues de guardarse para liberar memoria. El analisis espacial no imprime progreso por corrida; solo se muestra el resumen final. Los mapas espaciales se guardan en `<carpeta_salida_elegida>/04_spatial_stability/`, con `xi_maps` y `fn_maps` por corrida y por clase de estabilidad dinamica. Conserva las rutas y la seleccion de barras configuradas en playground.py; si context_path, assets_path u output_path es None, usa los selectores correspondientes. Cancelar la seleccion de salida detiene la ejecucion antes del analisis.

## Auditoria de promedios por estabilidad

La estructura es `04_spatial_stability/{xi_maps|fn_maps}/by_stability/{stable|unstable}/`, sin nivel dynamic intermedio. Si hay casos no evaluados, se conservan separados en not_evaluated; no se mezclan con estables. `cases.csv` enumera las corridas incluidas y sus tres resultados de criterio. `device_class_comparison.csv` presenta los dos promedios por unidad y sus diferencias. `spatial_metadata.json` registra la pertenencia exacta de cada grupo.

Los mapas xi agrupados usan una escala comun calculada con los promedios de los grupos; los mapas individuales conservan su escala comun por corrida. Los mapas fn mantienen las bandas fijas elegidas: valores numericos diferentes dentro de la misma banda tienen el mismo color. No se altera la clasificacion dinamica ni se cambia de banda para forzar diferencias visuales.
