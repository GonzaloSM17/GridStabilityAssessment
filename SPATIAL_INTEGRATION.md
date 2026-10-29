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

Existe en `SingleRunWorkbench` y `MonteCarloWorkbench`. Forma parte del flujo principal de ambos workbenches, sin detener los analisis electricos si sus metadatos no son evaluables. El primero solicita Assets si falta. En `single_run_playground.py` se elige el Excel al inicio y se reutilizan los calculos ya realizados:

```python
spatial_result = workbench.create_spatial_stability(
    voltage=voltage_result.analysis,
    damping=damping_result.analysis,
    angular=angular_result.analysis,
)
```

Los playgrounds solicitan una carpeta de salida antes de analizar. En `single_run_playground.py`, todos los resultados van a la carpeta elegida y los mapas a `06_Spatial_Stability/`; en `playground.py`, los mapas van a `04_spatial_stability/`. Las llamadas directas al workbench conservan la ruta `analysis_outputs` si no se selecciona otra mediante `select_output_directory()`.

## Datos y criterio

- `parameters.csv`: GeneratorName, InService, Kp, Ki. Se incluyen las filas activas con ambos parametros, aunque sean constantes entre corridas. UnitId/UnitID desambigua controladores repetidos. Sin identidad unica se informa un error.
- Hoja `Unit Manager`: emtp_name, App. Power [MVA], Inertia [s] (requisito de Assets existente), posX y posY. La inercia de las maquinas sincronas se utiliza en el calculo angular.
- La equivalencia confirmada PFV_POZOALMONTE_HUAYCA -> PFV_LA_HUAYCA_II se aplica para buscar coordenadas y potencia; nombres exactos tienen prioridad. No hay coincidencias difusas. Los nombres normalizados ambiguos se rechazan.
- xi = Kp/(2*sqrt(Ki)); fn [Hz] = sqrt(Ki)/(2*pi), igual que data.run.read_active_controller_parameters.
- Monte Carlo usa stability.dynamic_stability.assess_monte_carlo_stability: Unstable si la tension es Unstable O el amortiguamiento es Growing. Damped under (positivo y menor al 5 %) queda en el grupo Stable cuando la tension es estable, conservando el detalle en modal_damping_status. Si no hay fallo conocido y falta evidencia, queda Not evaluated.
- En playground.py se reutilizan las mismas evaluaciones del informe para las barras seleccionadas; no se relee Parquet para reclasificar ni se evalua el angulo. La llamada espacial de estudio sin analisis previo evalua tension/amortiguamiento para todas las barras disponibles y registra ese alcance.
- SingleRunWorkbench conserva su evaluacion independiente de tension, amortiguamiento y angulo mediante assess_dynamic_stability.
- spatial_metadata.json registra include_angle y voltage_scope. Los lectores conservan el criterio del archivo guardado: no reinterpretan silenciosamente resultados antiguos con angulo.

## Resultados

```
xi_maps/by_case/xi_map_run_000001.png
fn_maps/by_case/fn_map_run_000001.png
xi_maps/by_stability/stable/xi_mean_map.png
fn_maps/by_stability/unstable/fn_mean_map.png
```

Solo PNG, sin subcarpetas por caso, tablas laterales o identificadores. Color por xi o bandas de fn [0,2), [2,5), [5,10), [10,20), >=20 Hz. Area del punto proporcional a MVA. Las escalas son comunes a todo el estudio. No se fuerzan distancias geograficas: posX/posY son posiciones del esquema.

Tambien hay grupos not_evaluated, separados de Stable/Unstable. Las clases sin casos se muestran sin valores. Los promedios por unidad dan igual peso a cada corrida en la que la unidad esta activa. Una unidad ausente no vale cero; n_cases y n_cases_in_class permiten revisar la cobertura. Los controladores que comparten un activo se reparten su potencia nominal entre las filas activas. El promedio del conjunto es ponderado por MVA dentro de cada caso y luego promediado entre los casos del grupo. Si faltan parametros o coordenadas para una corrida, se registra como advertencia en `analysis.log` y en la hoja `Spatial Stability` del reporte; el resto de las corridas mapeables continua.

Los CSV device_case_parameters, case_assessments, excluded_devices y stability_group_summary, mas spatial_metadata.json, permiten revisar valores, exclusiones, diagnosticos, rutas y criterios. Si faltan coordenadas, se informa cada exclusion; si ninguna unidad puede representarse, el analisis falla con un mensaje claro.

## Verificacion

```powershell
python -B -m unittest discover -q
```

Incluye pruebas de conversion Hz/xi, equivalencias de nombres, unidades repetidas, limites de bandas, muestras no evaluables, reutilizacion de analisis, exportacion PNG y evaluacion nativa de Parquet de prueba. No es necesario iniciar los selectores para ejecutar las pruebas.

## Playground para multiples simulaciones

`python playground.py` ejecuta los analisis habituales, incluido espacial como etapa no bloqueante. Guarda los archivos sin abrir ventanas de Matplotlib. Las figuras se cierran despues de guardarse para liberar memoria. El analisis espacial no imprime progreso por corrida; el archivo `analysis.log` y el resumen final informan su estado. Los mapas espaciales se guardan en `<carpeta_salida_elegida>/04_spatial_stability/`, con `xi_maps` y `fn_maps` por corrida y por clase de estabilidad dinamica. Conserva las rutas y la seleccion de barras configuradas en playground.py; si context_path, assets_path u output_path es None, usa los selectores correspondientes. Cancelar la seleccion de salida detiene la ejecucion antes del analisis.

## Auditoria de promedios por estabilidad

La estructura Monte Carlo es `04_spatial_stability/{xi_maps|fn_maps}/by_stability/{stable|unstable}/`, sin nivel dynamic intermedio. Si hay casos no evaluados, se conservan separados en not_evaluated; no se mezclan con estables. `cases.csv` enumera las corridas incluidas y los resultados de criterio y el detalle Damped under; angular_status queda vacio en Monte Carlo. `device_class_comparison.csv` presenta los dos promedios por unidad y sus diferencias. `spatial_metadata.json` registra la pertenencia exacta de cada grupo.

Los mapas xi agrupados usan una escala comun calculada con los promedios de los grupos; los mapas individuales conservan su escala comun por corrida. Los mapas fn mantienen las bandas fijas elegidas: valores numericos diferentes dentro de la misma banda tienen el mismo color. No se altera la clasificacion dinamica ni se cambia de banda para forzar diferencias visuales.

## Distribuciones por estabilidad

Se generan automaticamente en `04_spatial_stability/distributions_by_stability/`, usando las mismas etiquetas del flujo que genero los mapas (Monte Carlo: tension + Growing, sin angulo; Single Run: incluye angulo). Para generar solo estas figuras desde resultados guardados, sin releer Parquet ni regenerar mapas:

```powershell
python -m montecarlo.spatial_stability.distributions --results "C:/ruta/04_spatial_stability"
```

- `01_case_parameter_distributions.png`: ECDF y cajas con puntos de los promedios por MVA de xi y fn de cada caso. Cada punto representa una corrida.
- `02_capacity_share_distributions.png`: distribucion entre casos de la fraccion MVA con xi < 0.7, fn >= 10 Hz o ambas condiciones. Los umbrales son exploratorios, no reglas de estabilidad. Se pueden configurar con `--low-xi-threshold` y `--high-fn-threshold-hz`.
- `03_population_mva_distributions.png`: ECDF de valores por unidad, ponderando por fraccion MVA dentro de cada caso y dando igual peso a cada caso dentro de su clase. No confunde miles de unidades/casos con miles de corridas independientes.
- `04_technology_case_distributions.png`: distribuciones de promedios por caso dentro de cada tecnologia, definida por el prefijo GeneratorName (por ejemplo BESS, PFV, PE, PMGD).

Los CSV registran indicadores por caso, por tecnologia, cuantiles y curvas poblacionales. Se muestran los tamanos de muestra, se conservan clases ausentes sin inventar valores y se incluyen no evaluados en gris solo cuando existen. Son graficos descriptivos, sin inferencia causal ni pruebas de significancia. Los mapas y planos anteriores conservan su formato.


## Clustering de corridas con diez indicadores

Se ejecuta automaticamente al crear los resultados espaciales con playground.py.
La salida queda en 04_spatial_stability/clustering/. No abre ventanas ni imprime
progreso por corrida. Tambien se puede ejecutar solo el clustering, reutilizando
los CSV espaciales existentes y sin releer Parquet:

```powershell
python clustering_playground.py
python -m montecarlo.spatial_stability.clustering --results "C:/ruta/04_spatial_stability" --output "C:/ruta/04_spatial_stability"
```

Sin rutas, solicita la carpeta de resultados espaciales y la carpeta de salida;
crea clustering/ dentro de la salida elegida.

Las diez entradas son weighted_xi_BESS, weighted_xi_PE, weighted_xi_PFV,
weighted_xi_PMGD, weighted_fn_hz_BESS, weighted_fn_hz_PE, weighted_fn_hz_PFV,
weighted_fn_hz_PMGD, xi_spread y fn_hz_spread. Los ocho promedios usan MVA
dentro de cada tecnologia; las dos dispersiones son desviaciones estandar MVA
sobre todos los dispositivos incluidos. Cada corrida pesa igual. Las clases
dinamicas del flujo activo solo se comparan despues del ajuste; en Monte Carlo no se evalua el angulo.

Requiere las cuatro tecnologias en cada caso; los casos incompletos se documentan
en excluded_cases.csv, sin rellenar con ceros. Con menos de cuatro casos completos,
o menos de tres perfiles distintos, se guarda el motivo sin ajustar grupos.

Se estandarizan las diez variables y se comparan k-means y PCA + k-means.
PCA conserva por defecto el 90 % de su variabilidad, no solo dos componentes.
Se prueban k=2..6, limitados por los datos, y se descartan grupos de un solo caso.
Se elige k por silhouette en el espacio original de los diez indicadores
estandarizados para comparar ambos metodos con la misma distancia.
Se registran tambien silhouette del espacio de ajuste y ARI de 20 submuestras
con preprocesado reestimado. Silhouette >= 0.25 y ARI mediana >= 0.60 son
filtros descriptivos configurables, no pruebas de significancia ni garantias
de que existan familias fisicas. Si no pasan, el resultado se identifica como
no_well_defined_groups y la particion se conserva como exploratoria.

Opciones CLI: --max-clusters, --pca-variance, --seed,
--silhouette-threshold y --ari-threshold.

- 01_model_selection.png: silhouette por k y variabilidad acumulada PCA.
- 03_cluster_stability.png: cantidad de casos estables, inestables y no evaluados.
- 04_pca_projection.png: mismas corridas en PC1-PC2, con colores por grupo
  y por estabilidad; los ejes informan la variabilidad representada.
- 05_pca_groups_and_stability.png: color por grupo y forma por estabilidad en cada metodo.
- case_features.csv y case_clusters.csv: indicadores y pertenencia por corrida.
- cluster_profiles.csv: perfiles medios en unidades originales.
- cluster_stability.csv: composicion de estabilidad y fallas por criterio,
  que pueden solaparse.
- model_selection.csv, pca_variance.csv, pca_loadings.csv y
  clustering_metadata.json: ajustes, escalado, semillas y diagnosticos.

### Lectura conjunta de grupos y estabilidad

Se elimina 02_cluster_profiles.png. Se conservan 01, 03 y 04, y se agrega
05_pca_groups_and_stability.png: dos paneles (kmeans y pca_kmeans) con color por
grupo y forma por estabilidad dinamica. Circulo = estable; X = inestable;
rombo = no evaluado solo si existe. Cada punto sigue siendo una corrida.
Las coordenadas y colores de grupo coinciden con el grafico 04.

En el grafico 03 la leyenda muestra estable/inestable. Si hay casos no evaluados,
se conservan como barras grises y se informa su cantidad en el titulo del panel,
sin incorporarlos a ninguno de los dos estados ni incluirlos en la leyenda.

PCA_DESCRIPTION.md explica la estandarizacion y la combinacion de los diez
indicadores con una tabla de coeficientes PC1/PC2 y sus porcentajes de variabilidad.
Se recalcula con cada estudio; pca_loadings.csv conserva los coeficientes completos.
PC1 y PC2 son combinaciones lineales, no coordenadas espaciales ni criterios de estabilidad.


### Porcentajes por tecnologia e inestabilidad por indicadores fisicos

Se incorporan automaticamente tres PNG adicionales en clustering/:

- 06_weighted_pca_projection.png: nuevos ejes PCA ponderados por unidades y MVA.
- 07_physical_instability.png: proporcion de corridas inestables frente a fraccion
  de unidades/MVA con xi < 0.7 y fn media, con ambas ponderaciones.
- 08_technology_instability.png: los mismos mapas por tecnologia, frente al
  resultado dinamico de cada caso completo.

Los limites por cuartiles no usan las etiquetas de estabilidad y son comunes
entre ponderaciones dentro de cada poblacion. Cada celda informa porcentaje y
conteo inestables/evaluados; se raya si tiene menos de cinco evaluados. Vacio
no equivale a estable. No evaluados quedan fuera del denominador y se cuentan
separadamente. xi < 0.7 es exploratorio, no un limite operativo.

TECHNOLOGY_ANALYSIS.md detalla las formulas, denominadores y alcances.
Los CSV conservan composicion por caso, porcentajes de coeficientes y conteos
de celdas con intervalos Wilson 95 %. --low-xi-threshold y --min-cell-cases
permiten ajustar el umbral exploratorio y el aviso de muestra reducida.


### Correccion: pesos aplicados antes de PCA

El grafico 06 ahora es 06_weighted_pca_projection.png y reemplaza el reparto
posterior de coeficientes. Usa los mismos diez indicadores con medias y
dispersiones calculadas por unidades o MVA. Se estandarizan, se multiplican por
la raiz de la fraccion de poblacion representada por cada indicador y se vuelve
a ajustar PCA. Los indicadores por tecnologia usan su fraccion n_t/n_total
o MVA_t/MVA_total promediada entre corridas; las dispersiones globales usan 1
porque representan el conjunto. No se estandariza otra vez despues del peso.

El color conserva las etiquetas originales de kmeans/pca_kmeans; el simbolo
indica estabilidad. Este cambio recalcula los ejes, no los grupos.
WEIGHTED_PCA_DESCRIPTION.md y weighted_pca_*.csv detallan entradas, pesos,
coordenadas, coeficientes efectivos y variabilidad. PCA_DESCRIPTION.md describe
el PCA original de los graficos 04/05; la descripcion ponderada es independiente.

En el grafico 07 se explicita que un panel calcula fraccion y promedio por
numero de unidades y el otro por MVA. Los intervalos de celda son comunes,
pero cada corrida puede cambiar de celda al cambiar su ponderacion.
El porcentaje de color siempre usa inestables/(estables+inestables), con igual
peso por corrida. No se grafican solo las corridas inestables.


### Dos grupos nuevos segun xi y su dispersion

Se agrega 09_xi_groups_weighted_pca.png y se mantiene el grafico 05.
Los nuevos grupos se ajustan exclusivamente con xi medio y desviacion estandar
de xi del conjunto de unidades de cada caso. Estos dos indicadores se
estandarizan y se aplica k-means con exactamente dos grupos e igual peso
por corrida. No se usa fn ni la estabilidad en la nueva separacion.

Cada ponderacion (unidades/MVA) forma sus propios grupos. G1 tiene menor
xi medio de grupo y G2 mayor; la relacion de dispersion se describe con los
promedios reales, sin forzar que un grupo tenga mayor o menor dispersion.
Los nombres son relativos a la muestra, no umbrales operativos.

Arriba se proyectan los nuevos grupos sobre los mismos ejes PCA ponderados
de diez indicadores del grafico 06. Abajo se muestra xi medio frente a
dispersion para entender la separacion. El color identifica los grupos nuevos
y el simbolo la estabilidad dinamica nativa. Se exportan pertenencias,
centroides, composicion de estabilidad, silhouette y ARI de submuestras.

XI_GROUPS_DESCRIPTION.md detalla el metodo y resume los perfiles.
Para generar solo el grafico nuevo sin tocar los anteriores:

```powershell
python -m montecarlo.spatial_stability.xi_groups --results "C:/ruta/04_spatial_stability" --output "C:/ruta/04_spatial_stability"
```

Sin rutas se solicitan carpetas de entrada y salida mediante selectores.


## Caracterizacion amplia sin PCA

playground.py ejecuta tambien el analisis supervisado adicional sobre los resultados
espaciales y guarda una carpeta independiente:
04_spatial_stability/stability_characterization/.
Los mapas y el grafico 05 de clustering mantienen su formato y sus indicadores.
La clasificacion del flujo activo nunca se reemplaza por predicciones: Monte Carlo usa tension o Growing; Damped under pertenece al grupo Stable si la tension es estable.

Se comparan logistica regularizada, logistica con interacciones, arbol pequeno,
bosque aleatorio, boosting, SVM radial y Naive Bayes. Cada familia prueba:
diez indicadores originales; nueve sin dispersion fn; nueve mas correlacion xi-fn.
Se usan ambos esquemas de agregacion (unidades/MVA); no se aplica PCA.
Los casos no evaluados y las tecnologias incompletas se excluyen con motivo.
La correlacion indefinida se imputa exclusivamente dentro del entrenamiento.

Validacion anidada estratificada, cinco particiones externas repetidas cinco veces
y tres internas. Familia, parametros y conjunto de indicadores se eligen dentro
del entrenamiento por log loss, incluyendo prevalencia como candidato.
La fila nested_selection mide ese procedimiento completo.
Las otras filas son comparaciones exploratorias de opciones prefijadas.
La seleccion de nueve indicadores ya fue motivada por esta muestra: hace falta
confirmacion externa con simulaciones nuevas, aunque se use validacion anidada.

Se guardan cinco PNG: comparacion de familias, aporte por bloques,
curvas de validacion, arboles descriptivos y diferencias de parametros por clase.
Los arboles dibujados se ajustan con toda la muestra y no representan validacion.
Los CSV conservan entradas, exclusiones, predicciones externas, parametros
seleccionados y dispersion entre particiones (no es un intervalo de confianza).

Para ejecutar solo este analisis, sin releer Parquet ni regenerar mapas:

```powershell
python characterization_playground.py
python -m montecarlo.spatial_stability.characterization --results "C:/ruta/04_spatial_stability" --output "C:/ruta/04_spatial_stability"
```

Sin rutas abre selectores de entrada/salida. --repeats configura repeticiones;
--families permite elegir un subconjunto de familias.
En playground.py, run_characterization controla esta etapa y
characterization_repeats configura su coste. No hay progreso por corrida
ni ventanas Matplotlib; la ejecucion completa puede tardar varios minutos.


### Perfiles estables e inestables dentro de cada hoja

04_descriptive_trees.png ahora muestra en cada hoja los conteos de ambas clases,
sus nueve medianas xi/fn y dispersion xi, y la diferencia de medianas I-E.
Los datos describen los casos de la hoja, sin crear divisiones adicionales
ni cambiar su clasificacion. Se mantienen profundidad 2 y los mismos umbrales.

Se agregan 04_descriptive_trees_units.png y 04_descriptive_trees_mva.png para
facilitar la lectura. Cada hoja informa fallos/evaluados de tension,
amortiguamiento modal y angulo entre sus inestables. Los criterios pueden
solaparse; una evidencia no evaluada no se cuenta como un cumplimiento.
Clases ausentes: --. Clases con menos de cinco corridas: aviso de poca muestra.
I-E es una diferencia descriptiva, no un coeficiente ni un efecto causal.

tree_leaf_profiles.csv conserva medianas y rangos; tree_leaf_summary.csv
conteos y criterios; tree_leaf_membership.csv pertenencias. TREE_PROFILES.md
explica la lectura. Para actualizar solo estos arboles usando resultados guardados,
sin repetir la comparacion de modelos ni modificar los mapas:

```powershell
python characterization_playground.py --trees-only --results "C:/ruta/04_spatial_stability" --output "C:/ruta/04_spatial_stability"
```


## Compatibilidad con corridas omitidas del analisis espacial

El flujo principal reutiliza spatial_stability_result calculado por analyze();
no vuelve a ejecutar el analisis espacial. Si no hay puntos utilizables se
omite la caracterizacion adicional sin detener el reporte electrico.

Las etiquetas nativas de todas las corridas se conservan en case_assessments.csv.
Los estudios de xi/fn solo usan corridas con controladores representables;
las omitidas se documentan y nunca se reemplazan por valores cero ni estables.
El cargador admite los registros spatial_status=Skipped del exportador nuevo
y conserva su razon. Los conteos minimos de mapas de indicadores fisicos
solo incluyen corridas con datos de controladores.

Monte Carlo conserva 04_spatial_stability/; el flujo individual usa
06_Spatial_Stability/ como en la estructura actual del repositorio.


## Arboles globales y figura de promedios MVA

04_descriptive_trees_mva.png y las versiones units/combinada ahora se ajustan
exclusivamente con cuatro indicadores de la poblacion incluida en cada corrida:
xi medio, fn media, desviacion estandar de xi y desviacion estandar de fn.
En MVA se usa w_i = MVA_i / suma(MVA), promedio = suma(w_i * parametro_i)
y dispersion = sqrt(suma(w_i * (parametro_i - promedio)**2)).
Cada corrida pesa igual al ajustar el arbol. Los nodos y las tablas de las hojas
usan estos mismos cuatro indicadores, sin divisiones por tecnologia. Las hojas
siguen mostrando medianas entre corridas de cada clase, conteos y fallos.
No se exige que esten presentes BESS/PE/PFV/PMGD; el arbol se genera aunque el
analisis de modelos por tecnologia no pueda ajustarse. Casos no evaluados y sin
controladores se excluyen y documentan en tree_excluded_cases.csv.
tree_case_features.csv conserva los cuatro valores por corrida y ponderacion.

distributions_by_stability/05_case_weighted_means.png conserva solo los dos
paneles inferiores de 01_case_parameter_distributions.png: cajas con puntos para
xi y fn medios ponderados por MVA. Cada punto es una corrida; se conserva la
figura 01 completa. La figura nueva tiene un archivo de metadatos independiente
con criterio, alcance de barras y recuentos por clase.

Ambos resultados usan las etiquetas del flujo activo. En Monte Carlo no
participa el angulo: tension Unstable o amortiguamiento Growing es Unstable;
Damped under pertenece al grupo Stable si la tension es estable. No evaluados
quedan separados y no se convierten en estables.


Los casos Not evaluated se conservan en los CSV y metadatos de auditoria, pero se excluyen de puntos, cajas, curvas, barras, mapas y leyendas. Las proyecciones conservan sus ajustes no supervisados; solo se ocultan las observaciones sin etiqueta evaluada. No se convierten a Stable.
