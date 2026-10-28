"""PCA coordinates fitted with unit-count or MVA feature weights."""
from dataclasses import replace
from pathlib import Path
import json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .clustering import FEATURES, TECHNOLOGIES, build_case_profiles
from .distributions import _csv, LABELS, STATUSES


def weighted_profiles(analysis, case_ids, basis):
    """Keep the ten indicators; use consistent within-technology unit/MVA means."""
    if basis not in {"units", "mva"}:
        raise ValueError("Weighting basis must be units or mva")
    ids = set(case_ids)
    selected = replace(analysis,
        points=tuple(p for p in analysis.points if p.case_id in ids),
        cases=tuple(c for c in analysis.cases if c.case_id in ids))
    if basis == "units":
        selected = replace(selected, points=tuple(
            replace(p, nominal_power_mva=1.) for p in selected.points))
    profiles = build_case_profiles(selected)
    if set(profiles.case_ids) != ids:
        raise ValueError("Weighted PCA requires all four technologies in every selected case")
    return profiles


def feature_population_weights(composition, basis):
    if basis not in {"units", "mva"}:
        raise ValueError("Weighting basis must be units or mva")
    rows = {row["technology"]: row for row in composition}
    key = "unit_share" if basis == "units" else "mva_share"
    # Each technology indicator represents its population fraction.
    # Each global spread represents the whole population and has weight 1.
    return np.array([rows[t][key] for metric in ("xi", "fn_hz")
                     for t in TECHNOLOGIES] + [1., 1.])


def fit_weighted_pca(values, feature_weights):
    """Fit on B = StandardScaler(X) * sqrt(population feature fractions).

    Multiplication AFTER standardization preserves the physical metric:
    squared distance = sum_j(a_j * (delta z_j)**2). Never rescale B to unit variance.
    """
    x = np.asarray(values, dtype=float)
    weights = np.asarray(feature_weights, dtype=float)
    if x.ndim != 2 or x.shape[1] != 10 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("Weighted PCA requires at least two finite ten-indicator profiles")
    if weights.shape != (10,) or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Weighted PCA requires ten positive finite feature weights")
    scaler = StandardScaler().fit(x)
    b = scaler.transform(x) * np.sqrt(weights)
    if not np.any(np.var(b, axis=0) > 0):
        return dict(status="not_fitted", reason="All weighted indicators are constant.")
    with threadpool_limits(limits=1):
        model = PCA(svd_solver="full").fit(b)
        scores = model.transform(b)
    return dict(
        status="fitted", feature_weights=weights.tolist(),
        mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist(),
        weighted_center=model.mean_.tolist(),
        components=model.components_.tolist(),
        effective_standardized_coefficients=(model.components_ * np.sqrt(weights)).tolist(),
        explained_variance_ratio=model.explained_variance_ratio_.tolist(),
        scores=scores.tolist())


def export_weighted_pca(analysis, out, result, composition):
    """New coordinates, original cluster assignments; report this distinction."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib import colormaps

    out = Path(out)
    cases = {c.case_id: c.assessment.status for c in analysis.cases}
    ids = result["included_case_ids"]
    methods = result["methods"]
    fitted, inputs, coordinate_rows, weight_rows, coefficient_rows, variance_rows = {}, [], [], [], [], []
    for basis in ("units", "mva"):
        profiles = weighted_profiles(analysis, ids, basis)
        # Follow the same case order as original cluster labels.
        lookup = {c: row for c, row in zip(profiles.case_ids, profiles.values)}
        values = np.array([lookup[c] for c in ids])
        weights = feature_population_weights(composition, basis)
        model = fit_weighted_pca(values, weights)
        fitted[basis] = model
        for c, row in zip(ids, values):
            inputs.append(dict(basis=basis, case_id=c, **dict(zip(FEATURES, row))))
        for feature, weight in zip(FEATURES, weights):
            weight_rows.append(dict(basis=basis, feature=feature, population_weight=float(weight),
                                    multiplier=float(np.sqrt(weight))))
        if model["status"] != "fitted":
            continue
        for index, c in enumerate(ids):
            row = dict(basis=basis, case_id=c, status=cases[c])
            row.update({f"PC{j+1}": v for j, v in enumerate(model["scores"][index])})
            row.update({method+"_original_cluster": info["labels"][index]
                        for method, info in methods.items()})
            coordinate_rows.append(row)
        cumulative = 0.
        for j, (coefficients, effective, variance) in enumerate(zip(
                model["components"], model["effective_standardized_coefficients"],
                model["explained_variance_ratio"])):
            coefficient_rows.append(dict(basis=basis, component=j+1,
                                         **dict(zip(FEATURES, coefficients))))
            coefficient_rows.append(dict(basis=basis+"_effective_on_z", component=j+1,
                                         **dict(zip(FEATURES, effective))))
            cumulative += variance
            variance_rows.append(dict(basis=basis, component=j+1,
                                      explained_variance=variance, cumulative_variance=cumulative))
    _csv(out/"weighted_pca_case_features.csv", inputs, ["basis", "case_id", *FEATURES])
    _csv(out/"weighted_pca_weights.csv", weight_rows, list(weight_rows[0]))
    _csv(out/"weighted_pca_coefficients.csv", coefficient_rows, ["basis", "component", *FEATURES])
    _csv(out/"weighted_pca_variance.csv", variance_rows,
         ["basis", "component", "explained_variance", "cumulative_variance"])
    if coordinate_rows:
        _csv(out/"weighted_pca_coordinates.csv", coordinate_rows, list(coordinate_rows[0]))
    metadata = dict(
        population_weights="Mean within-case technology shares; equal weight per simulation",
        formula="B = ((X-mean)/scale) * sqrt(a); PC = (B - PCA_center) @ component",
        dispersion_weights="Both global spreads have a=1 because each describes the whole population",
        clustering="Original kmeans and pca_kmeans assignments are overlaid; not refitted on new weights",
        models=fitted)
    (out/"weighted_pca_metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    fig = Figure(figsize=(14, 12), layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(2, len(methods), squeeze=False, sharex="row", sharey="row")
    palette = colormaps["tab10"]
    markers = {"Stable": "o", "Unstable": "X", "Not evaluated": "D"}
    status = np.array([cases[c] for c in ids])
    for row_index, basis in enumerate(("units", "mva")):
        model = fitted[basis]
        name = "Cantidad de unidades" if basis == "units" else "Potencia MVA"
        for axis, (method, info) in zip(axes[row_index], methods.items()):
            if model["status"] != "fitted":
                axis.text(.5, .5, "Indicadores constantes: sin ejes PCA", ha="center")
                axis.set_axis_off()
                continue
            scores = np.array(model["scores"])
            variance = model["explained_variance_ratio"]
            second = scores[:, 1] if scores.shape[1] > 1 else np.zeros(len(scores))
            second_variance = variance[1] if len(variance) > 1 else 0.
            labels = np.array(info["labels"])
            present = [s for s in STATUSES if np.any(status == s)]
            for c in range(1, info["k"]+1):
                for state in present:
                    mask = (labels == c) & (status == state)
                    if mask.any():
                        axis.scatter(scores[mask, 0], second[mask], marker=markers[state],
                                     s=55, color=palette((c-1) % 10), alpha=.9,
                                     edgecolor="white" if state == "Stable" else "#222222",
                                     linewidth=.6, zorder=3 if state == "Stable" else 2)
            legend = axis.legend(
                handles=[Patch(facecolor=palette((c-1) % 10), label=f"Grupo {c}")
                         for c in range(1, info["k"]+1)],
                title="Grupos originales", loc="upper left", fontsize=8)
            axis.add_artist(legend)
            axis.legend(handles=[
                Line2D([], [], marker=markers[s], linestyle="none", color="#777777",
                       markeredgecolor="#222222", label=LABELS[s]) for s in present],
                title="Estabilidad", loc="upper right", fontsize=8)
            axis.set(title=f"{name} | grupos de {method}\n"
                           f"PC1 + PC2: {100*(variance[0]+second_variance):.1f} %",
                     xlabel=f"PC1 ponderada ({100*variance[0]:.1f} %)",
                     ylabel=f"PC2 ponderada ({100*second_variance:.1f} %)")
            axis.grid(alpha=.2)
    fig.suptitle("PCA recalculado segun la poblacion de unidades y su potencia\n"
                 "Color: grupo original | Circulo: estable | X: inestable\n"
                 "Los pesos modifican los ejes; las etiquetas de estabilidad no entran al PCA")
    path = out/"06_weighted_pca_projection.png"
    try:
        fig.savefig(path, dpi=170)
    finally:
        fig.clear()
    lines = [
        "# Ejes PCA ponderados por unidades y MVA", "",
        "Este analisis reemplaza el reparto posterior de coeficientes del grafico 06 anterior. "
        "Ahora se recalculan las coordenadas PCA usando pesos de poblacion antes del ajuste.", "",
        "Se conservan los diez indicadores: ocho promedios xi/fn por tecnologia y dos "
        "dispersiones globales. En la version unidades se calculan con igual peso por unidad; "
        "en la version MVA los promedios y dispersiones se calculan ponderando por potencia.", "",
        "Para cada tecnologia t se obtiene a_t = promedio entre corridas de n_t/n_total "
        "o de MVA_t/MVA_total. Cada indicador de esa tecnologia usa ese peso. "
        "Las dos dispersiones globales usan a=1 porque describen el conjunto completo. "
        "Es una convencion explicita del analisis; los datos no determinan una unica ponderacion.", "",
        "Primero z_j=(X_j-media_j)/desviacion_j. Despues B_j=sqrt(a_j)*z_j. "
        "PCA se ajusta sobre B. No se vuelve a estandarizar despues de aplicar los pesos, "
        "porque eso anularia su efecto. El cuadrado de la distancia corresponde a "
        "sum(a_j*(delta z_j)^2). Los coeficientes PC1/PC2 se estiman con los datos, "
        "no se asignan manualmente ni equivalen directamente a porcentajes de unidades.", "",
        "Los colores conservan los grupos originales para comparar el cambio de ejes. "
        "No se recalcula k-means en este analisis adicional. Las posiciones, orientacion y "
        "porcentajes de variabilidad pueden cambiar entre unidades y MVA; no comparten "
        "necesariamente la misma base PCA. Los signos de los ejes pueden invertirse sin cambiar "
        "la informacion. Los ejes no son xi o Hz directamente ni causas de inestabilidad.", "",
    ]
    for basis, model in fitted.items():
        if model["status"] == "fitted":
            v = model["explained_variance_ratio"]
            lines += [
                f"- {basis}: PC1={100*v[0]:.2f} %, PC2={100*v[1]:.2f} %, "
                f"plano={100*sum(v[:2]):.2f} %.", "",
            ]
    lines += [
        "weighted_pca_weights.csv conserva los pesos y multiplicadores; "
        "weighted_pca_case_features.csv las entradas; weighted_pca_coordinates.csv "
        "las coordenadas; weighted_pca_coefficients.csv los coeficientes sobre B y sus "
        "equivalentes sobre z; weighted_pca_metadata.json incluye escalado y ajuste completo.",
        "",
    ]
    (out/"WEIGHTED_PCA_DESCRIPTION.md").write_text("\n".join(lines), encoding="utf-8")
    from .xi_groups import export_xi_groups
    new_groups = export_xi_groups(analysis, out, ids, pca_models=fitted,
                                 seed=result["config"]["seed"])
    return (path, *new_groups)
