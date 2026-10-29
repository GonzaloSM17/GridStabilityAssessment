"""Unsupervised case clustering of ten MVA-weighted PLL indicators.

Stability is an outcome for interpretation only, never a fitting feature.
"""
import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .distributions import (
    COLORS, LABELS, STATUSES, _csv, case_features, technology_features,
    load_saved_analysis,
)

TECHNOLOGIES = ("BESS", "PE", "PFV", "PMGD")
FEATURES = tuple(
    f"weighted_{metric}_{technology}"
    for metric in ("xi", "fn_hz") for technology in TECHNOLOGIES
) + ("xi_spread", "fn_hz_spread")


@dataclass(frozen=True)
class CaseProfiles:
    case_ids: tuple[str, ...]
    values: np.ndarray
    excluded: tuple[dict, ...]


def build_case_profiles(analysis):
    """Require all four technology means; never replace missing technology by 0."""
    summaries = case_features(analysis)
    technology = {
        (r["case_id"], r["technology"]): r for r in technology_features(analysis)
    }
    identifiers, values = [], []
    mapped_ids={point.case_id for point in analysis.points}
    skipped={item.case_id:item.reason for item in getattr(analysis,"skipped_runs",())}
    excluded=[dict(case_id=case.case_id,reason="No mapped controllers: "+
                   skipped.get(case.case_id,"No usable spatial controller metadata"))
              for case in analysis.cases if case.case_id not in mapped_ids]
    for row in summaries:
        identifier = row["case_id"]
        missing = [t for t in TECHNOLOGIES if (identifier, t) not in technology]
        if missing:
            excluded.append(dict(case_id=identifier,
                                 reason="Missing technology: " + ", ".join(missing)))
            continue
        identifiers.append(identifier)
        values.append([
            technology[identifier, t][f"weighted_{metric}"]
            for metric in ("xi", "fn_hz") for t in TECHNOLOGIES
        ] + [row["xi_spread"], row["fn_hz_spread"]])
    return CaseProfiles(tuple(identifiers),
                        np.asarray(values, dtype=float).reshape(-1, len(FEATURES)),
                        tuple(excluded))


def _fit(values, k, seed):
    return KMeans(n_clusters=k, n_init=30, random_state=seed).fit(values)


def _canonical_labels(labels, original):
    # Numbering is deterministic from physical profiles, never from stability.
    order = sorted(np.unique(labels),
                   key=lambda c: tuple(original[labels == c].mean(axis=0)))
    mapping = {int(c): i + 1 for i, c in enumerate(order)}
    return np.array([mapping[int(c)] for c in labels], dtype=int)


def fit_profiles(values, *, max_clusters=6, seed=42, pca_variance=.90,
                 silhouette_threshold=.25, ari_threshold=.60, resamples=20):
    """Compare raw standardized profiles and PCA profiles with the same distance audit.

    Thresholds are configurable descriptive screening rules, not hypothesis tests.
    Silhouette in the original standardized ten-feature space selects k for both
    methods; the PCA-space score is also retained, but cannot drive selection.
    """
    x = np.asarray(values, dtype=float)
    if x.ndim != 2 or x.shape[1] != len(FEATURES) or not np.isfinite(x).all():
        raise ValueError("Expected a finite matrix with ten indicator columns")
    if max_clusters < 2 or not 0 < pca_variance < 1 or resamples < 2:
        raise ValueError("Require max_clusters >= 2, 0 < PCA variance < 1, resamples >= 2")
    if not 0 <= silhouette_threshold <= 1 or not -1 <= ari_threshold <= 1:
        raise ValueError("Invalid descriptive quality thresholds")
    result = dict(n_cases=len(x), feature_names=list(FEATURES), methods={},
                  candidates=[], status="not_fitted", reason=None,
                  config=dict(max_clusters=max_clusters, seed=seed,
                              pca_variance=pca_variance,
                              silhouette_threshold=silhouette_threshold,
                              ari_threshold=ari_threshold, resamples=resamples))
    if len(x) < 4:
        result["reason"] = "At least four complete cases are required."
        return result
    if len(np.unique(x, axis=0)) < 3:
        result["reason"] = "Fewer than three distinct profiles; no cluster quality comparison."
        return result
    scaler = StandardScaler().fit(x)
    z = scaler.transform(x)
    result["scaler"] = dict(mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist(),
                            constant_features=[FEATURES[i] for i in range(len(FEATURES))
                                               if scaler.var_[i] == 0])
    with threadpool_limits(limits=1):
        pca = PCA(svd_solver="full").fit(z)
        variance = pca.explained_variance_ratio_
        retained = int(np.searchsorted(np.cumsum(variance), pca_variance)) + 1
        scores = pca.transform(z)
        result["pca"] = dict(retained_components=retained,
                             retained_variance=float(variance[:retained].sum()),
                             explained_variance_ratio=variance.tolist(),
                             components=pca.components_.tolist(),
                             scores=scores.tolist())
        for method, space in (("kmeans", z), ("pca_kmeans", scores[:, :retained])):
            candidates = []
            upper = min(max_clusters, len(x) - 1, len(np.unique(space, axis=0)))
            for k in range(2, upper + 1):
                model = _fit(space, k, seed)
                counts = np.bincount(model.labels_, minlength=k)
                if len(np.unique(model.labels_)) != k:
                    continue
                score = float(silhouette_score(z, model.labels_))
                record = dict(method=method, k=k, silhouette_original=score,
                              silhouette_fit_space=float(silhouette_score(space, model.labels_)),
                              inertia=float(model.inertia_), min_cluster_size=int(counts.min()),
                              eligible=bool(counts.min() >= 2), selected=False)
                result["candidates"].append(record)
                if record["eligible"]:
                    candidates.append((record, model))
            if not candidates:
                continue
            record, model = max(candidates, key=lambda pair: (pair[0]["silhouette_original"],
                                                              -pair[0]["k"]))
            record["selected"] = True
            labels = _canonical_labels(model.labels_, x)
            # Refit preprocessing on each subset; compare full-case predictions.
            rng = np.random.default_rng(seed)
            agreements = []
            sample_size = min(len(x) - 1, max(record["k"] + 1, int(.8 * len(x))))
            for trial in range(resamples):
                indices = rng.choice(len(x), sample_size, replace=False)
                subset_scaler = StandardScaler().fit(x[indices])
                full = subset_scaler.transform(x)
                subset = full[indices]
                if len(np.unique(subset, axis=0)) < record["k"]:
                    continue
                if method == "pca_kmeans":
                    subset_pca = PCA(n_components=pca_variance, svd_solver="full")
                    subset = subset_pca.fit_transform(subset)
                    full = subset_pca.transform(full)
                if len(np.unique(subset, axis=0)) < record["k"]:
                    continue
                fitted = _fit(subset, record["k"], seed + trial + 1)
                agreements.append(float(adjusted_rand_score(labels, fitted.predict(full))))
            median = float(np.median(agreements)) if agreements else None
            supported = (record["silhouette_original"] >= silhouette_threshold
                         and median is not None and median >= ari_threshold)
            result["methods"][method] = dict(
                k=record["k"], labels=labels.tolist(),
                silhouette_original=record["silhouette_original"],
                silhouette_fit_space=record["silhouette_fit_space"],
                resampling_ari_median=median, resampling_ari_values=agreements,
                conclusion=("passes_descriptive_screen" if supported
                            else "no_well_defined_groups"),
                cluster_sizes={str(c): int(np.sum(labels == c)) for c in np.unique(labels)})
    result["status"] = "fitted" if result["methods"] else "not_fitted"
    if not result["methods"]:
        result["reason"] = "No candidate had at least two cases in every cluster."
    return result


def _export_figures(out, profiles, result, cases):
    # Agg canvas owns the figures directly: no pyplot windows or global figures.
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib import colormaps

    paths = []
    methods = result["methods"]
    def figure(width, height):
        fig = Figure(figsize=(width, height), layout="constrained")
        FigureCanvasAgg(fig)
        return fig
    def save(fig, name):
        path = out / name
        try:
            fig.savefig(path, dpi=170)
        finally:
            fig.clear()
        paths.append(path)

    fig = figure(12, 5)
    axes = fig.subplots(1, 2)
    for method in methods:
        rows = [r for r in result["candidates"] if r["method"] == method]
        axes[0].plot([r["k"] for r in rows], [r["silhouette_original"] for r in rows],
                     marker="o", label=method)
    axes[0].axhline(result["config"]["silhouette_threshold"], color="gray", ls="--",
                    label="Referencia descriptiva configurable")
    axes[0].set(xlabel="Numero de grupos k", ylabel="Silhouette en los 10 indicadores estandarizados")
    axes[0].legend(fontsize=8)
    variance = np.asarray(result["pca"]["explained_variance_ratio"])
    axes[1].plot(np.arange(1, len(variance) + 1), 100 * np.cumsum(variance), marker="o")
    axes[1].axhline(100 * result["config"]["pca_variance"], color="gray", ls="--")
    axes[1].set(xlabel="Componentes PCA", ylabel="Variabilidad acumulada (%)", ylim=(0, 102))
    for axis in axes:
        axis.grid(alpha=.2)
    fig.suptitle("Seleccion exploratoria de k y comparacion PCA\n"
                 "Un mejor k dentro de los probados no demuestra grupos naturales")
    save(fig, "01_model_selection.png")

    fig = figure(12, 5)
    axes = np.asarray(fig.subplots(1, len(methods), squeeze=False)).ravel()
    for axis, (method, info) in zip(axes, methods.items()):
        labels = np.asarray(info["labels"])
        positions = np.arange(1, info["k"] + 1)
        bottom = np.zeros(info["k"])
        for status in STATUSES:
            counts = [sum(cases[identifier].assessment.status == status
                          for identifier, label in zip(profiles.case_ids, labels) if label == c)
                      for c in positions]
            if any(counts):
                axis.bar(positions, counts, bottom=bottom, color=COLORS[status],
                         label=LABELS[status] if status != "Not evaluated" else "_nolegend_")
            bottom += counts
        unknown = sum(cases[identifier].assessment.status == "Not evaluated"
                      for identifier in profiles.case_ids)
        title = method + (f"\nGris: {unknown} casos no evaluados" if unknown else "")
        axis.set(xticks=positions, xlabel="Grupo", ylabel="Numero de corridas", title=title)
        axis.legend(fontsize=8)
    fig.suptitle("Estabilidad dinamica por grupo: tension + amortiguamiento + angulo\n"
                 "Las etiquetas se comparan despues del agrupamiento")
    save(fig, "03_cluster_stability.png")

    scores = np.asarray(result["pca"]["scores"])
    y = scores[:, 1] if scores.shape[1] > 1 else np.zeros(len(scores))
    fig = figure(15, 5)
    axes = fig.subplots(1, len(methods) + 1, sharex=True, sharey=True)
    palette = colormaps["tab10"]
    for axis, (method, info) in zip(axes, methods.items()):
        labels = np.asarray(info["labels"])
        for c in range(1, info["k"] + 1):
            mask = labels == c
            axis.scatter(scores[mask, 0], y[mask], s=25, alpha=.75,
                         color=palette((c - 1) % 10), label=f"Grupo {c}")
        axis.set_title(method)
        axis.legend(fontsize=8)
    status_values = np.array([cases[c].assessment.status for c in profiles.case_ids])
    for status in STATUSES:
        mask = status_values == status
        if mask.any():
            axes[-1].scatter(scores[mask, 0], y[mask], s=25, alpha=.75,
                             color=COLORS[status], label=LABELS[status])
    axes[-1].set_title("Estabilidad observada")
    axes[-1].legend(fontsize=8)
    second = variance[1] if len(variance) > 1 else 0.
    for axis in axes:
        axis.set(xlabel=f"PC1 ({100 * variance[0]:.1f} %)",
                 ylabel=f"PC2 ({100 * second:.1f} %)")
        axis.grid(alpha=.2)
    fig.suptitle(f"Cada punto es una corrida; plano PCA de los diez indicadores\n"
                 f"PC1 + PC2 representa {100 * (variance[0] + second):.1f} %; "
                 "los grupos se ajustan en sus espacios completos")
    save(fig, "04_pca_projection.png")

    # Keep the same PCA axes and group colors, adding stability by marker shape.
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    markers = {"Stable": "o", "Unstable": "X", "Not evaluated": "D"}
    fig = figure(13, 6)
    axes = np.asarray(fig.subplots(1, len(methods), sharex=True, sharey=True,
                                   squeeze=False)).ravel()
    present = [status for status in STATUSES if np.any(status_values == status)]
    for axis, (method, info) in zip(axes, methods.items()):
        labels = np.asarray(info["labels"])
        for c in range(1, info["k"] + 1):
            for status in present:
                mask = (labels == c) & (status_values == status)
                if not mask.any():
                    continue
                axis.scatter(
                    scores[mask, 0], y[mask],
                    marker=markers[status], s=60 if status == "Stable" else 65,
                    facecolor=palette((c - 1) % 10),
                    edgecolor="white" if status == "Stable" else "#222222",
                    linewidth=.65, alpha=.9, zorder=3 if status == "Stable" else 2)
        group_handles = [
            Patch(facecolor=palette((c - 1) % 10), label=f"Grupo {c}")
            for c in range(1, info["k"] + 1)]
        group_legend = axis.legend(handles=group_handles, title="Color: grupo",
                                   loc="upper left", fontsize=8, title_fontsize=9)
        axis.add_artist(group_legend)
        status_handles = [
            Line2D([], [], linestyle="none", marker=markers[status],
                   markerfacecolor="#888888", markeredgecolor="#222222",
                   markersize=7, label=LABELS[status])
            for status in present]
        axis.legend(handles=status_handles, title="Forma: estabilidad",
                    loc="upper right", fontsize=8, title_fontsize=9)
        axis.set(title=method, xlabel=f"PC1 ({100 * variance[0]:.1f} %)",
                 ylabel=f"PC2 ({100 * second:.1f} %)")
        axis.grid(alpha=.2)
    fig.suptitle(
        "Grupos y estabilidad dinamica en el mismo plano PCA\n"
        "Circulo: estable | X: inestable"
        + (" | Rombo: no evaluado" if "Not evaluated" in present else "")
        + f"\nCada punto es una corrida; PC1 + PC2 representa "
          f"{100 * (variance[0] + second):.1f} %")
    save(fig, "05_pca_groups_and_stability.png")
    return tuple(paths)


def pca_description(result):
    """Explain the fitted directions using exact signed component coefficients."""
    pca = result["pca"]
    components = np.asarray(pca["components"])
    variance = pca["explained_variance_ratio"]
    names = [
        "xi medio BESS", "xi medio PE", "xi medio PFV", "xi medio PMGD",
        "fn media BESS [Hz]", "fn media PE [Hz]", "fn media PFV [Hz]",
        "fn media PMGD [Hz]", "Dispersion xi", "Dispersion fn [Hz]",
    ]
    lines = [
        "# Que combinan PC1 y PC2", "",
        "Los ejes se calculan con los diez indicadores de cada corrida, no con "
        "posX/posY ni con las etiquetas de estabilidad.",
        "Los ocho promedios y las dos dispersiones ya incorporan sus ponderaciones MVA. "
        "Despues se estandariza cada indicador entre corridas: "
        "z_j = (indicador_j - media_j) / escala_j. "
        "La escala es la desviacion estandar poblacional usada por StandardScaler "
        "(1 para una columna constante). Asi se combinan xi y Hz sin que sus unidades "
        "dominen por si solas.", "",
        "Cada coordenada se obtiene con PC_r = sum_j(coeficiente_rj * z_j). "
        "PCA vuelve a centrar z antes de proyectar; su media es cero salvo redondeo. "
        "Los coeficientes no son porcentajes ni efectos causales.", "",
        "| Indicador | Coeficiente PC1 | Coeficiente PC2 |",
        "|---|---:|---:|",
    ]
    for j, name in enumerate(names):
        second = f"{components[1, j]:+.6f}" if len(components) > 1 else "No disponible"
        lines.append(f"| {name} | {components[0, j]:+.6f} | {second} |")
    lines.append("")
    for index in range(min(2, len(components))):
        dominant = np.argsort(-np.abs(components[index]))[:4]
        contributions = ", ".join(
            f"{names[j]} ({components[index, j]:+.3f})" for j in dominant)
        lines += [
            f"PC{index + 1} representa {100 * variance[index]:.2f} % de la variabilidad. "
            f"Sus cuatro coeficientes de mayor magnitud son: {contributions}.",
            "",
        ]
    first_two = sum(variance[:2])
    lines += [
        "PC1 es la direccion que conserva la mayor variabilidad de los indicadores "
        "estandarizados. PC2 conserva la mayor variabilidad restante en una direccion "
        "perpendicular a PC1; sus coordenadas no estan correlacionadas linealmente "
        "con PC1 en los datos usados para ajustar PCA.", "",
        "Un coeficiente positivo aporta hacia la derecha (PC1) o hacia arriba (PC2); "
        "uno negativo aporta en sentido contrario, para valores z positivos. "
        "La posicion final suma todos los aportes: un punto a la derecha no implica "
        "que todos los indicadores positivos sean altos.", "",
        f"El plano PC1-PC2 representa {100 * first_two:.2f} % de la variabilidad; "
        f"PCA + k-means usa {pca['retained_components']} componentes y conserva "
        f"{100 * pca['retained_variance']:.2f} %. "
        "K-means directo usa los diez indicadores estandarizados. "
        "Ambos se dibujan sobre la misma proyeccion para compararlos.", "",
        "El signo de una componente puede invertirse sin cambiar la informacion. "
        "Los coeficientes se recalculan para cada estudio; no son constantes fisicas. "
        "Las direcciones describen variacion de parametros, no razones demostradas de "
        "inestabilidad. La estabilidad solo determina el simbolo del nuevo grafico 05.",
        "",
    ]
    return "\n".join(lines)


def export_clustering(analysis, output_directory, *, low_xi_threshold=.7, min_cell_cases=5, **options):
    """Save clustering/ under the selected spatial output, including skipped cases."""
    profiles = build_case_profiles(analysis)
    result = fit_profiles(profiles.values, **options)
    out = Path(output_directory) / "clustering"
    out.mkdir(parents=True, exist_ok=True)
    # Remove only known generated artifacts from an earlier module run.
    if (out / "clustering_metadata.json").is_file():
        for name in ("model_selection.csv", "case_clusters.csv", "cluster_profiles.csv",
                     "cluster_stability.csv", "pca_variance.csv", "pca_loadings.csv",
                     "01_model_selection.png", "02_cluster_profiles.png",
                     "03_cluster_stability.png", "04_pca_projection.png",
                     "05_pca_groups_and_stability.png", "PCA_DESCRIPTION.md"):
            path = out / name
            if path.is_file():
                path.unlink()
    cases = {c.case_id: c for c in analysis.cases}
    _csv(out / "case_features.csv",
         [dict(case_id=c, **dict(zip(FEATURES, row)))
          for c, row in zip(profiles.case_ids, profiles.values)],
         ["case_id", *FEATURES])
    _csv(out / "excluded_cases.csv", list(profiles.excluded), ["case_id", "reason"])
    result["included_case_ids"] = list(profiles.case_ids)
    result["excluded_cases"] = list(profiles.excluded)
    result["total_cases"] = len(cases)
    result["technology_rule"] = "GeneratorName prefix before underscore"
    result["weighting"] = "MVA within each technology; MVA across all devices for spreads; equal case weight"
    result["extra_technologies"] = sorted({
        p.generator_name.split("_", 1)[0] for p in analysis.points
    } - set(TECHNOLOGIES))
    result["interpretation"] = (
        "Descriptive exploration, not causal inference or a stability predictor. "
        "Thresholds are heuristics, not significance tests. PCA uses these same ten indicators. "
        "Stability, IDs, locations, Kp and Ki are not model inputs. "
        "Unknown stability cases can be clustered but are never counted as stable. "
        "Cluster numbers are separate for each method. Only complete cases enter clustering; "
        "all included devices, including additional technologies, enter the two spreads."
    )
    paths = ()
    if result["status"] == "fitted":
        _csv(out / "model_selection.csv", result["candidates"], list(result["candidates"][0]))
        assignments = []
        for index, identifier in enumerate(profiles.case_ids):
            assessment = cases[identifier].assessment
            row = dict(case_id=identifier, status=assessment.status,
                       voltage_status=assessment.voltage_status,
                       modal_damping_status=assessment.modal_damping_status,
                       angular_status=assessment.angular_status)
            for method, info in result["methods"].items():
                row[method + "_cluster"] = info["labels"][index]
            assignments.append(row)
        _csv(out / "case_clusters.csv", assignments, list(assignments[0]))
        centroid_rows, stability_rows = [], []
        for method, info in result["methods"].items():
            labels = np.asarray(info["labels"])
            for c in range(1, info["k"] + 1):
                indices = np.flatnonzero(labels == c)
                selected = [cases[profiles.case_ids[i]].assessment for i in indices]
                centroid_rows.append(dict(method=method, cluster=c, n_cases=len(indices),
                    **dict(zip(FEATURES, profiles.values[indices].mean(axis=0)))))
                counts = Counter(a.status for a in selected)
                known = counts["Stable"] + counts["Unstable"]
                stability_rows.append(dict(
                    method=method, cluster=c, n_cases=len(indices),
                    stable=counts["Stable"], unstable=counts["Unstable"],
                    not_evaluated=counts["Not evaluated"],
                    unstable_fraction_evaluated=counts["Unstable"] / known if known else None,
                    voltage_failures=sum(a.voltage_status == "Unstable" for a in selected),
                    damping_failures=sum(a.modal_damping_status == "Growing" or
                        (a.angular_status is not None and a.modal_damping_status == "Damped under")
                        for a in selected),
                    angular_failures=sum(a.angular_status == "Unstable" for a in selected)))
        _csv(out / "cluster_profiles.csv", centroid_rows, list(centroid_rows[0]))
        _csv(out / "cluster_stability.csv", stability_rows, list(stability_rows[0]))
        pca = result["pca"]
        _csv(out / "pca_variance.csv",
             [dict(component=i + 1, explained_variance=v, cumulative_variance=c,
                   retained=i < pca["retained_components"])
              for i, (v, c) in enumerate(zip(pca["explained_variance_ratio"],
                                              np.cumsum(pca["explained_variance_ratio"])))],
             ["component", "explained_variance", "cumulative_variance", "retained"])
        _csv(out / "pca_loadings.csv",
             [dict(component=i + 1, **dict(zip(FEATURES, row)))
              for i, row in enumerate(pca["components"])], ["component", *FEATURES])
        paths = _export_figures(out, profiles, result, cases)
        (out / "PCA_DESCRIPTION.md").write_text(pca_description(result), encoding="utf-8")
    from .technology_analysis import export_technology_analysis
    paths += export_technology_analysis(analysis, out, result,
                                       low_xi_threshold=low_xi_threshold,
                                       min_cases=min_cell_cases)
    result["figures"] = [p.name for p in paths]
    (out / "clustering_metadata.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    explanation = "\n".join(
        f"- {method}: k={info['k']}, silhouette={info['silhouette_original']:.3f}, "
        f"ARI mediana={info['resampling_ari_median']}, resultado={info['conclusion']}."
        for method, info in result["methods"].items())
    (out / "README.md").write_text(
        "# Clustering exploratorio de corridas\n\n"
        + (explanation or result["reason"]) + "\n\n"
        "Diez indicadores: medias MVA de xi y fn para BESS, PE, PFV y PMGD, "
        "y desviaciones estandar MVA de xi y fn de todos los dispositivos incluidos. "
        "Cada corrida pesa igual. Tecnologias ausentes excluyen el caso; nunca se rellenan con cero. "
        "No evaluado se conserva como resultado independiente.\n\n"
        "StandardScaler antes de k-means y PCA. Se prueban k=2..max_clusters, "
        "limitados por la muestra; se descartan grupos de un solo caso. El mejor k "
        "maximiza silhouette en los diez indicadores estandarizados para ambos metodos. "
        "PCA conserva por defecto 90 % de la variabilidad de estos diez indicadores. "
        "La segunda puntuacion silhouette usa el espacio de ajuste.\n\n"
        "Se reestima el preprocesado en 20 submuestras del 80 % (ajustado en muestras pequenas) "
        "y se compara la pertenencia predicha mediante ARI. Los limites configurables "
        "silhouette >= 0.25 y ARI mediana >= 0.60 son filtros descriptivos, no pruebas estadisticas. "
        "no_well_defined_groups significa que no se obtuvo respaldo suficiente con esos filtros; "
        "se exporta igualmente la particion exploratoria. No prueba ausencia de toda estructura.\n\n"
        "01 compara k y variabilidad PCA. Se elimina el grafico 02. "
        "03: azul estable, naranja inestable; se omite no evaluado de la leyenda. "
        "Si existen no evaluados se conservan como barras grises y se cuenta su cantidad en el titulo. "
        "04: colores por grupo en los primeros paneles y por estabilidad en el ultimo. "
        "05: paneles kmeans y pca_kmeans con color por grupo y forma por estabilidad; "
        "circulo estable, X inestable y rombo no evaluado solo si existe. "
        "Cada punto es una corrida, sin identificadores. "
        "PCA_DESCRIPTION.md explica que combina cada eje, con coeficientes PC1/PC2. "
        "El plano de dos componentes no contiene necesariamente toda la informacion usada.\n\n"
        "06: nuevos ejes PCA calculados con ponderacion por unidades y MVA "
        "antes del ajuste; los colores conservan los grupos originales. "
        "WEIGHTED_PCA_DESCRIPTION.md explica formulas y coordenadas. "
        "07 y 08: porcentaje real de corridas inestables segun fraccion con xi bajo "
        "y fn media, en el conjunto y por tecnologia. TECHNOLOGY_ANALYSIS.md detalla "
        "las formulas, muestras y limitaciones.\n\n"
        "case_clusters.csv identifica cada corrida y conserva los tres criterios. "
        "cluster_profiles.csv contiene medias de los casos asignados en unidades originales. "
        "cluster_stability.csv compara resultados; sus fallas pueden solaparse. "
        "case_features.csv, model_selection.csv y pca_*.csv permiten revisar entradas y ajustes. "
        "clustering_metadata.json contiene escalado, semillas, pertenencias y diagnosticos. "
        "Si status=not_fitted, no se ajusto un modelo. El resumen de estabilidad nunca entra al ajuste; no hay inferencia causal.\n",
        encoding="utf-8")
    return paths


def main():
    import matplotlib
    matplotlib.use("Agg")
    from utils.output_directory import select_output_directory
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path,
                        help="Existing 04_spatial_stability directory; omitted opens picker")
    parser.add_argument("--output", type=Path,
                        help="Directory to contain clustering/; omitted opens output picker")
    parser.add_argument("--max-clusters", type=int, default=6)
    parser.add_argument("--pca-variance", type=float, default=.90)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--low-xi-threshold", type=float, default=.7)
    parser.add_argument("--min-cell-cases", type=int, default=5)
    parser.add_argument("--silhouette-threshold", type=float, default=.25)
    parser.add_argument("--ari-threshold", type=float, default=.60)
    args = parser.parse_args()
    if args.results is None:
        from tkinter import Tk, filedialog
        window = Tk()
        window.withdraw()
        window.attributes("-topmost", True)
        try:
            selected = filedialog.askdirectory(title="Select existing 04_spatial_stability results")
        finally:
            window.destroy()
        if not selected:
            raise RuntimeError("Results selection was cancelled.")
        args.results = Path(selected)
    analysis = load_saved_analysis(args.results)
    output = select_output_directory(args.output)
    paths = export_clustering(
        analysis, output, max_clusters=args.max_clusters, seed=args.seed,
        pca_variance=args.pca_variance, silhouette_threshold=args.silhouette_threshold,
        ari_threshold=args.ari_threshold, low_xi_threshold=args.low_xi_threshold,
        min_cell_cases=args.min_cell_cases)
    print(f"Clustering completed: {len(paths)} PNGs in {output / 'clustering'}; "
          "see README.md and clustering_metadata.json for group quality.")


if __name__ == "__main__":
    main()
