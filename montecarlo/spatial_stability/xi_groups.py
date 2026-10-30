"""Two new groups from mean xi and xi spread, shown on weighted PCA axes."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from utils.figure_localization import remove_figure_with_spanish_copy, save_figure_with_spanish_copy
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .distributions import _csv, _validate, LABELS, STATUSES, PLOT_STATUSES, load_saved_analysis


def xi_case_features(analysis, case_ids, basis):
    """Both inputs describe the entire included population of each run."""
    _validate(analysis)
    if basis not in {"units", "mva"}:
        raise ValueError("Weighting basis must be units or mva")
    grouped = defaultdict(list)
    for point in analysis.points:
        grouped[point.case_id].append(point)
    rows = []
    for identifier in case_ids:
        points = grouped[identifier]
        if not points:
            raise ValueError("No controllers for selected case")
        xi = np.array([p.xi for p in points])
        weights = np.ones(len(points)) if basis == "units" else np.array([
            p.nominal_power_mva for p in points])
        mean = float(np.average(xi, weights=weights))
        spread = float(np.sqrt(np.average((xi-mean)**2, weights=weights)))
        rows.append([mean, spread])
    return np.asarray(rows, dtype=float).reshape(-1, 2)


def fit_xi_groups(values, *, seed=42, resamples=20):
    """Fit exactly two clusters; order their IDs by mean xi, never by stability."""
    x = np.asarray(values, dtype=float)
    if x.ndim != 2 or x.shape[1] != 2 or not np.isfinite(x).all():
        raise ValueError("Expected finite mean-xi/spread pairs")
    if resamples < 2:
        raise ValueError("At least two resampling trials are required")
    result = dict(status="not_fitted", reason=None, k=2, seed=seed,
                  input_features=["xi_mean", "xi_spread"], labels=[], groups=[])
    if len(x) < 4 or len(np.unique(x, axis=0)) < 2:
        result["reason"] = "Need four cases and at least two distinct xi profiles."
        return result
    scaler = StandardScaler().fit(x)
    z = scaler.transform(x)
    with threadpool_limits(limits=1):
        model = KMeans(n_clusters=2, n_init=30, random_state=seed).fit(z)
        if min(np.bincount(model.labels_, minlength=2)) < 2:
            result["reason"] = "One proposed group contains fewer than two cases."
            return result
        centers = {int(c): x[model.labels_ == c].mean(axis=0) for c in (0, 1)}
        order = sorted(centers, key=lambda c: tuple(centers[c]))
        labels = np.where(model.labels_ == order[0], 1, 2)
        groups = []
        for index, c in enumerate(order):
            other = centers[order[1-index]]
            xi, spread = centers[c]
            xi_relation = "similar" if np.isclose(xi, other[0], atol=1e-12, rtol=0) else (
                "menor" if index == 0 else "mayor")
            spread_relation = "similar" if np.isclose(spread, other[1], atol=1e-12, rtol=0) else (
                "menor" if spread < other[1] else "mayor")
            groups.append(dict(group=index+1, n_cases=int(np.sum(labels == index+1)),
                               mean_xi=float(xi), mean_xi_spread=float(spread),
                               xi_relation=xi_relation, spread_relation=spread_relation))
        rng = np.random.default_rng(seed)
        agreements = []
        for trial in range(resamples):
            indices = rng.choice(len(x), min(len(x)-1, max(3, int(.8*len(x)))), replace=False)
            if len(np.unique(x[indices], axis=0)) < 2:
                continue
            subset_scaler = StandardScaler().fit(x[indices])
            fitted = KMeans(n_clusters=2, n_init=30, random_state=seed+trial+1).fit(
                subset_scaler.transform(x[indices]))
            predicted = fitted.predict(subset_scaler.transform(x))
            agreements.append(float(adjusted_rand_score(labels, predicted)))
    result.update(status="fitted", labels=labels.tolist(), groups=groups,
                  scaler_mean=scaler.mean_.tolist(), scaler_scale=scaler.scale_.tolist(),
                  silhouette=float(silhouette_score(z, labels)),
                  resampling_ari_median=float(np.median(agreements)) if agreements else None,
                  resampling_ari_values=agreements,
                  interpretation="Exactly two exploratory groups requested; not evidence of two natural populations.")
    return result


def export_xi_groups(analysis, out, case_ids, *, pca_models=None, seed=42, resamples=20):
    """Write only graph 09 and its own files; never touch existing graphs."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    generated = ("09_xi_groups_weighted_pca.png", "xi_group_cases.csv",
                 "xi_group_summary.csv", "xi_groups_metadata.json", "XI_GROUPS_DESCRIPTION.md")
    if (out/"xi_groups_metadata.json").is_file():
        for name in generated:
            path = out/name
            if path.is_file():
                if path.suffix == ".png":
                    remove_figure_with_spanish_copy(path)
                else:
                    path.unlink()
    ids = tuple(case_ids)
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate selected case IDs")
    groups, features, models = {}, {}, dict(pca_models or {})
    for basis in ("units", "mva"):
        features[basis] = xi_case_features(analysis, ids, basis)
        groups[basis] = fit_xi_groups(features[basis], seed=seed, resamples=resamples)
    if any(g["status"] == "fitted" for g in groups.values()) and not models:
        from .clustering import build_case_profiles
        from .technology_analysis import technology_composition
        from .weighted_pca import weighted_profiles, feature_population_weights, fit_weighted_pca
        composition = technology_composition(analysis, ids)
        for basis in ("units", "mva"):
            profiles = weighted_profiles(analysis, ids, basis)
            lookup = dict(zip(profiles.case_ids, profiles.values))
            values = np.array([lookup[c] for c in ids])
            models[basis] = fit_weighted_pca(values, feature_population_weights(composition, basis))
    cases = {c.case_id: c.assessment for c in analysis.cases}
    case_rows, summary_rows = [], []
    for basis in ("units", "mva"):
        info = groups[basis]
        model = models.get(basis, dict(status="not_fitted"))
        fitted = info["status"] == "fitted"
        for index, identifier in enumerate(ids):
            assessment = cases[identifier]
            scores = model["scores"][index] if model["status"] == "fitted" else []
            case_rows.append(dict(
                basis=basis, case_id=identifier, new_group=info["labels"][index] if fitted else None,
                xi_mean=float(features[basis][index, 0]), xi_spread=float(features[basis][index, 1]),
                PC1=scores[0] if scores else None, PC2=scores[1] if len(scores) > 1 else None,
                status=assessment.status, voltage_status=assessment.voltage_status,
                modal_damping_status=assessment.modal_damping_status, angular_status=assessment.angular_status))
        for group in info["groups"]:
            selected = [cases[c] for c, label in zip(ids, info["labels"]) if label == group["group"]]
            summary_rows.append(dict(basis=basis, **group,
                stable=sum(c.status == "Stable" for c in selected),
                unstable=sum(c.status == "Unstable" for c in selected),
                not_evaluated=sum(c.status == "Not evaluated" for c in selected)))
    _csv(out/"xi_group_cases.csv", case_rows, ["basis", "case_id", "new_group", "xi_mean",
         "xi_spread", "PC1", "PC2", "status", "voltage_status", "modal_damping_status", "angular_status"])
    _csv(out/"xi_group_summary.csv", summary_rows, ["basis", "group", "n_cases", "mean_xi",
         "mean_xi_spread", "xi_relation", "spread_relation", "stable", "unstable", "not_evaluated"])
    metadata = dict(
        case_ids=list(ids), clustering=groups,
        pca_scope="Same ten-indicator population-weighted PCA coordinates as graph 06",
        group_scope="Two new groups per weighting basis, fitted only on global xi mean and xi standard deviation",
        outcome_scope=analysis.criterion_description + "; labels are displayed after grouping",
        weighting="Units: equal controller weight; MVA: nominal-power fractions within each case; equal case weight in k-means",
        group_numbering="G1 has lower centroid mean xi; G2 higher. Spread relations are measured, never forced.",
        pca_models=models, figures=[])
    paths = ()
    available = any(g["status"] == "fitted" for g in groups.values())
    if available:
        fig = Figure(figsize=(14, 12), layout="constrained")
        FigureCanvasAgg(fig)
        axes = fig.subplots(2, 2)
        colors = {1: "#1f77b4", 2: "#ff7f0e"}
        markers = {"Stable": "o", "Unstable": "X", "Not evaluated": "D"}
        statuses = np.array([cases[c].status for c in ids])
        for col, basis in enumerate(("units", "mva")):
            info = groups[basis]
            model = models.get(basis, dict(status="not_fitted"))
            if info["status"] != "fitted" or model["status"] != "fitted":
                for axis in axes[:, col]:
                    axis.text(.5, .5, info["reason"] or "Sin ejes PCA", ha="center", va="center")
                    axis.set_axis_off()
                continue
            labels = np.asarray(info["labels"])
            scores = np.asarray(model["scores"])
            variance = model["explained_variance_ratio"]
            second = scores[:, 1] if scores.shape[1] > 1 else np.zeros(len(scores))
            second_variance = variance[1] if len(variance) > 1 else 0.
            for row, coordinates in enumerate(((scores[:, 0], second),
                                                (features[basis][:, 0], features[basis][:, 1]))):
                axis = axes[row, col]
                for group in (1, 2):
                    for status in PLOT_STATUSES:
                        mask = (labels == group) & (statuses == status)
                        if mask.any():
                            axis.scatter(coordinates[0][mask], coordinates[1][mask], s=52,
                                marker=markers[status], color=colors[group], alpha=.9,
                                edgecolor="white" if status == "Stable" else "#222222",
                                linewidth=.65, zorder=3 if status == "Stable" else 2)
                axis.grid(alpha=.2)
            name = "Unidades" if basis == "units" else "MVA"
            axes[0, col].set(
                title=f"{name}: G1 n={int(np.sum((labels == 1) & np.isin(statuses, PLOT_STATUSES)))}; "
                      f"G2 n={int(np.sum((labels == 2) & np.isin(statuses, PLOT_STATUSES)))}",
                xlabel=f"PC1 ponderada ({100*variance[0]:.1f} %)",
                ylabel=f"PC2 ponderada ({100*second_variance:.1f} %)")
            description = " | ".join(
                f"G{g['group']}: dispersion {g['spread_relation']}" for g in info["groups"])
            axes[1, col].set(title=f"Indicadores usados para separar los grupos\n{description}",
                             xlabel=f"xi medio del caso ({name})",
                             ylabel=f"Desviacion estandar de xi ({name})")
        handles = [Patch(facecolor=colors[1], label="G1: xi medio menor"),
                   Patch(facecolor=colors[2], label="G2: xi medio mayor")]
        handles += [Line2D([], [], marker=markers[s], linestyle="none", color="#777777",
                            markeredgecolor="#222222", label=LABELS[s])
                    for s in PLOT_STATUSES if np.any(statuses == s)]
        fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), fontsize=10)
        fig.suptitle("Dos grupos nuevos segun xi medio y dispersion de xi\n"
                     "Arriba: proyeccion PCA ponderada | Abajo: indicadores que forman los grupos\n"
                     "fn y estabilidad no intervienen en esta nueva separacion; G1/G2 son relativos a la muestra")
        path = out/"09_xi_groups_weighted_pca.png"
        try:
            save_figure_with_spanish_copy(fig, path, dpi=170)
        finally:
            fig.clear()
        paths = (path,)
    metadata["figures"] = [p.name for p in paths]
    (out/"xi_groups_metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    lines = [
        "# Dos grupos nuevos por xi medio y dispersion", "",
        "Se conserva el grafico 05. El grafico 09 forma grupos independientes de los "
        "anteriores a partir de dos indicadores del conjunto de dispositivos de cada caso: "
        "xi medio y desviacion estandar de xi. Cada corrida es una observacion.", "",
        "Se estandarizan estos dos indicadores para darles igual escala y se ajusta "
        "k-means con k=2, n_init=30 y semilla registrada. No se usa fn, estabilidad, "
        "posicion ni coordenadas PCA para formar estos grupos. El promedio y dispersion "
        "usan igual peso por unidad o ponderacion MVA, segun el panel. Las dos versiones "
        "pueden producir pertenencias distintas. No se fijan umbrales operativos de xi.", "",
        "G1 tiene menor xi medio de grupo; G2, mayor. La mayor o menor dispersion "
        "se describe segun los promedios reales de cada grupo: no se fuerza un patron. "
        "Si los centroides tienen igual xi medio, los IDs no implican una diferencia real de xi. "
        "Son dos particiones exploratorias solicitadas; no se demuestra la existencia de dos "
        "familias naturales. Se registran silhouette y ARI de submuestras del 80 %.", "",
        "Los paneles superiores reutilizan exactamente los ejes ponderados del grafico 06 "
        "(diez indicadores). Los paneles inferiores muestran los dos indicadores fisicos usados "
        "para separar los grupos. fn puede afectar la proyeccion superior, pero no sus colores. "
        "Circulo: estable, X: inestable. No evaluados no se dibujan. La estabilidad proviene "
        "del criterio nativo registrado en xi_groups_metadata.json y no entra al ajuste.", "",
        "| Ponderacion | Grupo | Casos | xi medio | Dispersion media | Estables | Inestables |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in summary_rows:
        lines.append(f"| {r['basis']} | G{r['group']} | {r['n_cases']} | {r['mean_xi']:.5f} | "
                     f"{r['mean_xi_spread']:.5f} | {r['stable']} | {r['unstable']} |")
    lines += ["", "xi_group_cases.csv conserva pertenencias nuevas y coordenadas por corrida; "
              "xi_group_summary.csv resume los perfiles. xi_groups_metadata.json incluye "
              "escalado, semillas, calidad y los ejes PCA completos.", ""]
    (out/"XI_GROUPS_DESCRIPTION.md").write_text("\n".join(lines), encoding="utf-8")
    return paths


def main():
    from .clustering import build_case_profiles
    from utils.output_directory import select_output_directory
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, help="Existing 05_Spatial_Stability; omitted opens picker")
    parser.add_argument("--output", type=Path, help="Directory to contain clustering/; omitted opens picker")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.results is None:
        from tkinter import Tk, filedialog
        window = Tk()
        window.withdraw()
        window.attributes("-topmost", True)
        try:
            selected = filedialog.askdirectory(title="Select existing 05_Spatial_Stability results")
        finally:
            window.destroy()
        if not selected:
            raise RuntimeError("Results selection was cancelled.")
        args.results = Path(selected)
    analysis = load_saved_analysis(args.results)
    ids = build_case_profiles(analysis).case_ids
    output = select_output_directory(args.output)/"clustering"
    paths = export_xi_groups(analysis, output, ids, seed=args.seed)
    print(f"Created {len(paths)} new xi-group PNGs in {output}; existing graphs preserved.")


if __name__ == "__main__":
    main()
