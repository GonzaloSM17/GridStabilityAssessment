"""Population-weighted PCA coordinates and physical instability grids."""
from collections import defaultdict
import json
from pathlib import Path
import numpy as np

from .distributions import _csv, _validate



def technology_composition(analysis, case_ids):
    """Average within-case fractions: a large fleet does not give a run extra weight."""
    ids = set(case_ids)
    grouped = defaultdict(list)
    for point in analysis.points:
        if point.case_id in ids:
            grouped[point.case_id].append(point)
    if set(grouped) != ids or not ids:
        raise ValueError("Composition requires nonempty, matched case IDs")
    technologies = sorted({p.generator_name.split("_", 1)[0]
                           for points in grouped.values() for p in points})
    rows = []
    for technology in technologies:
        counts, mva, unit_shares, mva_shares, devices = [], [], [], [], set()
        for points in grouped.values():
            selected = [p for p in points if p.generator_name.split("_", 1)[0] == technology]
            counts.append(len(selected))
            mva.append(sum(p.nominal_power_mva for p in selected))
            unit_shares.append(len(selected) / len(points))
            mva_shares.append(mva[-1] / sum(p.nominal_power_mva for p in points))
            devices.update(p.device_id for p in selected)
        rows.append(dict(
            technology=technology, n_cases=len(ids), unique_device_ids=len(devices),
            unit_case_observations=sum(counts), mva_case_sum=sum(mva),
            mean_units_per_case=float(np.mean(counts)), mean_mva_per_case=float(np.mean(mva)),
            unit_share=float(np.mean(unit_shares)), mva_share=float(np.mean(mva_shares))))
    return rows


def physical_case_features(analysis, low_xi_threshold=.7):
    """One observation per case/scope/basis; labels do not enter the indicators."""
    _validate(analysis)
    if not np.isfinite(low_xi_threshold) or low_xi_threshold <= 0:
        raise ValueError("Low-xi threshold must be positive and finite")
    cases = {c.case_id: c.assessment.status for c in analysis.cases}
    grouped = defaultdict(list)
    for point in analysis.points:
        grouped[point.case_id].append(point)
    rows = []
    for case_id, points in sorted(grouped.items()):
        technologies = sorted({p.generator_name.split("_", 1)[0] for p in points})
        for scope in ("All", *technologies):
            selected = points if scope == "All" else [
                p for p in points if p.generator_name.split("_", 1)[0] == scope]
            low = np.array([p.xi < low_xi_threshold for p in selected])
            fn = np.array([p.fn_hz for p in selected])
            for basis in ("units", "mva"):
                w = np.ones(len(selected)) if basis == "units" else np.array([
                    p.nominal_power_mva for p in selected])
                rows.append(dict(
                    case_id=case_id, status=cases[case_id], scope=scope, basis=basis,
                    n_devices=len(selected), low_xi_count=int(low.sum()),
                    included_mva=sum(p.nominal_power_mva for p in selected),
                    low_xi_fraction=float(w[low].sum() / w.sum()),
                    mean_fn_hz=float(np.average(fn, weights=w))))
    return rows


def quantile_edges(values, bins=4):
    """Deduplicate ties, keeping a constant indicator in one explicit bin."""
    return np.unique(np.quantile(np.asarray(values, dtype=float),
                                 np.linspace(0., 1., bins + 1)))


def _bin(values, edges):
    return np.searchsorted(edges[1:-1], values, side="right")


def instability_grid(rows, x_edges, y_edges, *, min_cases=5):
    """Unknown outcomes never enter the denominator; empty cells are not zero risk."""
    nx, ny = max(1, len(x_edges) - 1), max(1, len(y_edges) - 1)
    x = _bin([r["low_xi_fraction"] for r in rows], x_edges)
    y = _bin([r["mean_fn_hz"] for r in rows], y_edges)
    cells = []
    for iy in range(ny):
        for ix in range(nx):
            selected = [r for r, a, b in zip(rows, x, y) if a == ix and b == iy]
            unstable = sum(r["status"] == "Unstable" for r in selected)
            stable = sum(r["status"] == "Stable" for r in selected)
            unknown = len(selected) - stable - unstable
            n = stable + unstable
            fraction, lower, upper = None, None, None
            if n:
                fraction = unstable / n
                z = 1.959963984540054
                denominator = 1 + z*z / n
                center = (fraction + z*z / (2*n)) / denominator
                half = z * np.sqrt(fraction*(1-fraction)/n + z*z/(4*n*n)) / denominator
                lower, upper = float(max(0., center-half)), float(min(1., center+half))
            cells.append(dict(x_bin=ix, y_bin=iy, stable=stable, unstable=unstable,
                              not_evaluated=unknown, evaluated=n,
                              instability_fraction=fraction, wilson95_lower=lower,
                              wilson95_upper=upper, low_sample=bool(0 < n < min_cases)))
    return cells


def _range_labels(edges, factor=1):
    if len(edges) == 1:
        return [f"= {edges[0]*factor:.2f}"]
    return [f"[{a*factor:.2f}, {b*factor:.2f}" + ("]" if i == len(edges)-2 else ")")
            for i, (a, b) in enumerate(zip(edges[:-1], edges[1:]))]


def _draw_grid(axis, cells, xe, ye, *, scope, basis, low_xi_threshold, min_cases):
    from matplotlib.patches import Rectangle
    nx, ny = max(1, len(xe)-1), max(1, len(ye)-1)
    values = np.full((ny, nx), np.nan)
    for cell in cells:
        if cell["instability_fraction"] is not None:
            values[cell["y_bin"], cell["x_bin"]] = 100 * cell["instability_fraction"]
    image = axis.imshow(np.ma.masked_invalid(values), origin="lower",
                        cmap="Reds", vmin=0, vmax=100, aspect="auto")
    axis.set_facecolor("#eeeeee")
    for cell in cells:
        ix, iy, n = cell["x_bin"], cell["y_bin"], cell["evaluated"]
        text = "Sin casos"

        if n:
            text = f"{100*cell['instability_fraction']:.0f}%\n{cell['unstable']}/{n}"
        if cell["low_sample"]:
            axis.add_patch(Rectangle((ix-.5, iy-.5), 1, 1, fill=False,
                                     hatch="//", edgecolor="#555555", linewidth=0))
        axis.text(ix, iy, text, ha="center", va="center", fontsize=8,
                  color="white" if n and cell["instability_fraction"] >= .65 else "#222222")
    axis.set_xticks(range(nx), _range_labels(xe, 100), rotation=25, ha="right", fontsize=8)
    axis.set_yticks(range(ny), _range_labels(ye), fontsize=8)
    axis.set_xticks(np.arange(-.5, nx, 1), minor=True)
    axis.set_yticks(np.arange(-.5, ny, 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=2)
    axis.tick_params(which="minor", bottom=False, left=False)
    known = sum(c["evaluated"] for c in cells)
    failed = sum(c["unstable"] for c in cells)
    rate = f"{100*failed/known:.1f}%" if known else "sin evaluados"
    label = "unidades" if basis == "units" else "potencia MVA"
    weighting = "Cada unidad pesa igual" if basis == "units" else "Cada unidad pesa sus MVA"
    mean_label = "fn promedio simple [Hz]" if basis == "units" else "fn ponderada por MVA [Hz]"
    axis.set(title=f"{scope} | {weighting}\nn={known}, inestables={rate}",
             xlabel=f"Fraccion de {label} con xi < {low_xi_threshold:g} (%)",
             ylabel=mean_label)
    return image


def export_technology_analysis(analysis, out, result, *, low_xi_threshold=.7, min_cases=5):
    """Save three additional PNGs, input summaries, grid counts and methods."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    if not isinstance(min_cases, int) or min_cases < 1:
        raise ValueError("min_cases must be a positive integer")
    if not np.isfinite(low_xi_threshold) or low_xi_threshold <= 0:
        raise ValueError("Low-xi threshold must be positive and finite")
    out = Path(out)
    generated = (
        "06_pca_technology_shares.png", "07_physical_instability.png",
        "08_technology_instability.png", "technology_composition.csv",
        "pca_technology_shares.csv", "physical_case_features.csv",
        "physical_instability_cells.csv", "technology_analysis_metadata.json",
        "TECHNOLOGY_ANALYSIS.md", "06_weighted_pca_projection.png",
        "weighted_pca_case_features.csv", "weighted_pca_weights.csv",
        "weighted_pca_coefficients.csv", "weighted_pca_coordinates.csv",
        "weighted_pca_variance.csv", "weighted_pca_metadata.json",
        "WEIGHTED_PCA_DESCRIPTION.md", "09_xi_groups_weighted_pca.png",
        "xi_group_cases.csv", "xi_group_summary.csv", "xi_groups_metadata.json",
        "XI_GROUPS_DESCRIPTION.md")
    if (out / "technology_analysis_metadata.json").is_file():
        for name in generated:
            path = out / name
            if path.is_file():
                path.unlink()
    mapped_ids={point.case_id for point in analysis.points}
    evaluated = sum(c.case_id in mapped_ids and c.assessment.status in {"Stable", "Unstable"} for c in analysis.cases)
    metadata = dict(low_xi_threshold=low_xi_threshold, min_cases=min_cases,
                    total_cases=len(analysis.cases), evaluated_cases=evaluated,
                    pca_cases=len(result["included_case_ids"]), figures=[])
    if evaluated < 4:
        metadata["reason"] = "At least four evaluated cases are needed for physical grids."
        (out / "technology_analysis_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8")
        return ()
    paths = []
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

    if result["methods"] and result["included_case_ids"]:
        from .weighted_pca import export_weighted_pca
        composition = technology_composition(analysis, result["included_case_ids"])
        _csv(out / "technology_composition.csv", composition, list(composition[0]))
        paths.extend(export_weighted_pca(analysis, out, result, composition))
        metadata["weighted_pca"] = "Recomputed axes after weighting standardized indicators by sqrt(unit/MVA technology share); original cluster labels retained"
        metadata["composition_scope"] = "Complete PCA cases; each case has equal weight; repeated device IDs are not distinct units"

    rows = physical_case_features(analysis, low_xi_threshold)
    _csv(out / "physical_case_features.csv", rows, list(rows[0]))
    scopes = ["All"] + sorted({r["scope"] for r in rows} - {"All"})
    grids, cell_rows = {}, []
    for scope in scopes:
        combined = [r for r in rows if r["scope"] == scope]
        xe = quantile_edges([r["low_xi_fraction"] for r in combined])
        ye = quantile_edges([r["mean_fn_hz"] for r in combined])
        for basis in ("units", "mva"):
            subset = [r for r in combined if r["basis"] == basis]
            cells = instability_grid(subset, xe, ye, min_cases=min_cases)
            grids[scope, basis] = (cells, xe, ye)
            for cell in cells:
                ix, iy = cell["x_bin"], cell["y_bin"]
                cell_rows.append(dict(scope=scope, basis=basis, **cell,
                    x_lower=float(xe[ix]), x_upper=float(xe[min(ix+1,len(xe)-1)]),
                    y_lower=float(ye[iy]), y_upper=float(ye[min(iy+1,len(ye)-1)])))
    _csv(out / "physical_instability_cells.csv", cell_rows, list(cell_rows[0]))
    fig = figure(14, 8)
    axes = fig.subplots(1, 2)
    for axis, basis in zip(axes, ("units", "mva")):
        cells, xe, ye = grids["All", basis]
        image = _draw_grid(axis, cells, xe, ye, scope="Todas las tecnologias", basis=basis,
                          low_xi_threshold=low_xi_threshold, min_cases=min_cases)
    fig.colorbar(image, ax=axes.tolist(), label="Corridas inestables / evaluadas (%)")
    fig.suptitle(f"Inestabilidad segun indicadores fisicos de cada corrida\n"
                 f"Celda: porcentaje y casos inestables/evaluados | Rayado: menos de {min_cases} casos\n"
                 "Mismos intervalos, distinta ponderacion: las corridas pueden cambiar de celda\n"
                 "Gris: sin evaluados; cada porcentaje usa casos estables + inestables")
    save(fig, "07_physical_instability.png")
    technologies = scopes[1:]
    fig = figure(max(12, 5*len(technologies)), 12)
    axes = fig.subplots(2, len(technologies), squeeze=False)
    for col, scope in enumerate(technologies):
        for row, basis in enumerate(("units", "mva")):
            cells, xe, ye = grids[scope, basis]
            image = _draw_grid(axes[row, col], cells, xe, ye, scope=scope, basis=basis,
                              low_xi_threshold=low_xi_threshold, min_cases=min_cases)
    fig.colorbar(image, ax=axes.ravel().tolist(), label="Corridas inestables / evaluadas (%)")
    fig.suptitle("Configuracion por tecnologia frente a la estabilidad del caso completo\n"
                 f"Celda: porcentaje e inestables/evaluados | Rayado: menos de {min_cases} casos\n"
                 "Cuartiles propios de cada tecnologia, comunes entre unidades y MVA; no son umbrales de estabilidad")
    save(fig, "08_technology_instability.png")
    metadata.update(
        physical_scope="All mapped cases, including cases incomplete for PCA; absent technologies are not zero-filled",
        cell_denominator="Stable + Unstable cases; Not evaluated counted separately",
        bin_rule="Quartiles of both weighting bases pooled within scope; ties collapsed; lower inclusive, upper exclusive, final upper inclusive",
        unit_definition="A mapped device_id/controller; multiple active controllers sharing an asset share its MVA in native analysis",
        interpretation="Descriptive associations; no causal or predictive claims; xi threshold exploratory",
        figures=[p.name for p in paths])
    (out / "technology_analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    (out / "TECHNOLOGY_ANALYSIS.md").write_text(
        "# Tecnologias, PCA e indicadores fisicos\n\n"
        "06: PCA recalculado con los diez indicadores, ponderados por numero de unidades "
        "o por potencia MVA antes de ajustar los ejes. Sustituye el reparto posterior "
        "de coeficientes; los grupos originales se conservan como referencia. "
        "WEIGHTED_PCA_DESCRIPTION.md explica formulas y coordenadas.\n\n"
        "technology_composition.csv incluye unidades distintas (device_id), observaciones "
        "unidad-caso, suma MVA-caso y medias por corrida. Las unidades repetidas entre corridas "
        "no son nuevas unidades ni casos independientes. Las proporciones usan igual peso "
        "por corrida y solo los casos completos usados por PCA.\n\n"
        "09: dos grupos nuevos segun xi medio y dispersion de xi; "
        "XI_GROUPS_DESCRIPTION.md detalla sus entradas y resultados. "
        "07: estabilidad frente a dos indicadores fisicos del conjunto. "
        "Los paneles usan las mismas corridas y los mismos intervalos, pero los indicadores "
        "se calculan con igual peso por unidad o con peso MVA; una corrida puede cambiar de celda. "
        "No muestran solo inestables: el porcentaje usa inestables/(estables+inestables). "
        "Ejemplo: una unidad de 100 MVA con xi bajo y otra de 900 MVA sin xi bajo "
        "dan 50 % de unidades y 10 % de MVA con xi bajo. "
        "08: los mismos indicadores dentro de cada tecnologia frente al resultado del caso "
        "completo, no una clasificacion de estabilidad de esa tecnologia. "
        "Horizontal: fraccion de unidades o MVA con xi < "
        f"{low_xi_threshold:g}. Vertical: fn media con la misma ponderacion. "
        "Son ajustes de controladores, no frecuencias o amortiguamientos modales medidos. "
        "Se incluyen todos los casos espaciales aunque falten tecnologias para PCA.\n\n"
        "Limites por cuartiles de los indicadores de ambas ponderaciones, comunes dentro "
        "de cada poblacion; se colapsan limites repetidos. No se eligen segun la estabilidad. "
        "Cada caso cuenta una vez en cada panel. Una unidad ausente no cuenta como cero. "
        "Cada celda muestra porcentaje e inestables/evaluados. No evaluados se cuentan "
        "como NE y no entran al denominador. Gris no significa 0 %; indica sin evaluados. "
        f"Rayado: menos de {min_cases} casos evaluados, aunque el porcentaje se muestra. "
        "Una celda de 100 % con pocos casos no demuestra una regla de inestabilidad. "
        "physical_instability_cells.csv conserva conteos, limites e intervalos Wilson del 95 %. "
        "Son intervalos descriptivos binomiales; no corrigen comparaciones multiples ni "
        "garantizan independencia entre corridas o generalizacion a otras contingencias.\n",
        encoding="utf-8")
    return tuple(paths)
