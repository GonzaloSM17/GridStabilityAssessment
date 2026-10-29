"""PNG-only xi/fn maps, with a common drawing/size/color scale across runs."""
from collections import defaultdict
from dataclasses import asdict, replace
from math import ceil, floor, isfinite, log10
from pathlib import Path
from statistics import mean
import csv
import json
import re
import unicodedata
from time import monotonic

from .models import SpatialStabilityResult
from .distributions import PLOT_STATUSES

FN_LABELS = ("0-2 Hz", "2-5 Hz", "5-10 Hz", "10-20 Hz", ">=20 Hz")
STATUS_DIRECTORIES = {"Stable": "stable", "Unstable": "unstable", "Not evaluated": "not_evaluated"}


def _write_csv(path, rows, columns):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader(); writer.writerows(rows)


def _filename(case_id):
    text = unicodedata.normalize("NFKD", case_id).encode("ascii", "ignore").decode()
    text = re.sub(r"[^A-Za-z0-9_-]", "_", text).strip("_")
    if not text: raise ValueError(f"Run name cannot be used as a filename: {case_id!r}")
    return text


def frequency_band(fn_hz):
    return FN_LABELS[sum(fn_hz >= edge for edge in (2., 5., 10., 20.))]


def weighted_value(points, metric):
    total = sum(point.nominal_power_mva for point in points)
    if total <= 0: return None
    return sum(getattr(point, metric)*point.nominal_power_mva for point in points)/total


def mean_points(points, case_ids):
    """Equal weight for each available active-unit observation, never absent=0."""
    grouped = defaultdict(list)
    for point in points:
        if point.case_id in case_ids: grouped[point.device_id].append(point)
    results, rows = [], []
    for device, selected in sorted(grouped.items()):
        first = selected[0]
        if any((p.asset_name, p.x, p.y) != (first.asset_name, first.x, first.y) for p in selected):
            raise ValueError(f"Device location changes between runs: {device}")
        point = replace(first, case_id="class_mean", kp=mean(p.kp for p in selected), ki=mean(p.ki for p in selected),
                        xi=mean(p.xi for p in selected), fn_hz=mean(p.fn_hz for p in selected),
                        nominal_power_mva=mean(p.nominal_power_mva for p in selected))
        results.append(point)
        rows.append(dict(device_id=device, generator_name=point.generator_name, asset_name=point.asset_name,
                         x=point.x, y=point.y, mva=point.nominal_power_mva, n_cases=len(selected),
                         n_cases_in_class=len(case_ids), mean_xi=point.xi, mean_fn_hz=point.fn_hz,
                         frequency_band=frequency_band(point.fn_hz)))
    return tuple(results), rows


def _nice_numeric_scale(values, *, target_intervals=5):
    """Return enclosing decimal limits and ticks for a continuous colour scale."""
    minimum, maximum = min(values), max(values)
    span = maximum - minimum or abs(minimum) * 0.2 or 0.1
    target_step = span / target_intervals
    magnitude = 10.0 ** floor(log10(target_step))
    step = min(
        (factor * magnitude for factor in (1.0, 2.0, 5.0, 10.0)),
        key=lambda candidate: abs(candidate - target_step),
    )
    lower = floor(minimum / step) * step
    upper = ceil(maximum / step) * step
    if lower == upper:
        lower -= step
        upper += step
    count = int(round((upper - lower) / step))
    ticks = tuple(round(lower + index * step, 12) for index in range(count + 1))
    return round(lower, 12), round(upper, 12), ticks


def _frequency_labels(fn_edges):
    """Label every displayed frequency band with finite, rounded endpoints."""
    return tuple(
        f"{lower:g}-{upper:g} Hz"
        for lower, upper in zip(fn_edges, fn_edges[1:])
    )


def _mva_legend_label(value):
    """Present marker-size references as whole MVA values."""
    return f"{value:.0f} MVA"


def _draw_mean_comparison(
    stable_points,
    unstable_points,
    universe,
    metric,
    bounds,
    xi_limits,
    xi_ticks,
    fn_edges,
    size_factor,
    stable_count,
    unstable_count,
):
    """Draw Stable and Unstable class means side by side on shared map scales."""
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import BoundaryNorm, Normalize

    figure, axes = plt.subplots(1, 2, figsize=(17, 8.5), layout="constrained")
    if metric == "xi":
        cmap, norm = plt.get_cmap("viridis"), Normalize(*xi_limits)
        color_label = "PLL damping ratio xi"
    else:
        cmap = plt.get_cmap("viridis", len(fn_edges) - 1)
        norm = BoundaryNorm(fn_edges, cmap.N, clip=True)
        color_label = "PLL natural-frequency band"

    sizes = sorted({
        min(point.nominal_power_mva for point in universe),
        float(np.median([point.nominal_power_mva for point in universe])),
        max(point.nominal_power_mva for point in universe),
    })
    handles = [
        axes[0].scatter(
            [], [], s=value * size_factor, facecolors="none", edgecolors="grey",
            label=_mva_legend_label(value),
        )
        for value in sizes
    ]
    for axis, status, points, count in (
        (axes[0], "Stable", stable_points, stable_count),
        (axes[1], "Unstable", unstable_points, unstable_count),
    ):
        visible = points if points else universe
        axis.scatter(
            [point.x for point in visible],
            [point.y for point in visible],
            s=[point.nominal_power_mva * size_factor for point in visible],
            facecolors="none", edgecolors="#8A8A8A", linewidths=0.6,
        )
        axis.scatter(
            [point.x for point in points],
            [point.y for point in points],
            s=[point.nominal_power_mva * size_factor for point in points],
            c=[getattr(point, metric) for point in points], cmap=cmap, norm=norm,
            edgecolors="#1A1A1A", linewidths=0.4,
        )
        if not points:
            axis.text(
                0.5, 0.5, "NO CASES IN THIS CLASS", ha="center", va="center",
                transform=axis.transAxes,
                bbox=dict(facecolor="white", alpha=0.9, edgecolor="grey"),
            )
        axis.set(
            xlim=bounds[:2], ylim=bounds[2:], aspect="equal",
            xlabel="posX (EMTP drawing units)", ylabel="posY (EMTP drawing units)",
            title=f"{status} (n={count})",
        )
        axis.grid(alpha=0.15)

    color_source = ScalarMappable(norm=norm, cmap=cmap)
    color_source.set_array([])
    if metric == "xi":
        colorbar = figure.colorbar(color_source, ax=axes.tolist(), shrink=0.78)
        colorbar.set_label(color_label)
        colorbar.set_ticks(xi_ticks)
    else:
        centers = (np.array(fn_edges[:-1]) + np.array(fn_edges[1:])) / 2
        colorbar = figure.colorbar(
            color_source, ax=axes.tolist(), shrink=0.78, boundaries=fn_edges,
            ticks=centers, spacing="uniform",
        )
        colorbar.ax.set_yticklabels(_frequency_labels(fn_edges))
        colorbar.set_label(color_label)
    figure.legend(
        handles=handles, title="Area proportional to MVA", loc="outside lower center",
        ncols=len(handles), fontsize=8,
    )
    title = (
        "Mean PLL damping ratio by dynamic-stability class"
        if metric == "xi"
        else "Mean PLL natural-frequency band by dynamic-stability class"
    )
    figure.suptitle(
        f"{title}\nPer-unit mean across available active cases; common position, size and colour scales",
        fontsize=12,
    )
    return figure


def _draw(points, universe, metric, title, subtitle, bounds, xi_limits, xi_ticks, fn_edges, size_factor, scale_context="Common scales across runs"):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, Normalize
    figure, axis = plt.subplots(figsize=(11, 13), layout="constrained")
    if metric == "xi":
        cmap, norm = plt.get_cmap("viridis"), Normalize(*xi_limits)
        label, color_text = "PLL damping ratio xi", "xi"
    else:
        cmap = plt.get_cmap("viridis", len(FN_LABELS))
        norm = BoundaryNorm(fn_edges, cmap.N, clip=True)
        label, color_text = "PLL natural-frequency band [Hz]", "fn band [Hz]"
    visible = points if points else universe
    axis.scatter([p.x for p in visible], [p.y for p in visible],
                 s=[p.nominal_power_mva*size_factor for p in visible],
                 facecolors="none", edgecolors="grey", linewidths=.6)
    scatter = axis.scatter([p.x for p in points], [p.y for p in points],
                           s=[p.nominal_power_mva*size_factor for p in points],
                           c=[getattr(p, metric) for p in points], cmap=cmap, norm=norm,
                           edgecolors="black", linewidths=.4)
    if metric == "xi":
        bar = figure.colorbar(scatter, ax=axis, shrink=.65, label=label)
        bar.set_ticks(xi_ticks)
    else:
        ticks = (np.array(fn_edges[:-1])+np.array(fn_edges[1:]))/2
        bar = figure.colorbar(scatter, ax=axis, shrink=.65, boundaries=fn_edges, ticks=ticks, spacing="uniform")
        bar.ax.set_yticklabels(_frequency_labels(fn_edges)); bar.set_label(label)
    if not points:
        axis.text(.5, .5, "NO CASES IN THIS CLASS", ha="center", transform=axis.transAxes,
                  bbox=dict(facecolor="white", alpha=.9, edgecolor="grey"))
    sizes = sorted({min(p.nominal_power_mva for p in universe),
                    float(np.median([p.nominal_power_mva for p in universe])),
                    max(p.nominal_power_mva for p in universe)})
    handles = [axis.scatter([], [], s=value*size_factor, facecolors="none", edgecolors="grey", label=_mva_legend_label(value)) for value in sizes]
    axis.legend(handles=handles, title="Area proportional to MVA", loc="upper left", fontsize=8)
    axis.set_xlim(bounds[:2]); axis.set_ylim(bounds[2:]); axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("posX (EMTP drawing units)"); axis.set_ylabel("posY (EMTP drawing units)")
    axis.grid(alpha=.15)
    figure.suptitle(f"{title}\n{subtitle}\nColor = {color_text} | Area = MVA | {scale_context}", fontsize=11)
    return figure


def export_maps(analysis, output_directory, assets, profile, runs, *, progress=None):
    import matplotlib.pyplot as plt
    report = progress or (lambda message: None)
    points, cases = analysis.points, analysis.cases
    criterion_label = "voltage + damping + angle" if analysis.include_angle else "voltage + damping; no angle"
    if not points: raise ValueError("No spatial controller points to export")
    if len({case.case_id for case in cases}) != len(cases): raise ValueError("Duplicate case assessment")
    case_map = {case.case_id: case for case in cases}
    mapped_case_ids = {point.case_id for point in points}
    if not mapped_case_ids.issubset(case_map): raise ValueError("Controller/assessment case mismatch")
    for case in cases:
        if case.assessment.status not in STATUS_DIRECTORIES: raise ValueError("Unknown dynamic-stability status")
    names = [_filename(case.case_id) for case in cases]
    if len(names) != len(set(names)): raise ValueError("Run names collide after filename normalization")
    scale_points = tuple(p for p in points if case_map[p.case_id].assessment.status in PLOT_STATUSES) or points
    unique = {}
    for point in scale_points:
        if point.device_id in unique and (point.x,point.y) != (unique[point.device_id].x,unique[point.device_id].y):
            raise ValueError(f"Device coordinates change between cases: {point.device_id}")
        unique[point.device_id] = point
    universe = tuple(unique.values())
    xmin, xmax = min(p.x for p in scale_points), max(p.x for p in scale_points)
    ymin, ymax = min(p.y for p in scale_points), max(p.y for p in scale_points)
    dx, dy = max((xmax-xmin)*.08,1.), max((ymax-ymin)*.08,1.)
    bounds = (xmin-dx,xmax+dx,ymin-dy,ymax+dy)
    xi_lower, xi_upper, xi_ticks = _nice_numeric_scale(
        [point.xi for point in scale_points]
    )
    xi_limits = (xi_lower, xi_upper)
    fn_edges = [0.,2.,5.,10.,20.,max(30.,ceil(max(p.fn_hz for p in scale_points)/10)*10.)]
    size_factor = 450./max(p.nominal_power_mva for p in scale_points)
    out = Path(output_directory).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    point_rows = [asdict(p) for p in points]
    _write_csv(out/"device_case_parameters.csv", point_rows, list(point_rows[0]))
    _write_csv(out/"excluded_devices.csv", list(analysis.excluded_devices), ["case_id","generator_name","reason"])
    skipped_by_case = {item.case_id: item.reason for item in analysis.skipped_runs}
    case_rows = []
    for case in cases:
        selected = tuple(p for p in points if p.case_id == case.case_id)
        row = dict(case_id=case.case_id, **asdict(case.assessment), diagnostics="; ".join(case.diagnostics),
                   spatial_status="Mapped" if selected else "Skipped",
                   spatial_reason=skipped_by_case.get(case.case_id),
                   n_devices=len(selected), included_mva=sum(p.nominal_power_mva for p in selected),
                   weighted_xi=weighted_value(selected,"xi"), weighted_fn_hz=weighted_value(selected,"fn_hz"))
        case_rows.append(row)
    _write_csv(out/"case_assessments.csv", case_rows, list(case_rows[0]))
    # Build each class from its own case IDs, once, and retain the membership for auditing.
    groups = {}
    mapped_cases = tuple(case for case in cases if case.case_id in mapped_case_ids)
    for status in STATUS_DIRECTORIES:
        ids = {case.case_id for case in mapped_cases if case.assessment.status == status}
        averages, rows = mean_points(points, ids)
        groups[status] = (ids, averages, rows)
    mean_xi_values = [
        point.xi
        for status in PLOT_STATUSES
        for point in groups[status][1]
    ]
    mean_xi_lower, mean_xi_upper, mean_xi_ticks = _nice_numeric_scale(
        mean_xi_values or list(xi_limits)
    )
    mean_xi_limits = (mean_xi_lower, mean_xi_upper)
    stable_rows = {row["device_id"]: row for row in groups["Stable"][2]}
    unstable_rows = {row["device_id"]: row for row in groups["Unstable"][2]}
    comparison = []
    for device in sorted(set(stable_rows) | set(unstable_rows)):
        a, b = stable_rows.get(device), unstable_rows.get(device)
        comparison.append(dict(device_id=device,
            n_stable=a["n_cases"] if a else 0, n_unstable=b["n_cases"] if b else 0,
            mean_xi_stable=a["mean_xi"] if a else None, mean_xi_unstable=b["mean_xi"] if b else None,
            delta_xi_unstable_minus_stable=b["mean_xi"]-a["mean_xi"] if a and b else None,
            mean_fn_hz_stable=a["mean_fn_hz"] if a else None, mean_fn_hz_unstable=b["mean_fn_hz"] if b else None,
            delta_fn_hz_unstable_minus_stable=b["mean_fn_hz"]-a["mean_fn_hz"] if a and b else None,
            fn_band_stable=a["frequency_band"] if a else None, fn_band_unstable=b["frequency_band"] if b else None,
            same_fn_band=a["frequency_band"] == b["frequency_band"] if a and b else None))
    comparison_columns = ["device_id","n_stable","n_unstable","mean_xi_stable","mean_xi_unstable",
        "delta_xi_unstable_minus_stable","mean_fn_hz_stable","mean_fn_hz_unstable",
        "delta_fn_hz_unstable_minus_stable","fn_band_stable","fn_band_unstable","same_fn_band"]
    _write_csv(out/"device_class_comparison.csv",comparison,comparison_columns)
    figures, summary = [], []
    report(f"Spatial stability: saved controller values and classifications in {out}")
    from .distributions import export_distributions
    report("Spatial stability: creating distributions by stability")
    distribution_paths = export_distributions(analysis, out)
    figures.extend(distribution_paths)
    report(f"Spatial stability: saved {len(distribution_paths)} distribution figure(s)")
    report("Spatial stability: creating mean maps by stability")
    columns = ["device_id","generator_name","asset_name","x","y","mva","n_cases","n_cases_in_class","mean_xi","mean_fn_hz","frequency_band"]
    for metric, folder_name, prefix in (("xi","xi_maps","xi"),("fn_hz","fn_maps","fn")):
        directory = out/folder_name
        old_unknown_map = directory/"by_stability"/"not_evaluated"/f"{prefix}_mean_map.png"
        if old_unknown_map.is_file():
            old_unknown_map.unlink()
        for status, status_folder in STATUS_DIRECTORIES.items():
            selected_cases, averages, rows = groups[status]
            # Always represent Stable/Unstable, and create an unknown class only when it exists.
            if status == "Not evaluated" and not selected_cases:
                summary.append(dict(metric=metric,status=status,n_cases=0,weighted_population_mean=None))
                continue
            folder = directory/"by_stability"/status_folder
            folder.mkdir(parents=True,exist_ok=True)
            _write_csv(folder/"device_means.csv", rows, columns)
            membership = [row for row in case_rows if row["case_id"] in selected_cases]
            _write_csv(folder/"cases.csv",membership,list(case_rows[0]))
            fleet = [row["weighted_xi" if metric=="xi" else "weighted_fn_hz"] for row in case_rows if row["case_id"] in selected_cases]
            average = mean(fleet) if fleet else None
            summary.append(dict(metric=metric,status=status,n_cases=len(selected_cases),weighted_population_mean=average))
            if status not in PLOT_STATUSES:
                old_map = folder/f"{prefix}_mean_map.png"
                if old_map.is_file():
                    old_map.unlink()
                continue
            old_map = folder/f"{prefix}_mean_map.png"
            if old_map.is_file():
                old_map.unlink()
        figure = _draw_mean_comparison(
            groups["Stable"][1],
            groups["Unstable"][1],
            universe,
            metric,
            bounds,
            mean_xi_limits if metric == "xi" else xi_limits,
            mean_xi_ticks if metric == "xi" else xi_ticks,
            fn_edges,
            size_factor,
            len(groups["Stable"][0]),
            len(groups["Unstable"][0]),
        )
        path = directory/f"{prefix}_mean_map.png"
        try: figure.savefig(path,dpi=150)
        finally: plt.close(figure)
        figures.append(path)
    _write_csv(out/"stability_group_summary.csv", summary, list(summary[0]))
    report("Spatial stability: mean maps by stability saved")
    from .clustering import export_clustering
    report("Spatial stability: creating clustering and PCA figures")
    clustering_paths = export_clustering(analysis, out)
    figures.extend(clustering_paths)
    report(f"Spatial stability: saved {len(clustering_paths)} clustering/PCA figure(s)")
    plotted_cases = tuple(case for case in mapped_cases if case.assessment.status in PLOT_STATUSES)
    total_maps = 2 * len(plotted_cases)
    completed_maps = 0
    last_report = monotonic()
    report(f"Spatial stability: creating {total_maps} individual maps; summary figures are already saved")
    for metric, folder_name, prefix in (("xi","xi_maps","xi"),("fn_hz","fn_maps","fn")):
        directory = out/folder_name
        (directory/"by_case").mkdir(parents=True, exist_ok=True)
        for case in mapped_cases:
            if case.assessment.status not in PLOT_STATUSES:
                old_map = directory/"by_case"/f"{prefix}_map_{_filename(case.case_id)}.png"
                if old_map.is_file():
                    old_map.unlink()
        for case in plotted_cases:
            selected = tuple(p for p in points if p.case_id == case.case_id)
            row = next(row for row in case_rows if row["case_id"] == case.case_id)
            value = row["weighted_xi" if metric=="xi" else "weighted_fn_hz"]
            units = " Hz" if metric=="fn_hz" else ""
            figure = _draw(selected,universe,metric,f"Run {case.case_id}: {'xi' if metric=='xi' else 'natural frequency'}",
                           f"Dynamic stability: {case.assessment.status} ({criterion_label})\n"
                           f"MVA-weighted included population = {value:.3f}{units}", bounds,xi_limits,xi_ticks,fn_edges,size_factor)
            path = directory/"by_case"/f"{prefix}_map_{_filename(case.case_id)}.png"
            try: figure.savefig(path,dpi=150)
            finally: plt.close(figure)
            figures.append(path)
            completed_maps += 1
            now = monotonic()
            if progress and completed_maps < total_maps and now - last_report >= 60:
                report(f"Spatial stability: saved {completed_maps}/{total_maps} individual maps")
                last_report = now
    report(f"Spatial stability: saved {completed_maps}/{total_maps} individual maps")
    metadata = dict(source="Native GridStabilityAssessment parameters.csv and Parquet", assets_workbook=str(assets.workbook_path),
                    run_paths=[str(run.run_path) for run in runs], profile=asdict(profile),
                    criterion=analysis.criterion_description, include_angle=analysis.include_angle,
                    monte_carlo_policy=analysis.monte_carlo_policy,
                    voltage_scope=analysis.voltage_scope,
                    counts={status:sum(c.assessment.status==status for c in cases) for status in STATUS_DIRECTORIES},
                    mapped_counts={status:sum(c.assessment.status==status for c in mapped_cases) for status in STATUS_DIRECTORIES},
                    cases=len(cases),mapped_cases=len(mapped_cases),devices=len(unique),points=len(points),excluded_devices=len(analysis.excluded_devices),
                    plotted_statuses=list(PLOT_STATUSES),
                    hidden_not_evaluated=[c.case_id for c in mapped_cases if c.assessment.status not in PLOT_STATUSES],
                    skipped_runs=[asdict(item) for item in analysis.skipped_runs],
                    xi_formula="Kp/(2*sqrt(Ki))",fn_formula_hz="sqrt(Ki)/(2*pi)",frequency_bands=FN_LABELS,
                    frequency_boundary_rule="lower inclusive, upper exclusive; final band >=20 Hz",
                    xi_color_limits=list(xi_limits),xi_color_ticks=list(xi_ticks),xi_mean_color_limits=list(mean_xi_limits),xi_mean_color_ticks=list(mean_xi_ticks),frequency_color_edges=fn_edges,position_bounds=bounds,marker_area_per_mva=size_factor,
                    group_membership={status:sorted(group[0]) for status,group in groups.items()},
                    weighting="Point area proportional to MVA; repeated active controllers share asset MVA. Equal case weight per active unit for class means. Population mean weighted by MVA per run then equally across runs.",
                    missing_data="Missing evidence is retained as Not evaluated according to native aggregation; not included in Stable means. Excluded coordinates/assets listed in excluded_devices.csv.",
                    distributions="distributions_by_stability/: descriptive comparisons with equal case weights",
                    scope="All active Kp/Ki controllers with mapped coordinates, including constant gains; positions are schematic and are not used as geographic distances.")
    (out/"spatial_metadata.json").write_text(json.dumps(metadata,indent=2,ensure_ascii=False),encoding="utf-8")
    return SpatialStabilityResult(out,analysis,tuple(figures))
