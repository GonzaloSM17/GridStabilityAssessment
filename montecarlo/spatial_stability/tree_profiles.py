"""Decision-tree leaves with native outcomes and class-specific parameter profiles."""
import json
from pathlib import Path
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from .clustering import FEATURES
from .distributions import _csv

NAMES = ("xi BESS", "xi PE", "xi PFV", "xi PMGD",
         "fn BESS [Hz]", "fn PE [Hz]", "fn PFV [Hz]", "fn PMGD [Hz]", "Disp. xi")
FILES = (
    "04_descriptive_trees.png", "04_descriptive_trees_units.png",
    "04_descriptive_trees_mva.png", "descriptive_tree_rules.txt",
    "tree_leaf_profiles.csv", "tree_leaf_summary.csv", "tree_leaf_membership.csv",
    "tree_profile_metadata.json", "TREE_PROFILES.md",
)


def _get(record, key):
    return record.get(key) if isinstance(record, dict) else getattr(record, key, None)


def leaf_profiles(tree, values, y, case_ids, assessments):
    """Summarize actual leaf membership, never infer outcomes from leaf majority."""
    x, y = np.asarray(values), np.asarray(y)
    ids = tuple(case_ids)
    if x.shape[0] != len(y) or len(ids) != len(y) or len(ids) != len(set(ids)):
        raise ValueError("Leaf inputs and unique case IDs must align")
    if x.ndim != 2 or x.shape[1] < 9 or not np.isfinite(x[:, :9]).all():
        raise ValueError("Leaf profiles require nine finite indicators")
    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("Only evaluated native classes can enter the tree")
    for identifier, label in zip(ids, y):
        expected = "Unstable" if label else "Stable"
        if identifier not in assessments or _get(assessments[identifier], "status") != expected:
            raise ValueError("Leaf labels differ from native assessments")
    nodes = tree.tree_
    leaves = tree.apply(x[:, :9])
    paths = {}
    def visit(node, conditions):
        if nodes.children_left[node] == -1:
            paths[node] = conditions
            return
        feature = FEATURES[nodes.feature[node]]
        threshold = float(nodes.threshold[node])
        visit(nodes.children_left[node], conditions+[(feature, "<=", threshold)])
        visit(nodes.children_right[node], conditions+[(feature, ">", threshold)])
    visit(0, [])
    summaries, profiles, membership = [], [], []
    for number, (node, conditions) in enumerate(paths.items(), 1):
        mask = leaves == node
        stable = mask & (y == 0)
        unstable = mask & (y == 1)
        n_stable, n_unstable = int(stable.sum()), int(unstable.sum())
        summary = dict(leaf=number, node=int(node), n_cases=int(mask.sum()),
            stable=n_stable, unstable=n_unstable,
            unstable_fraction=float(n_unstable/mask.sum()),
            small_stable=bool(0 < n_stable < 5), small_unstable=bool(0 < n_unstable < 5),
            gini=float(nodes.impurity[node]),
            path=" AND ".join(f"{feature} {operator} {threshold:.12g}"
                              for feature, operator, threshold in conditions) or "All cases")
        for criterion, field in (("voltage", "voltage_status"),
                                  ("damping", "modal_damping_status"),
                                  ("angle", "angular_status")):
            states = [_get(assessments[c], field) for c, selected in zip(ids, unstable) if selected]
            unknown = {None, "Not evaluated", "Not measurable", "Not applicable"}
            evaluated = [s for s in states if s not in unknown]
            failures = sum(s != "Damped" if criterion == "damping" else s == "Unstable"
                           for s in evaluated)
            summary[criterion+"_failures_unstable"] = int(failures)
            summary[criterion+"_evaluated_unstable"] = len(evaluated)
        summaries.append(summary)
        for j, feature in enumerate(FEATURES[:9]):
            a, b = x[stable, j], x[unstable, j]
            ma, mb = (float(np.median(a)) if len(a) else None,
                      float(np.median(b)) if len(b) else None)
            profiles.append(dict(leaf=number, node=int(node), feature=feature,
                n_stable=n_stable, n_unstable=n_unstable,
                median_stable=ma, median_unstable=mb,
                delta_unstable_minus_stable=mb-ma if ma is not None and mb is not None else None,
                minimum_stable=float(a.min()) if len(a) else None,
                maximum_stable=float(a.max()) if len(a) else None,
                minimum_unstable=float(b.min()) if len(b) else None,
                maximum_unstable=float(b.max()) if len(b) else None))
        for identifier, selected, label in zip(ids, mask, y):
            if selected:
                membership.append(dict(case_id=identifier, leaf=number, node=int(node),
                                       native_status="Unstable" if label else "Stable"))
    return summaries, profiles, membership


def leaf_text(summary, rows):
    """Counts and physical medians replace the ambiguous value=[E,I] label."""
    def value(v, digits=4, signed=False):
        if v is None:
            return "--"
        return f"{v:+.{digits}f}" if signed else f"{v:.{digits}f}"
    n, stable, unstable = summary["n_cases"], summary["stable"], summary["unstable"]
    lines = [
        f"HOJA {summary['leaf']} | {n} corridas",
        f"Estables: {stable} ({100*stable/n:.1f}%)",
        f"Inestables: {unstable} ({100*unstable/n:.1f}%)",
        "",
        "Medianas entre corridas | E=estable, I=inestable",
        f"{'Indicador':<13} {'E':>8} {'I':>8} {'I-E':>9}",
    ]
    for name, row in zip(NAMES, rows):
        digits = 3 if name.startswith("fn ") else 4
        lines.append(f"{name:<13} {value(row['median_stable'],digits):>8} "
                     f"{value(row['median_unstable'],digits):>8} "
                     f"{value(row['delta_unstable_minus_stable'],digits,True):>9}")
    lines += ["", "Fallas / evaluados entre los INESTABLES:"]
    for label, criterion in (("Tension", "voltage"), ("Amortiguamiento modal", "damping"), ("Angulo", "angle")):
        evaluated = summary[criterion+"_evaluated_unstable"]
        failures = summary[criterion+"_failures_unstable"]
        lines.append(f"{label}: {failures}/{evaluated}" if evaluated else f"{label}: sin evaluados")
    if summary["small_stable"] or summary["small_unstable"]:
        classes = ", ".join(name for name, key in (("E", "small_stable"), ("I", "small_unstable"))
                            if summary[key])
        lines += [f"Pocos casos en {classes}: n < 5"]
    if not stable or not unstable:
        lines += ["-- = clase ausente; no se reemplaza por cero"]
    return "\n".join(lines)


def _draw_tree(axis, tree, y, summaries, profiles, basis):
    """Use fixed-depth diagram layout with readable profile tables inside leaves."""
    nodes = tree.tree_
    positions, leaf_nodes = {}, [r["node"] for r in summaries]
    leaf_x = {node:(i+.5)/len(leaf_nodes) for i,node in enumerate(leaf_nodes)}
    def locate(node, depth=0):
        if nodes.children_left[node] == -1:
            positions[node] = (leaf_x[node], .30)
        else:
            locate(nodes.children_left[node], depth+1)
            locate(nodes.children_right[node], depth+1)
            left, right = positions[nodes.children_left[node]], positions[nodes.children_right[node]]
            positions[node] = ((left[0]+right[0])/2, .94-.20*depth)
    locate(0)
    summary_by_node = {r["node"]:r for r in summaries}
    texts = {}
    for node, position in positions.items():
        if node in summary_by_node:
            summary = summary_by_node[node]
            rows = [r for r in profiles if r["node"] == node]
            text = leaf_text(summary, rows)
            majority_stable = summary["stable"] > summary["unstable"]
            fontsize = 10.5
        else:
            counts = np.asarray(nodes.value[node][0], dtype=float).copy()
            counts *= nodes.n_node_samples[node]/counts.sum()
            stable, unstable = int(round(counts[0])), int(round(counts[1]))
            name = NAMES[nodes.feature[node]]
            text = (f"{name} <= {nodes.threshold[node]:.6g}\n"
                    f"Corridas: {nodes.n_node_samples[node]}\n"
                    f"Estables: {stable} | Inestables: {unstable}\nGini: {nodes.impurity[node]:.3f}")
            majority_stable = stable > unstable
            fontsize = 11
        face, edge = ("#e5f0fa", "#0072B2") if majority_stable else ("#fbe7da", "#D55E00")
        texts[node] = axis.text(*position, text, transform=axis.transAxes, ha="center",
            va="center", fontfamily="DejaVu Sans Mono", fontsize=fontsize, linespacing=1.25,
            bbox=dict(boxstyle="round,pad=.55", facecolor=face, edgecolor=edge, linewidth=1.1),
            zorder=3)
    axis.figure.canvas.draw()
    for node, (x,ypos) in positions.items():
        if nodes.children_left[node] == -1:
            continue
        for child, label in ((nodes.children_left[node], "Cumple"),
                              (nodes.children_right[node], "No cumple")):
            target = positions[child]
            axis.annotate("", xy=target, xycoords="axes fraction",
                xytext=(x,ypos), textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color="#444444", linewidth=1.1,
                                patchA=texts[node].get_bbox_patch(),
                                patchB=texts[child].get_bbox_patch(),
                                shrinkA=5, shrinkB=5), zorder=1)
            axis.text((x+target[0])/2,(ypos+target[1])/2+.015,label,
                      transform=axis.transAxes,ha="center",fontsize=9,
                      bbox=dict(facecolor="white", edgecolor="none",pad=1),zorder=2)
    title = ("Igual peso por unidad" if basis == "units" else "Ponderacion por potencia MVA")
    axis.set_title(title+"\nMedianas de indicadores por corrida en cada clase; I-E es una diferencia, no un coeficiente",
                   fontsize=12, pad=22)
    axis.set_axis_off()


def export_descriptive_trees(out, matrices, y, case_ids, assessments, *, seed=1900):
    """Refit only the same descriptive trees; never rerun predictive validation."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    y = np.asarray(y)
    if len(y) < 2 or set(np.unique(y)) != {0,1}:
        raise ValueError("Descriptive trees require both evaluated native classes")
    fitted, summaries, profiles, memberships = {}, [], [], []
    rules = []
    for basis in ("units", "mva"):
        tree = DecisionTreeClassifier(max_depth=2,min_samples_leaf=max(8,int(.1*len(y))),
                                       random_state=seed).fit(matrices[basis][:,:9],y)
        a,b,c = leaf_profiles(tree,matrices[basis],y,case_ids,assessments)
        fitted[basis] = (tree,a,b)
        summaries.extend(dict(basis=basis,**r) for r in a)
        profiles.extend(dict(basis=basis,**r) for r in b)
        memberships.extend(dict(basis=basis,**r) for r in c)
        rules.append(basis+"\n0=Estable; 1=Inestable\n"+
                     export_text(tree, feature_names=list(NAMES),decimals=6))
    _csv(out/"tree_leaf_summary.csv",summaries,list(summaries[0]))
    _csv(out/"tree_leaf_profiles.csv",profiles,list(profiles[0]))
    _csv(out/"tree_leaf_membership.csv",memberships,list(memberships[0]))
    (out/"descriptive_tree_rules.txt").write_text("\n\n".join(rules),encoding="utf-8")
    paths = []
    def figure(height):
        fig = Figure(figsize=(22,height),layout="constrained")
        FigureCanvasAgg(fig)
        return fig
    def finish(fig,name):
        fig.suptitle("Arbol descriptivo con perfiles de xi y fn separados por estabilidad nativa\n"
                     "Azul: mayoria estable | Naranja: mayoria inestable; cada hoja conserva ambas clases\n"
                     "Medianas y umbrales descriptivos de esta muestra; no demuestran influencia causal",
                     fontsize=14)
        path = out/name
        temporary = path.with_name(path.stem+"_pending.png")
        try:
            fig.savefig(temporary,dpi=170)
            temporary.replace(path)
        finally:
            fig.clear()
            if temporary.is_file():
                temporary.unlink()
        paths.append(path)
    fig = figure(23)
    axes = fig.subplots(2,1)
    for axis,basis in zip(axes,("units","mva")):
        tree,a,b = fitted[basis]
        _draw_tree(axis,tree,y,a,b,basis)
    finish(fig,"04_descriptive_trees.png")
    for basis in ("units","mva"):
        fig = figure(12)
        tree,a,b = fitted[basis]
        _draw_tree(fig.subplots(),tree,y,a,b,basis)
        finish(fig,f"04_descriptive_trees_{basis}.png")
    metadata = dict(
        seed=seed,max_depth=2,min_samples_leaf=max(8,int(.1*len(y))),case_ids=list(case_ids),
        features=list(FEATURES[:9]),summary="Class-specific medians and median(I)-median(E); raw ranges in CSV",
        native_labels="Unchanged; class-specific counts use native labels, never node predictions",
        criteria="Failures/evaluated within native Unstable cases; criteria can overlap; unknown evidence excluded from each denominator",
        missing_class="Shown as --; never imputed as zero",
        small_class="Flag if 0 < class count < 5; descriptive summaries can be unstable with few cases",
        interpretation="Descriptive post-hoc leaf profiles of full-sample trees; not causal effects, validated thresholds or out-of-sample leaf performance",
        figures=[p.name for p in paths])
    (out/"tree_profile_metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    (out/"TREE_PROFILES.md").write_text(
        "# Perfiles dentro de las hojas del arbol\n\n"
        "Se mantienen las mismas divisiones del arbol: profundidad 2 y minimo de casos "
        "por hoja igual a max(8, 10 % de la muestra). Los nuevos datos no generan divisiones "
        "adicionales ni cambian el criterio de estabilidad.\n\n"
        "Cada hoja muestra conteos y porcentajes de estables/inestables y una tabla con "
        "los nueve indicadores: ocho promedios xi/fn por tecnologia y dispersion global xi. "
        "E e I son medianas entre las corridas estables e inestables de esa hoja. "
        "I-E es diferencia de medianas, no un coeficiente de modelo ni una diferencia causal. "
        "En la version unidades los indicadores usan igual peso por dispositivo; en MVA "
        "usan potencia nominal. Los rangos completos de cada clase se guardan en CSV.\n\n"
        "Se reemplaza la etiqueta ambigua value=[E,I] por conteos explicitos. "
        "Azul y naranja indican la clase mayoritaria, no que todos los casos compartan "
        "la clase. Las clases ausentes se muestran con --, nunca con cero. "
        "Clases con menos de cinco casos se marcan como muestra pequena.\n\n"
        "Al final de cada hoja se cuentan fallos/evaluados de tension, amortiguamiento modal "
        "y angulo entre sus casos inestables. Los criterios pueden fallar simultaneamente. "
        "Una evidencia no evaluada queda fuera del denominador correspondiente.\n\n"
        "04_descriptive_trees.png contiene ambas ponderaciones; las versiones _units y _mva "
        "facilitan la lectura. tree_leaf_profiles.csv conserva medianas, diferencias y rangos; "
        "tree_leaf_summary.csv conteos y criterios; tree_leaf_membership.csv identifica "
        "las corridas de cada hoja. Los umbrales completos estan en descriptive_tree_rules.txt. "
        "Los perfiles exploran una muestra usada para ajustar el arbol; no confirman reglas "
        "operativas ni efectos de modificar controladores. No se repite la validacion de modelos.\n",
        encoding="utf-8")
    return tuple(paths)
