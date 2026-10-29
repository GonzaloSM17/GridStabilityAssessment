"""Compare stability-characterization models without a PCA transform.

Native stability is the observed outcome; predicted probabilities never replace it.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, log_loss, brier_score_loss,
    balanced_accuracy_score, recall_score, roc_curve, precision_recall_curve,
)
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

from .clustering import FEATURES, build_case_profiles
from .distributions import _csv, load_saved_analysis
from .weighted_pca import weighted_profiles
from .tree_profiles import export_global_descriptive_trees, build_global_tree_inputs, FILES as TREE_PROFILE_FILES

FEATURE_NAMES = (*FEATURES, "xi_fn_correlation")
FAMILIES = ("logistic", "interactions", "tree", "forest", "boosting", "svm", "naive_bayes")
SETS = {"reference10": tuple(range(10)), "candidate9": tuple(range(9)),
        "candidate9_correlation": (*range(9), 10)}
ABLATIONS = {
    "without_mean_xi": tuple(range(4, 10)),
    "without_mean_fn": (*range(4), 8, 9),
    "without_xi_spread": (*range(8), 9),
}
FAMILY_LABELS = {
    "logistic": "Logistica", "interactions": "Logistica + interacciones",
    "tree": "Arbol pequeno", "forest": "Bosque aleatorio", "boosting": "Boosting",
    "svm": "SVM radial", "naive_bayes": "Naive Bayes", "prior": "Solo prevalencia",
    "nested_selection": "Seleccion dentro del entrenamiento",
}
SHORT_FEATURES = ("xi BESS", "xi PE", "xi PFV", "xi PMGD",
                  "fn BESS", "fn PE", "fn PFV", "fn PMGD", "Dispersion xi")
OWN_FILES = (
    "01_model_comparison.png", "02_block_contribution.png", "03_validation_curves.png",
    "04_descriptive_trees.png", "05_parameter_associations.png", "model_summary.csv",
    "cv_predictions.csv", "block_contribution.csv", "case_features.csv", "excluded_cases.csv",
    "selected_models.csv", "parameter_associations.csv", "descriptive_tree_rules.txt",
    "characterization_metadata.json", "README.md", *TREE_PROFILE_FILES,
)


class SelectColumns(TransformerMixin, BaseEstimator):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.columns]


def build_inputs(analysis):
    profiles = build_case_profiles(analysis)
    statuses = {c.case_id: c.assessment.status for c in analysis.cases}
    ids = tuple(c for c in profiles.case_ids if statuses[c] in {"Stable", "Unstable"})
    excluded = list(profiles.excluded)
    excluded.extend(dict(case_id=c, reason="Native stability not evaluated")
                    for c in profiles.case_ids if statuses[c] == "Not evaluated")
    y = np.array([statuses[c] == "Unstable" for c in ids], dtype=int)
    grouped = defaultdict(list)
    for point in analysis.points:
        grouped[point.case_id].append(point)
    matrices = {}
    for basis in ("units", "mva"):
        if not ids:
            matrices[basis] = np.empty((0, 11))
            continue
        selected = weighted_profiles(analysis, ids, basis)
        lookup = dict(zip(selected.case_ids, selected.values))
        rows = []
        for identifier in ids:
            points = grouped[identifier]
            w = np.ones(len(points)) if basis == "units" else np.array([
                p.nominal_power_mva for p in points])
            w /= w.sum()
            xi = np.array([p.xi for p in points])
            fn = np.array([p.fn_hz for p in points])
            dx, df = xi-w@xi, fn-w@fn
            denominator = np.sqrt((w@(dx*dx)) * (w@(df*df)))
            correlation = float(np.clip(w@(dx*df)/denominator, -1., 1.)) if denominator > 0 else np.nan
            rows.append([*lookup[identifier], correlation])
        matrices[basis] = np.asarray(rows)
    return ids, y, matrices, excluded


def model_spec(family, columns, seed):
    scale, poly, postscale = "passthrough", "passthrough", "passthrough"
    if family in {"logistic", "interactions", "svm", "naive_bayes"}:
        scale = StandardScaler()
    if family == "logistic":
        estimator = LogisticRegression(max_iter=2000, solver="lbfgs")
        grid = {"model__C": [.01, .1, 1., 10.]}
    elif family == "interactions":
        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        postscale = StandardScaler()
        estimator = LogisticRegression(max_iter=3000, solver="lbfgs")
        grid = {"model__C": [.001, .01, .1]}
    elif family == "tree":
        estimator = DecisionTreeClassifier(random_state=seed)
        grid = [{"model__max_depth": [2], "model__min_samples_leaf": [8]},
                {"model__max_depth": [3], "model__min_samples_leaf": [12]}]
    elif family == "forest":
        estimator = RandomForestClassifier(n_estimators=80, max_depth=3,
            min_samples_leaf=8, random_state=seed, n_jobs=1)
        grid = {"model__max_features": ["sqrt", 1.]}
    elif family == "boosting":
        estimator = HistGradientBoostingClassifier(max_iter=80, learning_rate=.05,
            max_leaf_nodes=7, min_samples_leaf=10, early_stopping=False, random_state=seed)
        grid = {"model__l2_regularization": [1., 10.]}
    elif family == "svm":
        estimator = SVC(kernel="rbf", gamma="scale", random_state=seed)
        grid = {"estimator__model__C": [.1, 1., 10.]}
    elif family == "naive_bayes":
        estimator = GaussianNB()
        grid = {"model__var_smoothing": [1e-8, .01]}
    elif family == "prior":
        estimator, grid = DummyClassifier(strategy="prior"), {}
    else:
        raise ValueError(f"Unknown model family: {family}")
    pipeline = Pipeline([
        ("select", SelectColumns(columns)),
        ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("scale", scale), ("interactions", poly), ("postscale", postscale),
        ("model", estimator)])
    if family == "svm":
        return CalibratedClassifierCV(estimator=pipeline, method="sigmoid",
                                      cv=2, ensemble=False), grid
    return pipeline, grid


def score_predictions(y, probability, threshold=.5):
    predicted = probability >= threshold
    return dict(
        auc=float(roc_auc_score(y, probability)),
        stable_average_precision=float(average_precision_score(1-y, 1-probability)),
        log_loss=float(log_loss(y, probability, labels=[0, 1])),
        brier=float(brier_score_loss(y, probability)),
        balanced_accuracy=float(balanced_accuracy_score(y, predicted)),
        stable_recall=float(recall_score(y, predicted, pos_label=0, zero_division=0)),
        unstable_recall=float(recall_score(y, predicted, pos_label=1, zero_division=0)))


def compare_models(matrices, y, *, families=FAMILIES, repeats=5, outer_folds=5,
                   inner_folds=3, seed=1900, progress=None):
    """Nested evaluation of fixed options and the whole inner-selection procedure."""
    y = np.asarray(y, dtype=int)
    if len(y) < 16 or set(np.unique(y)) != {0, 1} or min(np.bincount(y)) < 8:
        return dict(status="not_fitted", reason="Need at least eight evaluated cases in each class.")
    if repeats < 2 or outer_folds < 2 or inner_folds < 2 or outer_folds > min(np.bincount(y)):
        raise ValueError("Invalid repeated cross-validation configuration")
    if any(f not in FAMILIES for f in families) or not families or len(set(families)) != len(families):
        raise ValueError("Choose at least one supported model family")
    for values in matrices.values():
        if np.shape(values) != (len(y), 11) or not np.isfinite(values[:, :10]).all():
            raise ValueError("Expected aligned eleven-column inputs, finite in first ten")
    candidates = [(family+"__"+name, family, name, columns, True)
                  for family in families for name, columns in SETS.items()]
    candidates.append(("prior", "prior", "prior", tuple(range(10)), True))
    if "logistic" in families:
        candidates.extend(("logistic__"+name, "logistic", name, columns, False)
                          for name, columns in ABLATIONS.items())
    descriptors = {key: dict(family=family, feature_set=name, selection_candidate=eligible)
                   for key, family, name, columns, eligible in candidates}
    descriptors["nested_selection"] = dict(family="nested_selection", feature_set="inner_selected",
                                           selection_candidate=False)
    probabilities = {basis: {key: np.zeros((repeats, len(y))) for key in descriptors}
                     for basis in matrices}
    thresholds = {basis: np.full((repeats, len(y)), .5) for basis in matrices}
    fold_ids = np.zeros((repeats, len(y)), dtype=int)
    selections = []
    started = time.perf_counter()
    with threadpool_limits(limits=1):
        for repeat in range(repeats):
            outer = StratifiedKFold(outer_folds, shuffle=True, random_state=seed+repeat)
            for fold, (train, test) in enumerate(outer.split(np.zeros(len(y)), y)):
                fold_ids[repeat, test] = fold
                inner = StratifiedKFold(inner_folds, shuffle=True,
                                        random_state=seed+3200+repeat*outer_folds+fold)
                if min(np.bincount(y[train])) < inner_folds:
                    raise ValueError("Training class too small for inner folds")
                splits = list(inner.split(np.zeros(len(train)), y[train]))
                for basis, values in matrices.items():
                    choices = []
                    for key, family, name, columns, eligible in candidates:
                        pipeline, grid = model_spec(family, columns, seed+repeat*outer_folds+fold)
                        search = GridSearchCV(pipeline, grid, scoring="neg_log_loss",
                                              cv=splits, n_jobs=1, error_score="raise")
                        search.fit(values[train], y[train])
                        prediction = search.predict_proba(values[test])[:, 1]
                        probabilities[basis][key][repeat, test] = prediction
                        if eligible:
                            choices.append((float(search.best_score_), key, search))
                    _, key, chosen = max(choices, key=lambda r: (r[0], r[1]))
                    probabilities[basis]["nested_selection"][repeat, test] = probabilities[basis][key][repeat, test]
                    # Select a classification threshold from training-only OOF predictions.
                    inner_probability = cross_val_predict(chosen.best_estimator_, values[train],
                        y[train], cv=splits, method="predict_proba", n_jobs=1)[:, 1]
                    options = np.linspace(.05, .95, 19)
                    scores = [balanced_accuracy_score(y[train], inner_probability >= t) for t in options]
                    selected_threshold = float(options[max(range(len(options)),
                        key=lambda j: (scores[j], -abs(options[j]-.5)))])
                    thresholds[basis][repeat, test] = selected_threshold
                    selections.append(dict(basis=basis, repeat=repeat, fold=fold,
                        model=key, family=descriptors[key]["family"],
                        feature_set=descriptors[key]["feature_set"],
                        inner_log_loss=-float(chosen.best_score_),
                        threshold=selected_threshold, parameters=chosen.best_params_))
            if progress:
                progress(repeat+1, repeats)
    summaries, blocks = [], []
    for basis, predictions in probabilities.items():
        scores_by_key = {}
        for key, values in predictions.items():
            scores = [score_predictions(y, values[r],
                thresholds[basis][r] if key == "nested_selection" else .5) for r in range(repeats)]
            scores_by_key[key] = scores
            summary = dict(basis=basis, model=key, **descriptors[key])
            for metric in scores[0]:
                numbers = np.array([s[metric] for s in scores])
                summary[metric] = float(numbers.mean())
                summary[metric+"_partition_sd"] = float(numbers.std())
            summaries.append(summary)
        if "logistic__reference10" in predictions:
            full = scores_by_key["logistic__reference10"]
            removed = {**{k: "logistic__"+k for k in ABLATIONS},
                       "without_fn_spread": "logistic__candidate9"}
            for label, key in removed.items():
                reduced = scores_by_key[key]
                for r in range(repeats):
                    blocks.append(dict(basis=basis, removed=label, repeat=r,
                        delta_auc=full[r]["auc"]-reduced[r]["auc"],
                        delta_log_loss=reduced[r]["log_loss"]-full[r]["log_loss"]))
    return dict(status="fitted", reason=None, summaries=summaries, blocks=blocks,
                probabilities=probabilities, thresholds=thresholds, fold_ids=fold_ids,
                selections=selections, descriptors=descriptors,
                elapsed_seconds=time.perf_counter()-started,
                config=dict(repeats=repeats, outer_folds=outer_folds,
                            inner_folds=inner_folds, seed=seed, families=list(families)))


def parameter_associations(x, y, *, seed=42, bootstrap=1000):
    """Hedges g with case-level stratified bootstrap; descriptive, no feature selection."""
    rng = np.random.default_rng(seed)
    stable, unstable = x[y == 0, :9], x[y == 1, :9]
    def difference(a, b):
        denominator = np.sqrt(((len(a)-1)*a.var(axis=0, ddof=1)
            +(len(b)-1)*b.var(axis=0, ddof=1))/(len(a)+len(b)-2))
        raw = np.divide(b.mean(axis=0)-a.mean(axis=0), denominator,
                        out=np.full(9, np.nan), where=denominator > 0)
        return raw*(1-3/(4*(len(a)+len(b)-2)-1))
    point = difference(stable, unstable)
    draws = np.array([difference(stable[rng.integers(len(stable), size=len(stable))],
                                 unstable[rng.integers(len(unstable), size=len(unstable))])
                      for _ in range(bootstrap)])
    rows = []
    for j in range(9):
        finite = draws[:, j][np.isfinite(draws[:, j])]
        rows.append(dict(
            feature=FEATURES[j], hedges_g=float(point[j]) if np.isfinite(point[j]) else None,
            bootstrap_lower=float(np.quantile(finite, .025)) if len(finite) else None,
            bootstrap_upper=float(np.quantile(finite, .975)) if len(finite) else None))
    return rows


def export_characterization(analysis, output_directory, *, families=FAMILIES, repeats=5,
                            outer_folds=5, inner_folds=3, seed=1900, progress=None):
    """Save an independent report; existing clustering PNGs are never written."""
    ids, y, matrices, excluded = build_inputs(analysis)
    out = Path(output_directory)/"stability_characterization"
    out.mkdir(parents=True, exist_ok=True)
    if (out/"characterization_metadata.json").is_file():
        for name in OWN_FILES:
            path = out/name
            if path.is_file():
                path.unlink()
    _csv(out/"excluded_cases.csv", excluded, ["case_id", "reason"])
    input_rows = [dict(basis=basis, case_id=c, status="Unstable" if y[i] else "Stable",
                      **{name:float(v) if np.isfinite(v) else None
                         for name,v in zip(FEATURE_NAMES, values[i])})
                  for basis, values in matrices.items() for i,c in enumerate(ids)]
    _csv(out/"case_features.csv", input_rows, ["basis", "case_id", "status", *FEATURE_NAMES])
    tree_paths = export_global_descriptive_trees(analysis, out, seed=seed)
    tree_metadata = json.loads((out/"tree_profile_metadata.json").read_text(encoding="utf-8"))
    result = compare_models(matrices, y, families=families, repeats=repeats,
        outer_folds=outer_folds, inner_folds=inner_folds, seed=seed, progress=progress)
    metadata = dict(status=result["status"], reason=result["reason"], n_cases=len(ids),
        stable=int(np.sum(y == 0)), unstable=int(np.sum(y == 1)), excluded=excluded,
        native_criterion=("Existing voltage + modal damping + angle aggregation, unchanged"
                          if analysis.include_angle else analysis.criterion_description),
        selection_metric="Inner-fold log loss; family and indicator set selected before outer test",
        case_ids=list(ids), feature_sets={key:[FEATURE_NAMES[i] for i in cols] for key,cols in SETS.items()},
        missing_correlation="Training-fold median imputation plus missingness indicator; never imputed using test cases",
        scope="Exploratory within one study; feature hypotheses were motivated by earlier analyses of this same sample. Confirmation needs new simulations.",
        interpretation="Conditional predictive association, not physical causality. Partition SD is not a confidence interval. Repeats share cases. No simulated label is replaced by a model output.",
        tree_n_cases=len(tree_metadata.get("case_ids", [])),
        tree_case_ids=tree_metadata.get("case_ids", []), tree_features=tree_metadata.get("features", []),
        figures=[p.name for p in tree_paths])
    if result["status"] != "fitted":
        (out/"characterization_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        (out/"README.md").write_text(result["reason"]+"\n\n"+
            ("Arboles globales disponibles: TREE_PROFILES.md.\n" if tree_paths else ""), encoding="utf-8")
        return tree_paths
    metadata.update(config=result["config"], elapsed_seconds=result["elapsed_seconds"])
    _csv(out/"model_summary.csv", result["summaries"], list(result["summaries"][0]))
    _csv(out/"block_contribution.csv", result["blocks"],
         ["basis", "removed", "repeat", "delta_auc", "delta_log_loss"])
    selected = [dict(**{k:v for k,v in row.items() if k!="parameters"},
                     parameters=json.dumps(row["parameters"], sort_keys=True))
                for row in result["selections"]]
    _csv(out/"selected_models.csv", selected, list(selected[0]))
    predictions = []
    for basis, variants in result["probabilities"].items():
        for key, values in variants.items():
            for repeat in range(repeats):
                for i, identifier in enumerate(ids):
                    predictions.append(dict(basis=basis, model=key, repeat=repeat,
                        fold=int(result["fold_ids"][repeat,i]), case_id=identifier,
                        native_status="Unstable" if y[i] else "Stable",
                        probability_unstable=float(values[repeat,i]),
                        threshold=float(result["thresholds"][basis][repeat,i]) if key=="nested_selection" else .5))
    _csv(out/"cv_predictions.csv", predictions, list(predictions[0]))
    result["case_ids"] = ids
    result["native_assessments"] = {c.case_id:c.assessment for c in analysis.cases}
    paths = tuple(sorted((*tree_paths, *_export_figures(out, matrices, y, result)), key=lambda p: p.name))
    metadata["figures"] = [p.name for p in paths]
    (out/"characterization_metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    (out/"README.md").write_text(_description(result, metadata), encoding="utf-8")
    return paths


def _export_figures(out, matrices, y, result):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    paths = []
    def figure(width, height):
        fig = Figure(figsize=(width, height), layout="constrained")
        FigureCanvasAgg(fig)
        return fig
    def save(fig, name):
        path = out/name
        try:
            fig.savefig(path, dpi=170)
        finally:
            fig.clear()
        paths.append(path)
    lookup = {(r["basis"],r["model"]):r for r in result["summaries"]}
    families = result["config"]["families"]
    fig = figure(13, max(5, .65*len(families)+2))
    axes = fig.subplots(1, 2)
    for axis,basis in zip(axes, ("units", "mva")):
        values = np.array([[lookup[basis,f+"__"+s]["auc"] for s in SETS] for f in families])
        image = axis.imshow(values, cmap="viridis", vmin=.5, vmax=1, aspect="auto")
        axis.set_xticks(range(3), ["10 indicadores", "9 indicadores", "9 + correlacion"])
        axis.set_yticks(range(len(families)), [FAMILY_LABELS[f] for f in families])
        for i in range(len(families)):
            for j in range(3):
                axis.text(j,i,f"{values[i,j]:.3f}",ha="center",va="center",
                          color="white" if values[i,j]<.77 else "black")
        score = lookup[basis,"nested_selection"]["auc"]
        axis.set_title(f"{basis}: seleccion dentro del entrenamiento AUC={score:.3f}")
    fig.colorbar(image, ax=axes.tolist(), label="AUC media de validacion (no porcentaje de aciertos)")
    fig.suptitle("Comparacion sin PCA: metodos lineales, reglas y relaciones no lineales\n"
                 "Cada celda fija una opcion; la seleccion global se evalua por separado")
    save(fig, "01_model_comparison.png")

    if result["blocks"]:
        fig = figure(13, 5)
        axes = fig.subplots(1, 2)
        names = ("without_mean_xi", "without_mean_fn", "without_xi_spread", "without_fn_spread")
        labels = ("Promedios xi", "Promedios fn", "Dispersion xi", "Dispersion fn")
        for axis,basis in zip(axes, ("units", "mva")):
            for i,name in enumerate(names):
                values = [r["delta_auc"] for r in result["blocks"]
                          if r["basis"]==basis and r["removed"]==name]
                axis.barh(i, np.mean(values), color="#4477AA" if np.mean(values)>=0 else "#CC6677")
                axis.scatter(values, np.full(len(values),i), color="#222222", s=14, alpha=.5)
            axis.axvline(0,color="gray",lw=1)
            axis.set(yticks=range(4),yticklabels=labels,title=basis,
                     xlabel="AUC completo - AUC sin el bloque (positivo = retirar empeora)")
            axis.grid(axis="x",alpha=.2)
        fig.suptitle("Aporte de bloques en regresion logistica\n"
                     "Barras: media; puntos: particiones repetidas, no experimentos independientes")
        save(fig, "02_block_contribution.png")

    fig = figure(13, 10)
    axes = fig.subplots(2, 2)
    keys = ["nested_selection", "logistic__reference10", "logistic__candidate9", "prior"]
    for row,basis in enumerate(("units", "mva")):
        for key in keys:
            if key not in result["probabilities"][basis]:
                continue
            p = result["probabilities"][basis][key][0]
            fpr,tpr,_ = roc_curve(y,p)
            precision,recall,_ = precision_recall_curve(1-y,1-p)
            label = ("Seleccion interna" if key=="nested_selection" else
                     "Logistica 10" if key=="logistic__reference10" else
                     "Logistica 9" if key=="logistic__candidate9" else "Prevalencia")
            axes[row,0].plot(fpr,tpr,label=label)
            axes[row,1].step(recall,precision,where="post",label=label)
        axes[row,0].plot([0,1],[0,1],color="gray",ls="--")
        axes[row,1].axhline(float(np.mean(y==0)),color="gray",ls="--")
        axes[row,0].set(title=basis+" | ROC: inestable",xlabel="Falsos positivos",ylabel="Sensibilidad")
        axes[row,1].set(title=basis+" | Precision-recall: estable",
                        xlabel="Fraccion de estables detectados",ylabel="Precision para estables")
        for axis in axes[row]:
            axis.legend(fontsize=8)
            axis.grid(alpha=.2)
            axis.set(xlim=(0,1),ylim=(0,1.02))
    fig.suptitle("Predicciones fuera del entrenamiento: primera repeticion prefijada\n"
                 "Los resumenes principales promedian todas las repeticiones")
    save(fig, "03_validation_curves.png")

    fig = figure(13, 7)
    axes = fig.subplots(1, 2, sharey=True)
    association_rows = []
    for axis,basis in zip(axes, ("units", "mva")):
        rows = parameter_associations(matrices[basis],y,seed=result["config"]["seed"])
        association_rows.extend(dict(basis=basis,**r) for r in rows)
        for i, r in enumerate(rows):
            if r["hedges_g"] is None:
                axis.text(0, i, "Sin dispersion intraclase", fontsize=7)
                continue
            if r["bootstrap_lower"] is not None:
                axis.plot([r["bootstrap_lower"],r["bootstrap_upper"]],[i,i],color="#4477AA",lw=1.5)
            axis.scatter([r["hedges_g"]],[i],c="#4477AA")
        axis.axvline(0,c="gray",lw=1)
        axis.set(title=basis,yticks=range(9),yticklabels=SHORT_FEATURES,
                 xlabel="Diferencia estandarizada: inestable - estable (Hedges g)")
        axis.grid(axis="x",alpha=.2)
    _csv(out/"parameter_associations.csv",association_rows,list(association_rows[0]))
    fig.suptitle("Perfil de parametros por estabilidad nativa\n"
                 "Intervalos bootstrap 95 % por corrida; descriptivos, sin correccion por comparaciones multiples")
    save(fig, "05_parameter_associations.png")
    return tuple(paths)


def _description(result, metadata):
    lines = [
        "# Caracterizacion de estabilidad sin PCA", "",
        f"Casos: {metadata['n_cases']}; estables: {metadata['stable']}; "
        f"inestables: {metadata['unstable']}. La etiqueta dinamica original no se cambia.", "",
        "Se comparan siete familias (o el subconjunto configurado) sobre diez indicadores, "
        "nueve sin dispersion fn, y nueve mas correlacion xi-fn dentro de cada corrida. "
        "Unidades/MVA modifican los promedios y dispersiones dentro del caso. "
        "No hay transformacion PCA en ningun modelo de este analisis.", "",
        "La correlacion nueva utiliza pares xi-fn del mismo dispositivo. Si no esta definida "
        "por varianza nula, se imputa con la mediana del entrenamiento y un indicador de ausencia. "
        "Toda imputacion, estandarizacion e interaccion se ajusta dentro de los folds de entrenamiento.", "",
        "Validacion anidada estratificada: la regularizacion, familia y conjunto de indicadores "
        "se seleccionan por log loss dentro del entrenamiento; los casos externos solo se usan "
        "para evaluacion. Se incluye un modelo de prevalencia como candidato y referencia. "
        "La fila nested_selection evalua la seleccion completa. Comparar el mejor metodo fijo "
        "despues de ver la tabla es exploratorio y necesita confirmacion con nuevas corridas.", "",
        "El umbral binario de nested_selection se elige por balanced accuracy con predicciones "
        "internas fuera de entrenamiento. Los demas metodos usan 0.5, sin optimizar en el test. "
        "Las probabilidades y predicciones del modelo nunca reemplazan las etiquetas originales.", "",
        "AUC mide ordenamiento, no porcentaje de aciertos. Stable average precision toma "
        "estable como clase positiva para evaluar la clase minoritaria. Log loss y Brier: "
        "menor es mejor. El promedio y desviacion entre repeticiones no son un intervalo "
        "de confianza: las particiones comparten corridas. No se atribuye causalidad fisica.", "",
        "Los nueve indicadores surgieron de analisis previos de esta muestra. Esta validacion "
        "es exploratoria dentro de la misma campana; no es una prueba externa independiente.", "",
        "01: comparacion AUC por familia e indicadores; 02: ablacion de bloques logistica; "
        "03: curvas de primera repeticion prefijada; 04: arboles globales (promedios xi/fn y "
        "dispersiones xi/fn; sin indicadores por tecnologia) descriptivos ajustados con "
        "todos los datos, con medianas por clase y fallos de criterios dentro de cada hoja "
        "(sus umbrales no son limites operativos ni validacion). Las versiones units/mva "
        "permiten leer los perfiles en detalle; TREE_PROFILES.md explica la lectura. "
        "05: diferencias de parametros e intervalos bootstrap por corrida, sin seleccion de "
        "variables ni pruebas multiples a partir de estos intervalos.", "",
        "| Ponderacion | Procedimiento | AUC | AP estables | Log loss |",
        "|---|---|---:|---:|---:|",
    ]
    for row in result["summaries"]:
        if row["model"] in {"nested_selection","logistic__reference10","logistic__candidate9","prior"}:
            lines.append(f"| {row['basis']} | {row['model']} | {row['auc']:.3f} | "
                         f"{row['stable_average_precision']:.3f} | {row['log_loss']:.3f} |")
    lines += ["", "model_summary.csv resume cada opcion; selected_models.csv conserva la "
              "seleccion interna por fold y sus parametros; cv_predictions.csv identifica "
              "cada prediccion externa; block_contribution.csv contiene diferencias pareadas; "
              "case_features.csv permite auditar entradas y etiquetas.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, help="Existing 05_Spatial_Stability; omitted opens picker")
    parser.add_argument("--output", type=Path, help="Parent of stability_characterization/; omitted opens picker")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--trees-only", action="store_true",
                        help="Update only descriptive trees and leaf profiles; reuse native outcomes")
    parser.add_argument("--families", nargs="+", choices=FAMILIES, default=list(FAMILIES))
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
    from utils.output_directory import select_output_directory
    analysis = load_saved_analysis(args.results)
    output = select_output_directory(args.output)
    if args.trees_only:
        ids,y,matrices,_ = build_global_tree_inputs(analysis)
        out = output/"stability_characterization"
        previous = out/"characterization_metadata.json"
        metadata = json.loads(previous.read_text(encoding="utf-8")) if previous.is_file() else None
        seed = metadata.get("config",{}).get("seed",1900) if metadata else 1900
        paths = export_global_descriptive_trees(analysis,out,seed=seed)
        if metadata:
            metadata["figures"] = list(dict.fromkeys([*metadata.get("figures",[]),*[p.name for p in paths]]))
            metadata["tree_profiles"] = "Four global xi/fn means and dispersions; TREE_PROFILES.md"
            metadata["tree_case_ids"] = list(ids)
            metadata["tree_native_criterion"] = analysis.criterion_description
            previous.write_text(json.dumps(metadata,indent=2,allow_nan=False),encoding="utf-8")
        readme = out/"README.md"
        if readme.is_file() and "TREE_PROFILES.md" not in readme.read_text(encoding="utf-8"):
            with readme.open("a",encoding="utf-8") as stream:
                stream.write("\n\nPerfiles por clase en cada hoja: TREE_PROFILES.md. "
                             "Versiones ampliadas units/mva; CSV con medianas, rangos y criterios.\n")
    else:
        paths = export_characterization(analysis, output, repeats=args.repeats, families=tuple(args.families))
    print(f"Created {len(paths)} characterization PNGs in {output/'stability_characterization'}")


if __name__ == "__main__":
    main()
