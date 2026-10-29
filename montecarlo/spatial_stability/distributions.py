"""Descriptive parameter distributions grouped by native dynamic stability.

Case summaries use one observation per run. Population ECDFs normalize MVA
within each run, then give every run equal weight within its stability class.
"""
from collections import defaultdict
from dataclasses import fields
from pathlib import Path
import argparse
import csv
import json
from math import isfinite
import numpy as np

from stability.dynamic_stability import assess_dynamic_stability, assess_monte_carlo_stability
from .models import SpatialControllerPoint, SpatialCaseAssessment, SpatialStabilityAnalysis, SpatialSkippedRun

STATUSES = ("Stable", "Unstable", "Not evaluated")
COLORS = {"Stable":"#0072B2", "Unstable":"#D55E00", "Not evaluated":"#777777"}
LABELS = {"Stable":"Estable", "Unstable":"Inestable", "Not evaluated":"No evaluado"}


def _csv(path, rows, columns):
    with path.open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=columns)
        writer.writeheader();writer.writerows(rows)


def weighted_ecdf(values, weights=None):
    """Return distinct x values and normalized cumulative mass, including ties."""
    x=np.asarray(values,dtype=float)
    if x.ndim != 1 or not np.isfinite(x).all():raise ValueError("ECDF values must be finite and one-dimensional")
    w=np.ones(len(x)) if weights is None else np.asarray(weights,dtype=float)
    if w.shape != x.shape or not np.isfinite(w).all() or np.any(w<0):
        raise ValueError("ECDF weights must be finite, nonnegative and aligned")
    if not len(x):return np.array([],dtype=float),np.array([],dtype=float)
    if w.sum()<=0:raise ValueError("ECDF weights must have positive total mass")
    positive=w>0;x=x[positive];w=w[positive]
    order=np.argsort(x);x=x[order];w=w[order]
    unique,starts=np.unique(x,return_index=True)
    cumulative=np.cumsum(np.add.reduceat(w,starts))/w.sum()
    cumulative=np.clip(cumulative,0,1);cumulative[-1]=1.
    return unique,cumulative


def _validate(analysis):
    cases={case.case_id:case for case in analysis.cases}
    if len(cases)!=len(analysis.cases):raise ValueError("Duplicate case labels")
    if not cases or not analysis.points:raise ValueError("Distribution plots require cases and controller values")
    mapped_ids={p.case_id for p in analysis.points}
    if not mapped_ids.issubset(cases):raise ValueError("Controller cases and stability labels do not match")
    if len({(p.case_id,p.device_id) for p in analysis.points})!=len(analysis.points):
        raise ValueError("Duplicate device/case observations")
    for case in cases.values():
        if case.assessment.status not in STATUSES:raise ValueError("Unknown stability class")
    for p in analysis.points:
        if not all(isfinite(v) for v in (p.nominal_power_mva,p.xi,p.fn_hz)) or p.nominal_power_mva<=0 or p.fn_hz<=0:
            raise ValueError("Distributions require finite parameters, positive MVA and positive fn")
    return {case_id:case for case_id,case in cases.items() if case_id in mapped_ids}


def case_features(analysis, low_xi_threshold=.7, high_fn_threshold_hz=10.):
    cases=_validate(analysis)
    if not isfinite(low_xi_threshold) or low_xi_threshold<=0 or not isfinite(high_fn_threshold_hz) or high_fn_threshold_hz<=0:
        raise ValueError("Exploratory thresholds must be positive and finite")
    grouped=defaultdict(list)
    for p in analysis.points:grouped[p.case_id].append(p)
    rows=[]
    for case_id,points in sorted(grouped.items()):
        w=np.array([p.nominal_power_mva for p in points]);total=float(w.sum());w=w/total
        xi=np.array([p.xi for p in points]);fn=np.array([p.fn_hz for p in points])
        mx=float(w@xi);mf=float(w@fn)
        rows.append(dict(case_id=case_id,status=cases[case_id].assessment.status,n_devices=len(points),included_mva=total,
                         weighted_xi=mx,weighted_fn_hz=mf,
                         xi_spread=float(np.sqrt(w@((xi-mx)**2))),fn_hz_spread=float(np.sqrt(w@((fn-mf)**2))),
                         low_xi_mva_fraction=float(w[xi<low_xi_threshold].sum()),
                         high_fn_mva_fraction=float(w[fn>=high_fn_threshold_hz].sum()),
                         joint_low_xi_high_fn_mva_fraction=float(w[(xi<low_xi_threshold)&(fn>=high_fn_threshold_hz)].sum())))
    return rows


def technology_features(analysis):
    cases=_validate(analysis);grouped=defaultdict(list)
    for p in analysis.points:grouped[(p.case_id,p.generator_name.split("_",1)[0])].append(p)
    rows=[]
    for (case_id,technology),points in sorted(grouped.items()):
        total=sum(p.nominal_power_mva for p in points)
        rows.append(dict(case_id=case_id,status=cases[case_id].assessment.status,technology=technology,
                         n_devices=len(points),included_mva=total,
                         weighted_xi=sum(p.xi*p.nominal_power_mva for p in points)/total,
                         weighted_fn_hz=sum(p.fn_hz*p.nominal_power_mva for p in points)/total))
    return rows


def population_ecdfs(analysis):
    """Within-run MVA weighting, then equal run weighting; no pooled-run counts."""
    cases=_validate(analysis);grouped=defaultdict(list)
    for p in analysis.points:grouped[p.case_id].append(p)
    curves=[]
    for status in STATUSES:
        ids=[case_id for case_id,case in cases.items() if case.assessment.status==status]
        for metric in ("xi","fn_hz"):
            values=[];weights=[]
            for case_id in ids:
                points=grouped[case_id];total=sum(p.nominal_power_mva for p in points)
                values.extend(getattr(p,metric) for p in points)
                weights.extend(p.nominal_power_mva/total/len(ids) for p in points)
            x,y=weighted_ecdf(values,weights)
            curves.extend(dict(status=status,metric=metric,value=float(a),cumulative_mva_fraction=float(b),n_cases=len(ids)) for a,b in zip(x,y))
    return curves


def _statuses(rows):
    return [s for s in STATUSES if s!="Not evaluated" or any(r["status"]==s for r in rows)]


def _ecdf_panel(axis,rows,key,xlabel,statuses,*,factor=1.):
    for status in statuses:
        values=[r[key]*factor for r in rows if r["status"]==status]
        x,y=weighted_ecdf(values)
        label=f"{LABELS[status]} (n={len(values)})"
        if len(x):axis.step(np.r_[x[0],x],np.r_[0,y*100],where="post",label=label,color=COLORS[status],lw=2)
        else:axis.plot([],[],label=label+"; sin casos",color=COLORS[status])
    axis.set(xlabel=xlabel,ylabel="Casos acumulados (%)",ylim=(0,102))
    axis.grid(alpha=.2);axis.legend(fontsize=8)


def _box_panel(axis,rows,key,ylabel,statuses,*,factor=1.):
    for pos,status in enumerate(statuses,1):
        values=np.array([r[key]*factor for r in rows if r["status"]==status])
        if len(values):
            box=axis.boxplot([values],positions=[pos],widths=.45,patch_artist=True,showfliers=False,manage_ticks=False)
            box["boxes"][0].set(facecolor=COLORS[status],alpha=.22)
            offsets=np.zeros(1) if len(values)==1 else np.random.default_rng(104+pos).uniform(-.14,.14,len(values))
            axis.scatter(pos+offsets,values,s=18,c=COLORS[status],alpha=.6,edgecolors="none",zorder=3)
        else:axis.text(pos,.5,"Sin casos",ha="center",transform=axis.get_xaxis_transform(),fontsize=8)
    axis.set_xticks(range(1,len(statuses)+1),[f"{LABELS[s]}\n(n={sum(r['status']==s for r in rows)})" for s in statuses])
    axis.set(xlim=(.5,len(statuses)+.5),ylabel=ylabel)
    axis.grid(axis="y",alpha=.2)


def export_distributions(analysis, output_directory, *, low_xi_threshold=.7, high_fn_threshold_hz=10.):
    """Save four descriptive PNGs and auditable CSVs without opening figures."""
    import matplotlib.pyplot as plt
    rows=case_features(analysis,low_xi_threshold,high_fn_threshold_hz)
    tech_rows=technology_features(analysis);curves=population_ecdfs(analysis)
    statuses=_statuses(rows)
    out=Path(output_directory)/"distributions_by_stability"
    out.mkdir(parents=True,exist_ok=True)
    _csv(out/"case_distribution_features.csv",rows,list(rows[0]))
    _csv(out/"technology_case_features.csv",tech_rows,list(tech_rows[0]))
    _csv(out/"population_ecdf.csv",curves,["status","metric","value","cumulative_mva_fraction","n_cases"])
    summary=[]
    numeric=[key for key in rows[0] if key not in {"case_id","status","n_devices","included_mva"}]
    for status in statuses:
        selected=[row for row in rows if row["status"]==status]
        for key in numeric:
            values=np.array([r[key] for r in selected])
            stats=dict(minimum=None,q25=None,median=None,q75=None,maximum=None,mean=None)
            if len(values):
                stats.update(zip(["minimum","q25","median","q75","maximum"],map(float,np.quantile(values,[0,.25,.5,.75,1]))))
                stats["mean"]=float(values.mean())
            summary.append(dict(status=status,feature=key,n_cases=len(values),**stats))
    _csv(out/"case_distribution_summary.csv",summary,list(summary[0]))
    paths=[]
    def save(figure,name):
        path=out/name
        try:figure.savefig(path,dpi=170)
        finally:plt.close(figure)
        paths.append(path)
    fig,axes=plt.subplots(2,2,figsize=(12,9),layout="constrained")
    for col,(key,label) in enumerate([("weighted_xi","xi ponderado por MVA"),("weighted_fn_hz","fn ponderada por MVA [Hz]")]):
        _ecdf_panel(axes[0,col],rows,key,label,statuses)
        _box_panel(axes[1,col],rows,key,label,statuses)
    fig.suptitle("Distribuciones por estabilidad dinamica: parametros medios de cada caso\nCada punto es una corrida; igual peso por caso",fontsize=13)
    save(fig,"01_case_parameter_distributions.png")
    fig,axes=plt.subplots(2,3,figsize=(16,9),layout="constrained")
    specs=[("low_xi_mva_fraction",f"MVA con xi < {low_xi_threshold:g} (%)"),
           ("high_fn_mva_fraction",f"MVA con fn >= {high_fn_threshold_hz:g} Hz (%)"),
           ("joint_low_xi_high_fn_mva_fraction",f"MVA con xi < {low_xi_threshold:g} y fn >= {high_fn_threshold_hz:g} Hz (%)")]
    for col,(key,label) in enumerate(specs):
        _ecdf_panel(axes[0,col],rows,key,label,statuses,factor=100.)
        _box_panel(axes[1,col],rows,key,label,statuses,factor=100.)
        axes[0,col].set_xlim(0,100);axes[1,col].set_ylim(0,100)
    fig.suptitle("Distribuciones por estabilidad: composicion de potencia de cada corrida\nUmbrales exploratorios; no definen la etiqueta de estabilidad",fontsize=13)
    save(fig,"02_capacity_share_distributions.png")
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout="constrained")
    for ax,metric,label in zip(axes,("xi","fn_hz"),("xi del dispositivo","fn del dispositivo [Hz]")):
        for status in statuses:
            curve=[r for r in curves if r["status"]==status and r["metric"]==metric]
            n=sum(r["status"]==status for r in rows)
            if curve:
                x=[r["value"] for r in curve];y=[r["cumulative_mva_fraction"]*100 for r in curve]
                ax.step([x[0],*x],[0.,*y],where="post",lw=2,color=COLORS[status],label=f"{LABELS[status]} (n={n} casos)")
            else:ax.plot([],[],color=COLORS[status],label=f"{LABELS[status]}: sin casos")
        ax.set(xlabel=label,ylabel="MVA acumulados: promedio entre casos (%)",ylim=(0,102))
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle("Distribucion de parametros de la poblacion incluida\nPonderacion por MVA dentro del caso; igual peso entre casos",fontsize=13)
    save(fig,"03_population_mva_distributions.png")
    technologies=sorted({r["technology"] for r in tech_rows})
    fig,axes=plt.subplots(2,len(technologies),figsize=(max(9,len(technologies)*3.7),8),squeeze=False,layout="constrained")
    for col,technology in enumerate(technologies):
        selected=[r for r in tech_rows if r["technology"]==technology]
        for row,(key,label) in enumerate([("weighted_xi","xi ponderado por MVA"),("weighted_fn_hz","fn ponderada por MVA [Hz]")]):
            _box_panel(axes[row,col],selected,key,label,statuses)
            axes[row,col].set_title(technology)
    fig.suptitle("Distribuciones por tecnologia y estabilidad dinamica\nCada punto es el promedio ponderado de esa tecnologia en una corrida",fontsize=13)
    save(fig,"04_technology_case_distributions.png")
    metadata=dict(criterion=analysis.criterion_description,include_angle=analysis.include_angle,counts={s:sum(r["status"]==s for r in rows) for s in STATUSES},
                  cases=len(rows),devices=len({p.device_id for p in analysis.points}),
                  case_weighting="One observation per run; equal run weight, including within each technology",
                  population_ecdf_weighting="For each class: each run has mass 1/n_runs; each active unit has its within-run MVA fraction of that mass",
                  technology_rule="Prefix of GeneratorName before first underscore",low_xi_threshold=low_xi_threshold,high_fn_threshold_hz=high_fn_threshold_hz,
                  thresholds_interpretation="Exploratory composition indicators, not operational stability limits",
                  interpretation="Descriptive class comparisons; no causal effects or significance claims. Unit observations within a case are not independent simulation cases.",
                  missing_classes="Retained as empty labeled groups, never replaced by zero-valued cases",figures=[p.name for p in paths])
    (out/"distribution_metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    (out/"README.md").write_text(f"# Distribuciones por estabilidad\n\n01: ECDF y cajas de xi/fn medios ponderados por MVA de cada corrida. En las cajas, cada punto es un caso; la caja muestra percentiles 25-75 y mediana, los bigotes siguen la regla de 1.5 IQR.\n\n02: ECDF y cajas del porcentaje MVA con xi < {low_xi_threshold:g}, fn >= {high_fn_threshold_hz:g} Hz y ambas condiciones. Son umbrales exploratorios y no cambian el criterio de estabilidad; los valores configurados se registran en distribution_metadata.json.\n\n03: ECDF de los valores de cada dispositivo. Cada caso pesa igual y dentro del caso se pondera por su fraccion de MVA. Es una distribucion descriptiva de capacidad, no miles de corridas independientes.\n\n04: promedios por caso separados por prefijo de tecnologia. Tecnologias ausentes no se reemplazan por cero.\n\nAzul: estable; naranja: inestable; gris: no evaluado si existe. Criterio: {analysis.criterion_description}. Los grupos se normalizan independientemente para comparar tamanos distintos. No se dibujan curvas suavizadas ni se calculan pruebas de significancia a partir de unidades repetidas.\n",encoding="utf-8")
    return tuple(paths)


def load_saved_analysis(results):
    """Read saved native values and recheck the current stability aggregation."""
    def read(name):
        with (Path(results)/name).open(encoding="utf-8-sig",newline="") as stream:return list(csv.DictReader(stream))
    points=[]
    for row in read("device_case_parameters.csv"):
        values={field.name:row[field.name] for field in fields(SpatialControllerPoint)}
        for key in ("x","y","nominal_power_mva","kp","ki","xi","fn_hz"):values[key]=float(values[key])
        values["run_id"]=int(values["run_id"]);points.append(SpatialControllerPoint(**values))
    metadata_path = Path(results)/"spatial_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    # Unmarked legacy exports retain their angular labels; interrupted Monte Carlo exports have blank angles.
    saved_cases = read("case_assessments.csv")
    include_angle = metadata.get("include_angle", any(row.get("angular_status") for row in saved_cases))
    if not isinstance(include_angle, bool):
        raise ValueError("Saved include_angle must be boolean")
    cases=[];skipped=[]
    mapped_ids={point.case_id for point in points}
    for row in saved_cases:
        if include_angle:
            if row.get("angular_status") not in {"Stable","Unstable","Not evaluated"}:
                raise ValueError("Saved labels must include the angular criterion")
            assessment=assess_dynamic_stability(int(row["run_id"]),row["voltage_status"],row["modal_damping_status"],row["angular_status"])
        else:
            if row.get("angular_status"):
                raise ValueError("Monte Carlo labels must not use the angular criterion")
            assessment=assess_monte_carlo_stability(int(row["run_id"]),row["voltage_status"],row["modal_damping_status"])
        if assessment.status!=row["status"]:raise ValueError("Saved labels differ from the current criterion; rerun spatial analysis")
        cases.append(SpatialCaseAssessment(row["case_id"],assessment,(row["diagnostics"],) if row.get("diagnostics") else ()))
        if row["case_id"] not in mapped_ids:
            if row.get("spatial_status")!="Skipped":
                raise ValueError("Saved cases without controllers must be explicitly marked Skipped")
            skipped.append(SpatialSkippedRun(row["case_id"],row.get("spatial_reason") or
                                             row.get("diagnostics") or "No mapped controllers"))
    analysis=SpatialStabilityAnalysis(tuple(points),tuple(cases),skipped_runs=tuple(skipped),
                                     voltage_scope=metadata.get("voltage_scope", "All available 220/500 kV buses"))
    _validate(analysis)
    return analysis


def main():
    import matplotlib
    matplotlib.use("Agg")
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results",type=Path,required=True,help="Existing 04_spatial_stability output directory")
    parser.add_argument("--low-xi-threshold",type=float,default=.7)
    parser.add_argument("--high-fn-threshold-hz",type=float,default=10.)
    args=parser.parse_args()
    analysis=load_saved_analysis(args.results)
    paths=export_distributions(analysis,args.results,low_xi_threshold=args.low_xi_threshold,high_fn_threshold_hz=args.high_fn_threshold_hz)
    print(f"Created {len(paths)} distribution figures in {args.results/'distributions_by_stability'}")


if __name__=="__main__":main()
