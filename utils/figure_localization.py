"""Create a Spanish presentation copy of an already assembled Matplotlib figure."""

from __future__ import annotations

from pathlib import Path


# Longest and most specific phrases must be considered first.  The plotting
# code remains English; this presentation-only pass avoids duplicating each
# analysis figure and guarantees that both images carry the same evidence.
_SPANISH_REPLACEMENTS = (
    ("500 kV transmission-system active-power transfers", "Transferencias de potencia activa del Sistema de Transmisión 500 kV"),
    ("Lowest Measured Electromechanical-Mode Damping by 500 kV Transmission System", "Menor amortiguamiento medido de modo electromecánico por Sistema de Transmisión 500 kV"),
    ("Isolated EM Components for Worst Active-Power Transfer Modal-Damping Cases", "Componentes electromecánicas aisladas para los casos de peor amortiguamiento modal de transferencia de potencia activa"),
    ("Electromechanical-Mode Decomposition for Worst Active-Power Transfer Cases", "Descomposición de modos electromecánicos para los peores casos de transferencia de potencia activa"),
    ("High-Frequency Components for Worst Active-Power Transfer Cases", "Componentes de alta frecuencia para los peores casos de transferencia de potencia activa"),
    ("High-Frequency Component Decomposition for Worst Active-Power Transfer Cases", "Descomposición de componentes de alta frecuencia para los peores casos de transferencia de potencia activa"),
    ("Dominant Electromechanical-Mode Damping", "Amortiguamiento del modo electromecánico dominante"),
    ("All Electromechanical-Mode Damping", "Amortiguamiento de todos los modos electromecánicos"),
    ("Power-weighted natural frequency", "Frecuencia natural ponderada por potencia"),
    ("Power-weighted damping ratio", "Razón de amortiguamiento ponderada por potencia"),
    ("Fault-on absolute-voltage area", "Área absoluta de tensión durante la falla"),
    ("Recovery absolute-voltage area", "Área absoluta de tensión de recuperación"),
    ("Unified response absolute-voltage area", "Área absoluta de tensión de respuesta unificada"),
    ("0.70 pu decision time after clearance", "Tiempo de decisión a 0,70 pu después del despeje"),
    ("Cumulative time below 0.80 pu", "Tiempo acumulado bajo 0,80 pu"),
    ("Damped (xi at least 5%)", "Amortiguado (xi de al menos 5%)"),
    ("Growing or damped under", "Creciente o subamortiguado"),
    ("Not measurable / not applicable", "No medible / no aplicable"),
    ("excluded from xi distribution: opened, inactive, or not measurable.", "excluidos de la distribución de xi: abiertos, inactivos o no medibles."),
    ("See Transfer Assessments in the report.", "Ver Evaluaciones de Transferencias en el reporte."),
    ("EM mode frequency", "Frecuencia de modo electromecánico"),
    ("Linear fit", "Ajuste lineal"),
    ("Design range", "Rango de diseño"),
    ("Bandwidth range", "Rango de ancho de banda"),
    ("0.70 pu deadline", "Plazo a 0,70 pu"),
    ("0.80 pu limit", "Límite a 0,80 pu"),
    ("Pass 0.70 pu recovery", "Cumple recuperación a 0,70 pu"),
    ("Fail 0.70 pu recovery", "No cumple recuperación a 0,70 pu"),
    ("Pass all voltage criteria", "Cumple todos los criterios de tensión"),
    ("Fail one or more voltage criteria", "No cumple uno o más criterios de tensión"),
    ("Voltage recovery window", "Ventana de recuperación de tensión"),
    ("50 ms after clearance", "50 ms después del despeje"),
    ("recovery and final settlement", "recuperación y establecimiento final"),
    ("Fast-oscillation P99", "P99 de oscilación rápida"),
    ("reference voltage", "tensión de referencia"),
    ("acceptance limit", "límite de aceptación"),
    ("(best)", "(mejor)"),
    ("(limit)", "(límite)"),
    ("(worst)", "(peor)"),
    ("Controller point", "Punto de controlador"),
    ("transfer(s)", "transferencia(s)"),
    ("run assessment(s)", "evaluación(es) de caso"),
    ("modes", "modos"),
    ("mode", "modo"),
    ("â€”", "—"),
    ("HÂ·S", "H·S"),
    ("Â±", "±"),
    ("Maximum Angular Excursion by Synchronous Machine", "Máxima excursión angular por máquina sincrónica"),
    ("Machine Excursion Envelope Relative to the Centre of Inertia", "Envolvente de excursión de máquinas respecto al centro de inercia"),
    ("Angular excursion relative to COI", "Excursión angular respecto al COI"),
    ("Synchronous machine", "Máquina sincrónica"),
    ("Within ±120°", "Dentro de ±120°"),
    ("At or beyond ±120°", "En o fuera de ±120°"),
    ("Extreme with maximum |Δδ|", "Extremo con mayor |Δδ|"),
    ("Each horizontal segment spans the machine minimum and maximum COI-referenced excursion; the marker identifies the signed extreme with the greatest absolute value.", "Cada segmento horizontal abarca la excursión mínima y máxima de la máquina respecto al COI; el marcador identifica el extremo con signo de mayor valor absoluto."),
    ("Machine Angular Excursions Relative to the Centre of Inertia — Full Simulation", "Excursiones angulares de máquinas respecto al centro de inercia — Simulación completa"),
    ("Displayed through the available simulation tmax; the ±120° lines remain the angular-separation limits.", "Mostrado hasta el tmax disponible de la simulación; las líneas ±120° mantienen los límites de separación angular."),
    ("H·S-Weighted Centre-of-Inertia Angle", "Ángulo del centro de inercia ponderado por H·S"),
    ("Machine Angular Excursions Relative to the Centre of Inertia", "Excursiones angulares de máquinas respecto al centro de inercia"),
    ("Angular Stability", "Estabilidad angular"),
    ("COI angle", "Ángulo del COI"),
    ("Angular excursion", "Excursión angular"),
    ("RMS-Voltage Initialization Assessment", "Evaluación de inicialización de tensión RMS"),
    ("No RMS-voltage signals", "Sin señales de tensión RMS"),
    ("Well initialized", "Bien inicializado"),
    ("Initialization warning", "Advertencia de inicialización"),
    ("Evaluation window", "Ventana de evaluación"),
    ("High-Frequency Transfer Spectrum — Observation Only", "Espectro de transferencia de alta frecuencia — Solo observación"),
    ("Detected components are warnings only and do not alter the dynamic-stability result.", "Los componentes detectados son solo advertencias y no modifican el resultado de estabilidad dinámica."),
    ("No observable spectrum", "Sin espectro observable"),
    ("Active-Power Transfers", "Transferencias de potencia activa"),
    ("Electromechanical Transfer Response", "Respuesta electromecánica de transferencias"),
    ("Electromechanical Mode Decomposition", "Descomposición de modos electromecánicos"),
    ("No evaluable EM response", "Sin respuesta electromecánica evaluable"),
    ("No measurable electromechanical modes", "Sin modos electromecánicos medibles"),
    ("Active power (MW)", "Potencia activa (MW)"),
    ("Reactive power (MVAr)", "Potencia reactiva (MVAr)"),
    ("Electrical power (pu)", "Potencia eléctrica (pu)"),
    ("Mechanical power (pu)", "Potencia mecánica (pu)"),
    ("Rotor speed (pu)", "Velocidad de rotor (pu)"),
    ("Terminal Voltage", "Tensión terminal"),
    ("RMS Voltage", "Tensión RMS"),
    ("Reactive Power", "Potencia reactiva"),
    ("Electrical Power", "Potencia eléctrica"),
    ("Mechanical Power", "Potencia mecánica"),
    ("Active Power", "Potencia activa"),
    ("Rotor Speed", "Velocidad de rotor"),
    ("SM Electrical and Mechanical Power", "Potencia eléctrica y mecánica de máquinas sincrónicas"),
    ("SM Terminal Voltage", "Tensión terminal de máquinas sincrónicas"),
    ("SM Dynamics", "Dinámica de máquinas sincrónicas"),
    ("SM Power", "Potencia de máquinas sincrónicas"),
    ("500 kV Buses", "Barras de 500 kV"),
    ("220 kV Buses", "Barras de 220 kV"),
    ("PV Power", "Potencia fotovoltaica"),
    ("WF Power", "Potencia eólica"),
    ("BESS Power", "Potencia BESS"),
    ("DER Power", "Potencia DER"),
    ("STATCOM / SVC Power", "Potencia STATCOM / SVC"),
    ("Plant Recovery — Terminal Exceptions", "Recuperación de centrales — Excepciones terminales"),
    ("Pre-fault reference", "Referencia prefalla"),
    ("Recovery evaluation starts", "Inicio de evaluación de recuperación"),
    ("Complete Simulation", "Simulación completa"),
    ("Fault and Clearance", "Falla y despeje"),
    ("Contingency Close-up", "Ampliación de contingencia"),
    ("Voltage Stability", "Estabilidad de tensión"),
    ("500 kV buses", "Barras de 500 kV"),
    ("Unified voltage-response area", "Área unificada de respuesta de tensión"),
    ("Scale:", "Escala:"),
    ("observed:", "observado:"),
    ("P(t) âˆ’ P_final", "P(t) − P_final"),
    ("P(t) − P_final", "P(t) − P_final"),
    ("Controller population across all active machines", "Población de controladores de todas las máquinas activas"),
    ("Nominal-power-weighted controller point by run", "Punto de controlador ponderado por potencia nominal por caso"),
    ("global dynamic-stability status", "estado global de estabilidad dinámica"),
    ("consolidated selected-analysis status", "estado consolidado del análisis seleccionado"),
    ("Well-initialized runs", "Casos bien inicializados"),
    ("Poorly initialized runs", "Casos mal inicializados"),
    ("Pass all active voltage criteria", "Cumple todos los criterios activos de tensión"),
    ("Fail final recovery above 0.70 pu", "No cumple recuperación final sobre 0,70 pu"),
    ("Fail cumulative time below 0.80 pu", "No cumple tiempo acumulado bajo 0,80 pu"),
    ("Fail final-settlement criterion", "No cumple criterio de establecimiento final"),
    ("Contingency / recovery", "Contingencia / recuperación"),
    ("Centre-of-Inertia Angle Across Monte Carlo Runs", "Ángulo del centro de inercia en casos Monte Carlo"),
    ("Critical COI-Referenced Angular Excursion by Run", "Excursión angular crítica respecto al COI por caso"),
    ("Critical Machine for Angularly Unstable Runs", "Máquina crítica de casos angularmente inestables"),
    ("posX (EMTP drawing units)", "posX (unidades de diagrama EMTP)"),
    ("posY (EMTP drawing units)", "posY (unidades de diagrama EMTP)"),
    ("Per-unit mean", "Media por unidad"),
    ("available active cases", "casos activos disponibles"),
    ("Color =", "Color ="),
    ("Area =", "Área ="),
    ("Mean PLL damping ratio by dynamic-stability class", "Razón media de amortiguamiento PLL por clase de estabilidad dinámica"),
    ("Mean PLL natural-frequency band by dynamic-stability class", "Banda media de frecuencia natural PLL por clase de estabilidad dinámica"),
    ("Per-unit mean across available active cases; common position, size and colour scales", "Media por unidad entre los casos activos disponibles; escalas comunes de posición, tamaño y color"),
    ("No high-frequency observation is available.", "No hay observación de alta frecuencia disponible."),
    ("No observable high-frequency spectrum.", "No hay espectro observable de alta frecuencia."),
    ("No detected high-frequency components.", "No hay componentes de alta frecuencia detectadas."),
    ("Failure panels are non-exclusive: a run can appear in more than one.", "Los paneles de incumplimiento no son excluyentes: un caso puede aparecer en más de uno."),
    ("Each point is the largest absolute machine excursion in that run. A value at or above 120 deg is angularly unstable.", "Cada punto es la mayor excursión absoluta de una máquina en ese caso. Un valor igual o superior a 120 grados es angularmente inestable."),
    ("Each point is one unstable run, grouped by its critical machine.", "Cada punto es un caso inestable, agrupado por su máquina crítica."),
    ("No angularly unstable runs.", "No hay casos angularmente inestables."),
    ("Fast-oscillation rejections", "Rechazos por oscilación rápida"),
    ("initialization assessment", "evaluación de inicialización"),
    ("fault and clearance close-up", "ampliación de falla y despeje"),
    ("contingency close-up", "ampliación de contingencia"),
    ("grouped by voltage stability", "agrupada según estabilidad de tensión"),
    ("fault-on area", "área durante la falla"),
    ("recovery area", "área de recuperación"),
    ("unified response area", "área de respuesta unificada"),
    ("cumulative 0.80 pu duration", "duración acumulada bajo 0,80 pu"),
    ("0.70 pu stability decision", "decisión de estabilidad a 0,70 pu"),
    ("did not complete final recovery", "no completó la recuperación final"),
    ("vs. weighted parameters", "frente a parámetros ponderados"),
    ("Pointwise median", "Mediana puntual"),
    ("MVA-weighted included population", "Población incluida ponderada por MVA"),
    ("Dynamic stability", "Estabilidad dinámica"),
    ("dynamic-stability class", "clase de estabilidad dinámica"),
    ("common position, size and colour scales", "escalas comunes de posición, tamaño y color"),
    ("Monte Carlo runs", "Casos Monte Carlo"),
    ("Monte Carlo run", "Caso Monte Carlo"),
    ("Maximum absolute angular excursion", "Máxima excursión angular absoluta"),
    ("Centre-of-Inertia", "Centro de Inercia"),
    ("Center-of-Inertia", "Centro de Inercia"),
    ("COI-referenced", "referida al COI"),
    ("Power Transfers", "Transferencias de Potencia"),
    ("Power Transfer", "Transferencia de Potencia"),
    ("Active power", "Potencia activa"),
    ("Spectral power", "Potencia espectral"),
    ("Natural frequency", "Frecuencia natural"),
    ("natural frequency", "frecuencia natural"),
    ("Damping ratio", "Razón de amortiguamiento"),
    ("damping ratio", "razón de amortiguamiento"),
    ("PLL damping ratio", "Razón de amortiguamiento PLL"),
    ("PLL natural-frequency band", "Banda de frecuencia natural PLL"),
    ("High-Frequency", "Alta frecuencia"),
    ("high-frequency", "alta frecuencia"),
    ("High frequency", "Alta frecuencia"),
    ("high frequency", "alta frecuencia"),
    ("Electromechanical", "Electromecánico"),
    ("electromechanical", "electromecánico"),
    ("Transmission System", "Sistema de Transmisión"),
    ("transmission system", "sistema de transmisión"),
    ("Angular limit", "Límite angular"),
    ("angularly unstable", "angularmente inestable"),
    ("Angularly unstable", "Angularmente inestable"),
    ("Well-initialized", "Bien inicializado"),
    ("Initialization-error", "Error de inicialización"),
    ("Initialization", "Inicialización"),
    ("initialization", "inicialización"),
    ("Recovery window", "Ventana de recuperación"),
    ("recovery window", "ventana de recuperación"),
    ("Contingency", "Contingencia"),
    ("contingency", "contingencia"),
    ("Frequency", "Frecuencia"),
    ("frequency", "frecuencia"),
    ("Voltage", "Tensión"),
    ("voltage", "tensión"),
    ("Stable", "Estable"),
    ("Unstable", "Inestable"),
    ("Not evaluated", "No evaluado"),
    ("Not measurable", "No medible"),
    ("Damped under", "Subamortiguado"),
    ("Damped", "Amortiguado"),
    ("Growing", "Creciente"),
    ("Fault", "Falla"),
    ("fault", "falla"),
    ("Clearance", "Despeje"),
    ("clearance", "despeje"),
    ("Time (s)", "Tiempo (s)"),
    ("runs", "casos"),
    ("Runs", "Casos"),
    ("run", "caso"),
    ("Run", "Caso"),
    ("Mode", "Modo"),
    ("mode", "modo"),
    ("System", "Sistema"),
    ("system", "sistema"),
    ("Area proportional to MVA", "Área proporcional a MVA"),
    ("Common scales across", "Escalas comunes entre"),
    ("Each point is", "Cada punto es"),
    ("No cases", "Sin casos"),
    ("NO CASES IN THIS CLASS", "SIN CASOS EN ESTA CLASE"),
    ("No observable", "Sin espectro observable"),
    ("No detected", "Sin componentes detectadas"),
)


def spanish_copy_path(path: str | Path) -> Path:
    """Return the sibling PNG path identified as the Spanish presentation."""
    source = Path(path)
    return source.with_name(f"{source.stem}_ES{source.suffix}")


def remove_figure_with_spanish_copy(path: str | Path) -> None:
    """Remove a generated figure together with its localized sibling."""
    output_path = Path(path)
    output_path.unlink(missing_ok=True)
    spanish_copy_path(output_path).unlink(missing_ok=True)


def _translate(text: str) -> str:
    translated = text
    for source, target in sorted(
        _SPANISH_REPLACEMENTS,
        key=lambda replacement: len(replacement[0]),
        reverse=True,
    ):
        translated = translated.replace(source, target)
    return translated


def save_figure_with_spanish_copy(figure, path: str | Path, *, dpi: int) -> tuple[Path, Path]:
    """Save the original image and a localized sibling without changing the figure.

    Only Matplotlib text artists are temporarily translated.  Numerical data,
    traces, colours, limits, names of electrical assets and the original PNG
    remain untouched.
    """
    from matplotlib.text import Text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.unlink(missing_ok=True)
    spanish_path = spanish_copy_path(output_path)
    spanish_path.unlink(missing_ok=True)
    figure.savefig(output_path, dpi=dpi)

    originals: list[tuple[Text, str]] = []
    for artist in figure.findobj(match=lambda item: isinstance(item, Text)):
        original = artist.get_text()
        translated = _translate(original)
        if translated != original:
            originals.append((artist, original))
            artist.set_text(translated)
    try:
        figure.savefig(spanish_path, dpi=dpi)
    finally:
        for artist, original in originals:
            artist.set_text(original)
    return output_path, spanish_path
