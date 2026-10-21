"""Excel reports produced from completed analysis results."""

from dataclasses import dataclass
from pathlib import Path

from analyzer import RunSignalAnalysis, SignalStudyAnalysis


@dataclass(frozen=True)
class AnalysisReport:
    """Write the current per-bus analysis result into the study-wide report."""

    bus_name: str
    voltage_analysis: SignalStudyAnalysis

    def write(self, output_directory: str | Path) -> Path:
        """Write the report at the root of the study analysis outputs."""
        from openpyxl import Workbook

        output_path = Path(output_directory) / "analysis_report.xlsx"
        workbook = Workbook()
        initialization_sheet = workbook.active
        initialization_sheet.title = "Initialization"
        stability_sheet = workbook.create_sheet("Stability")
        criteria_sheet = workbook.create_sheet("Criteria")

        self._write_initialization(initialization_sheet)
        self._write_stability(stability_sheet)
        self._write_criteria(criteria_sheet)
        workbook.save(output_path)
        workbook.close()
        return output_path

    def _write_initialization(self, sheet) -> None:
        criteria = self._initialization_criteria()
        sheet.append(
            (
                "Bus",
                "Signal",
                "Run",
                "Status",
                "P99 Fast Error [%]",
                "Limit [%]",
                "Reason",
                "Reference Voltage [pu]",
            )
        )
        for result in self.voltage_analysis.run_results:
            initialization = result.initialization
            sheet.append(
                (
                    self.bus_name,
                    self.voltage_analysis.signal_name,
                    self._run_name(result.run_id),
                    (
                        "Well initialized"
                        if initialization.is_well_initialized
                        else "Poorly initialized"
                    ),
                    initialization.fast_relative_error_percentile_99 * 100,
                    criteria.maximum_fast_relative_error_percentile_99 * 100,
                    ", ".join(initialization.rejection_reasons) or "-",
                    initialization.reference_value,
                )
            )

        self._format_sheet(sheet)

    def _write_stability(self, sheet) -> None:
        sheet.append(
            (
                "Bus",
                "Signal",
                "Run",
                "Status",
                "Reason",
                "Post-clearance Area [s]",
                "Area Limit [s]",
                "Final Error [%]",
                "Final Error Limit [%]",
                "Final Range [%]",
                "Final Range Limit [%]",
                "Minimum Voltage [pu]",
                "Minimum Voltage Limit [pu]",
            )
        )
        for result in self.voltage_analysis.run_results:
            stability = result.stability
            if stability is None:
                status = "Not evaluated"
                reason = (
                    "Initialization failed"
                    if not result.initialization.is_well_initialized
                    else "Simulation duration is insufficient"
                )
                sheet.append(
                    (
                        self.bus_name,
                        self.voltage_analysis.signal_name,
                        self._run_name(result.run_id),
                        status,
                        reason,
                    )
                )
                continue

            is_stable = (
                stability.remains_above_voltage_threshold_after_evaluation
                and stability.is_settled
            )
            sheet.append(
                (
                    self.bus_name,
                    self.voltage_analysis.signal_name,
                    self._run_name(result.run_id),
                    "Stable" if is_stable else "Unstable",
                    ", ".join(self._stability_failures(result)) if not is_stable else "-",
                    stability.relative_error_area_s,
                    stability.criteria.maximum_relative_error_area_s,
                    stability.final_settlement_peak_absolute_relative_error * 100,
                    stability.criteria.maximum_final_absolute_relative_error * 100,
                    stability.final_settlement_peak_to_peak_relative_range * 100,
                    stability.criteria.maximum_final_peak_to_peak_relative_range * 100,
                    stability.minimum_voltage_after_evaluation_pu,
                    stability.criteria.post_evaluation_voltage_threshold_pu,
                )
            )

        self._format_sheet(sheet)

    def _write_criteria(self, sheet) -> None:
        initialization = self._initialization_criteria()
        stability = self._stability_criteria()
        sheet.append(("Analysis", "Criterion", "Value"))
        rows = (
            ("Initialization", "Evaluation window [s]", f"{initialization.evaluation_start_time_s:.2f} to {initialization.evaluation_end_time_s:.2f}"),
            ("Initialization", "Smoothing window [s]", initialization.smoothing_duration_s),
            ("Initialization", "Maximum P99 fast error [%]", initialization.maximum_fast_relative_error_percentile_99 * 100),
            ("Stability", "Minimum simulation end time [s]", stability.minimum_simulation_end_time_s),
            ("Stability", "Post-clearance area limit [s]", stability.maximum_relative_error_area_s),
            ("Stability", "Final absolute error limit [%]", stability.maximum_final_absolute_relative_error * 100),
            ("Stability", "Final range limit [%]", stability.maximum_final_peak_to_peak_relative_range * 100),
            ("Stability", "Minimum post-evaluation voltage [pu]", stability.post_evaluation_voltage_threshold_pu),
        )
        for row in rows:
            sheet.append(row)

        self._format_sheet(sheet)

    def _initialization_criteria(self):
        if not self.voltage_analysis.run_results:
            from signal_analysis import InitializationCriteria

            return InitializationCriteria()
        return self.voltage_analysis.run_results[0].initialization.criteria

    def _stability_criteria(self):
        for result in self.voltage_analysis.run_results:
            if result.stability is not None:
                return result.stability.criteria
        from signal_analysis import StabilityCriteria

        return StabilityCriteria()

    @staticmethod
    def _run_name(run_id: int) -> str:
        return f"run_{run_id:05d}"

    @staticmethod
    def _format_sheet(sheet) -> None:
        from openpyxl.styles import Font

        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions
        for cell in sheet[1]:
            cell.font = Font(bold=True)
        for column in sheet.columns:
            for cell in column[1:]:
                if isinstance(cell.value, float):
                    cell.number_format = "0.###"
            width = max(len(str(cell.value or "")) for cell in column) + 2
            sheet.column_dimensions[column[0].column_letter].width = min(width, 45)

    @staticmethod
    def _stability_failures(result: RunSignalAnalysis) -> tuple[str, ...]:
        if result.stability is None:
            return ("not evaluated",)

        stability = result.stability
        failures = []
        if not stability.is_area_within_limit:
            failures.append("Post-clearance area")
        if (
            stability.final_settlement_peak_absolute_relative_error
            > stability.criteria.maximum_final_absolute_relative_error
        ):
            failures.append("Final absolute error")
        if (
            stability.final_settlement_peak_to_peak_relative_range
            > stability.criteria.maximum_final_peak_to_peak_relative_range
        ):
            failures.append("Final peak-to-peak range")
        if not stability.remains_above_voltage_threshold_after_evaluation:
            failures.append("Minimum post-evaluation voltage")
        return tuple(failures) or ("Combined settlement criterion",)
