"""Excel reports produced from completed analysis results."""

from dataclasses import dataclass
from pathlib import Path

from analyzer import SignalStudyAnalysis, StudyAnalysis
from scenario import ContingencyScenario, EvaluationProfile
from transfer_analysis import TransferStudyAnalysis


@dataclass(frozen=True)
class TransferDampingRunSummary:
    """One run's consolidated active-transfer damping conclusion."""

    status: str
    findings: tuple[str, ...]


@dataclass(frozen=True)
class AnalysisReport:
    """Write selected bus analyses and their consolidated run result."""

    study_analysis: StudyAnalysis
    transfer_analysis: TransferStudyAnalysis | None = None
    scenario: ContingencyScenario | None = None
    profile: EvaluationProfile | None = None

    def write(self, output_directory: str | Path) -> Path:
        """Write the report at the root of the study analysis outputs."""
        from openpyxl import Workbook

        output_path = Path(output_directory) / "analysis_report.xlsx"
        workbook = Workbook()
        summary_sheet = workbook.active
        summary_sheet.title = "Summary"
        initialization_sheet = workbook.create_sheet("Initialization")
        stability_sheet = workbook.create_sheet("Stability")
        criteria_sheet = workbook.create_sheet("Criteria")
        if self.scenario is not None:
            scenario_sheet = workbook.create_sheet("Scenario")
        if self.transfer_analysis is not None:
            transfer_summary_sheet = workbook.create_sheet("Transfer Summary")
            transfer_damping_sheet = workbook.create_sheet("Transfer Damping")
            damping_pairs_sheet = workbook.create_sheet("Candidate Damping Pairs")

        self._write_summary(summary_sheet)
        self._write_initialization(initialization_sheet)
        self._write_stability(stability_sheet)
        self._write_criteria(criteria_sheet)
        if self.scenario is not None:
            self._write_scenario(scenario_sheet)
        if self.transfer_analysis is not None:
            self._write_transfer_summary(transfer_summary_sheet)
            self._write_transfer_damping(transfer_damping_sheet)
            self._write_damping_pairs(damping_pairs_sheet)
        workbook.save(output_path)
        workbook.close()
        return output_path

    def _write_scenario(self, sheet) -> None:
        """Document the physical contingency and operating state under evaluation."""
        sheet.append(("Field", "Value"))
        values = (
            ("Name", self.scenario.name),
            ("Severity", self.scenario.severity),
            ("Fault type", self.scenario.fault_type),
            ("Fault time [s]", self.scenario.fault_time_s),
            ("Clearance time [s]", self.scenario.clearance_time_s),
            ("Location", self.scenario.location or "-"),
            ("Operating state", self.scenario.operating_state),
        )
        for value in values:
            sheet.append(value)
        self._format_sheet(sheet)

    def _write_summary(self, sheet) -> None:
        """Write separate voltage and transfer-damping conclusions per run."""
        sheet.append(
            (
                "Run",
                "Overall Assessment",
                "Voltage Stability",
                "Selected Buses",
                "Voltage Findings",
                "Transfer Damping",
                "Transfer Damping Findings",
            )
        )
        selected_buses = ", ".join(
            bus_analysis.bus_name
            for bus_analysis in self.study_analysis.bus_analyses
        )
        for run_id in self.study_analysis.run_ids:
            voltage_status = self.study_analysis.status_for_run(run_id)
            voltage_findings = self.study_analysis.reasons_for_run(run_id)
            damping = self._transfer_damping_summary_for_run(run_id)
            sheet.append(
                (
                    self._run_name(run_id),
                    self._overall_assessment(voltage_status, damping.status),
                    voltage_status,
                    selected_buses,
                    "; ".join(voltage_findings) or "-",
                    damping.status,
                    "; ".join(damping.findings) or "-",
                )
            )

        self._format_sheet(sheet)

    def _write_transfer_summary(self, sheet) -> None:
        sheet.append(
            (
                "Transfer", "Compliant Runs", "Under-damped Runs", "Growing-mode Runs", "Not Evaluated Runs",
                "Minimum Selected Damping [%]", "P05 Damping [%]",
            )
        )
        import numpy as np

        for transfer_name in self.transfer_analysis.transfer_names:
            assessments = self.transfer_analysis.assessments_for_transfer(transfer_name)
            active_assessments = [
                item
                for item in assessments
                if item.status not in {"opened", "inactive_pre_fault"}
            ]
            if not active_assessments:
                continue
            damping = [
                item.damping_ratio
                for item in active_assessments
                if item.damping_ratio is not None
            ]
            compliant = [item for item in active_assessments if item.status == "compliant"]
            under_damped = [item for item in active_assessments if item.status == "under_damped"]
            growing_mode = [item for item in active_assessments if item.status == "growing_mode"]
            not_evaluable = [
                item for item in active_assessments if item.status == "not_evaluated"
            ]
            sheet.append(
                (
                    transfer_name,
                    len(compliant),
                    len(under_damped),
                    len(growing_mode),
                    len(not_evaluable),
                    min(damping) * 100 if damping else None,
                    float(np.quantile(damping, 0.05)) * 100 if damping else None,
                )
            )
        self._format_transfer_sheet(sheet)

    def _write_transfer_damping(self, sheet) -> None:
        sheet.append(
            (
                "Run", "Transfer", "Pre-fault P", "Permanent P", "Status", "Warning", "Comment",
                "Dominant Frequency [Hz]", "Damping [%]", "Selected Polarity",
                "Peak 1 Time [s]", "Peak 2 Time [s]", "Period [s]",
                "P1", "P2", "A1", "A2", "A2 / A1",
            )
        )
        for item in self.transfer_analysis.assessments:
            pair = item.selected_damping_pair
            direction = 1.0 if pair and pair.polarity == "positive" else -1.0
            first_peak_power = (
                item.final_power + direction * pair.first_amplitude if pair else None
            )
            second_peak_power = (
                item.final_power + direction * pair.second_amplitude if pair else None
            )
            sheet.append(
                (
                    self._run_name(item.run_id), item.transfer_name, item.pre_fault_power,
                    item.final_power, item.classification, item.warning, item.comment, item.dominant_frequency_hz,
                    item.damping_ratio * 100 if item.damping_ratio is not None else None,
                    pair.polarity if pair else None, pair.first_time_s if pair else None,
                    pair.second_time_s if pair else None, pair.period_s if pair else None,
                    first_peak_power, second_peak_power,
                    pair.first_amplitude if pair else None, pair.second_amplitude if pair else None,
                    pair.second_amplitude / pair.first_amplitude if pair else None,
                )
            )
        self._format_transfer_sheet(sheet)

    def _write_damping_pairs(self, sheet) -> None:
        sheet.append(("Run", "Transfer", "Selected", "Polarity", "Peak 1 Time [s]", "Peak 2 Time [s]", "A1", "A2", "Period [s]", "Damping [%]"))
        for item in self.transfer_analysis.assessments:
            for pair in item.damping_pairs:
                sheet.append((
                    self._run_name(item.run_id),
                    item.transfer_name,
                    pair == item.selected_damping_pair,
                    pair.polarity,
                    pair.first_time_s,
                    pair.second_time_s,
                    pair.first_amplitude,
                    pair.second_amplitude,
                    pair.period_s,
                    pair.damping_ratio * 100,
                ))
        self._format_transfer_sheet(sheet)

    def _transfer_damping_summary_for_run(self, run_id: int) -> TransferDampingRunSummary:
        """Return only actionable active-transfer damping findings for one run."""
        if self.transfer_analysis is None:
            return TransferDampingRunSummary("Not evaluated", ())
        assessments = [
            item
            for item in self.transfer_analysis.assessments
            if item.run_id == run_id
            and item.status not in {"opened", "inactive_pre_fault"}
        ]
        if not assessments:
            return TransferDampingRunSummary("Not applicable", ())
        growing_mode = [item for item in assessments if item.status == "growing_mode"]
        under_damped = [item for item in assessments if item.status == "under_damped"]
        not_evaluable = [item for item in assessments if item.status == "not_evaluated"]
        findings = [
            "Growing mode: "
            + ", ".join(
                f"{item.transfer_name} ({item.damping_ratio * 100:.2f}%)"
                for item in growing_mode
            )
        ] if growing_mode else []
        if under_damped:
            findings.append(
                "Under-damped: "
                + ", ".join(
                    f"{item.transfer_name} ({item.damping_ratio * 100:.2f}%)"
                    for item in under_damped
                )
            )
        if not_evaluable:
            findings.append(
                "Not evaluated: "
                + ", ".join(item.transfer_name for item in not_evaluable)
            )
        if growing_mode:
            status = "Growing mode"
        elif not_evaluable:
            status = "Not evaluated"
        elif under_damped:
            status = "Under-damped"
        else:
            status = "Compliant"
        return TransferDampingRunSummary(status, tuple(findings))

    @staticmethod
    def _overall_assessment(voltage_status: str, damping_status: str) -> str:
        """Combine separate voltage and damping conclusions without hiding either."""
        if voltage_status == "Stable" and damping_status == "Compliant":
            return "Fully compliant"
        if voltage_status == "Stable" and damping_status == "Under-damped":
            return "Stable with under-damping"
        if voltage_status == "Unstable" or damping_status == "Growing mode":
            return "Non-compliant"
        return "Not evaluated"

    def _write_initialization(self, sheet) -> None:
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
        for bus_analysis in self.study_analysis.bus_analyses:
            voltage_analysis = bus_analysis.signal_analysis
            criteria = self._initialization_criteria(voltage_analysis)
            for result in voltage_analysis.run_results:
                initialization = result.initialization
                sheet.append(
                    (
                        bus_analysis.bus_name,
                        voltage_analysis.signal_name,
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
                "Failed Criteria",
                "0.70 pu Decision Time [s]",
                "0.70 pu Decision After Clearance [s]",
                "0.70 pu Decision Event",
                "0.70 pu Deadline After Clearance [s]",
                "0.70 pu Status",
                "Cumulative V < 0.80 pu [s]",
                "0.80 pu Duration Limit [s]",
                "0.80 pu Status",
                "Final Settlement Status",
                "Fault-on Area [pu*s]",
                "Recovery Area [pu*s]",
                "Unified Response Area [pu*s]",
                "Final Error [%]",
                "Final Error Limit [%]",
                "Final Range [%]",
                "Final Range Limit [%]",
            )
        )
        for bus_analysis in self.study_analysis.bus_analyses:
            voltage_analysis = bus_analysis.signal_analysis
            for result in voltage_analysis.run_results:
                stability = result.stability
                if stability is None:
                    reason = (
                        "Initialization failed"
                        if not result.initialization.is_well_initialized
                        else "Simulation duration is insufficient"
                    )
                    sheet.append(
                        (
                            bus_analysis.bus_name,
                            voltage_analysis.signal_name,
                            self._run_name(result.run_id),
                            "Not evaluated",
                            reason,
                        )
                    )
                    continue

                is_stable = stability.is_stable
                sheet.append(
                    (
                        bus_analysis.bus_name,
                        voltage_analysis.signal_name,
                        self._run_name(result.run_id),
                        "Stable" if is_stable else "Unstable",
                        ", ".join(stability.failed_criteria) if not is_stable else "-",
                        stability.crisis_decision_time_s,
                        stability.crisis_decision_duration_s,
                        stability.crisis_decision_event,
                        stability.criteria.post_clearance_grace_period_s,
                        self._pass_fail(stability.is_crisis_voltage_compliant),
                        stability.cumulative_low_voltage_duration_s,
                        stability.criteria.maximum_cumulative_low_voltage_duration_s,
                        self._pass_fail(stability.is_low_voltage_duration_compliant),
                        self._pass_fail(stability.is_settled),
                        stability.fault_on_area_pu_s,
                        stability.recovery_area_pu_s,
                        stability.unified_response_area_pu_s,
                        stability.final_settlement_peak_absolute_relative_error * 100,
                        stability.criteria.maximum_final_absolute_relative_error * 100,
                        stability.final_settlement_peak_to_peak_relative_range * 100,
                        stability.criteria.maximum_final_peak_to_peak_relative_range * 100,
                    )
                )

        self._format_sheet(sheet)

    def _write_criteria(self, sheet) -> None:
        reference_analysis = self.study_analysis.bus_analyses[0].signal_analysis
        initialization = self._initialization_criteria(reference_analysis)
        stability = self._stability_criteria(reference_analysis)
        sheet.append(("Analysis", "Criterion", "Value"))
        rows = [
            ("Initialization", "Evaluation window [s]", f"{initialization.evaluation_start_time_s:.2f} to {initialization.evaluation_end_time_s:.2f}"),
            ("Initialization", "Smoothing window [s]", initialization.smoothing_duration_s),
            ("Initialization", "Maximum P99 fast error [%]", initialization.maximum_fast_relative_error_percentile_99 * 100),
            ("Stability", "Minimum simulation end time [s]", stability.minimum_simulation_end_time_s),
            ("Voltage", "Fault time [s]", stability.fault_time_s),
            ("Voltage", "Clearance time [s]", stability.clearance_time_s),
            ("Voltage", "Response end time [s]", stability.response_end_time_s),
            ("Voltage", "0.70 pu crisis threshold [pu]", stability.crisis_voltage_threshold_pu),
            ("Voltage", "0.70 pu deadline after clearance [s]", stability.post_clearance_grace_period_s),
            ("Voltage", "Low-voltage threshold [pu]", stability.low_voltage_threshold_pu),
            ("Voltage", "Maximum cumulative duration below 0.80 pu [s]", stability.maximum_cumulative_low_voltage_duration_s),
            ("Stability", "Final absolute error limit [%]", stability.maximum_final_absolute_relative_error * 100),
            ("Stability", "Final range limit [%]", stability.maximum_final_peak_to_peak_relative_range * 100),
        ]
        if self.transfer_analysis and self.transfer_analysis.assessments:
            transfer_criteria = self.transfer_analysis.assessments[0].criteria
            rows.extend(
                (
                    (
                        "Transfer damping",
                        "Electromechanical mode band [Hz]",
                        f"{transfer_criteria.mode_minimum_frequency_hz:.1f} to "
                        f"{transfer_criteria.mode_maximum_frequency_hz:.1f}",
                    ),
                    (
                        "Transfer damping",
                        "Post-clearance guard time [s]",
                        transfer_criteria.post_clearance_guard_time_s,
                    ),
                    (
                        "Transfer damping",
                        "Analysis start time [s]",
                        transfer_criteria.analysis_start_time_s,
                    ),
                    (
                        "Transfer damping",
                        "Growing-mode confirmation",
                        (
                            f"{transfer_criteria.growth_confirmation_peak_count} consecutive extrema, "
                            f"each at least {((transfer_criteria.growth_confirmation_amplitude_ratio - 1) * 100):.0f}% larger"
                        ),
                    ),
                    ("Transfer damping", "Minimum damping ratio [%]", transfer_criteria.minimum_damping_ratio * 100),
                )
            )
        for row in rows:
            sheet.append(row)

        self._format_sheet(sheet)

    def _initialization_criteria(self, voltage_analysis: SignalStudyAnalysis):
        if not voltage_analysis.run_results:
            if self.profile is not None:
                return self.profile.initialization_criteria
            from signal_analysis import InitializationCriteria

            return InitializationCriteria()
        return voltage_analysis.run_results[0].initialization.criteria

    def _stability_criteria(self, voltage_analysis: SignalStudyAnalysis):
        for result in voltage_analysis.run_results:
            if result.stability is not None:
                return result.stability.criteria
        if self.profile is not None:
            return self.profile.voltage_criteria
        from signal_analysis import StabilityCriteria

        return StabilityCriteria()

    @staticmethod
    def _run_name(run_id: int) -> str:
        return f"run_{run_id:05d}"

    @staticmethod
    def _pass_fail(value: bool) -> str:
        return "Pass" if value else "Fail"

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
    def _format_transfer_sheet(sheet) -> None:
        AnalysisReport._format_sheet(sheet)
        for column in sheet.columns:
            for cell in column[1:]:
                if isinstance(cell.value, float):
                    cell.number_format = "0.00"
