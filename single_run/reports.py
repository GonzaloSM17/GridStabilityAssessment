"""Database-style Excel report for a focused single-run assessment."""

from pathlib import Path

from data.run import MonteCarloRun
from single_run.angular_stability import AngularStabilityAnalysis
from single_run.damping_stability import TransferDampingAnalysis
from single_run.voltage_stability import VoltageStabilityAnalysis
from stability.scenario import EvaluationProfile


class SingleRunReport:
    """Write concise conclusions plus traceable, relational-style details."""

    def __init__(self, run: MonteCarloRun, profile: EvaluationProfile, voltage: VoltageStabilityAnalysis, damping: TransferDampingAnalysis, angular: AngularStabilityAnalysis) -> None:
        self.run = run
        self.profile = profile
        self.voltage = voltage
        self.damping = damping
        self.angular = angular

    def write(self, output_directory: str | Path) -> Path:
        """Create the workbook at the root of the run analysis outputs."""
        from openpyxl import Workbook

        output_path = Path(output_directory) / "analysis_report.xlsx"
        workbook = Workbook()
        sheets = {
            "Summary": workbook.active,
            "Voltage Assessment": workbook.create_sheet("Voltage Assessment"),
            "Transfer Assessment": workbook.create_sheet("Transfer Assessment"),
            "EM Modes": workbook.create_sheet("EM Modes"),
            "Damping Pairs": workbook.create_sheet("Damping Pairs"),
            "Angular Assessment": workbook.create_sheet("Angular Assessment"),
            "Definitions": workbook.create_sheet("Definitions"),
        }
        sheets["Summary"].title = "Summary"
        self._write_summary(sheets["Summary"])
        self._write_voltage(sheets["Voltage Assessment"])
        self._write_transfers(sheets["Transfer Assessment"])
        self._write_modes(sheets["EM Modes"])
        self._write_pairs(sheets["Damping Pairs"])
        self._write_angular(sheets["Angular Assessment"])
        self._write_definitions(sheets["Definitions"])
        for sheet in workbook.worksheets:
            self._format_sheet(sheet)
        workbook.save(output_path)
        workbook.close()
        return output_path

    def _write_summary(self, sheet) -> None:
        voltage_total = len(self.voltage.assessments)
        voltage_stable = sum(item.assessment.is_stable for item in self.voltage.assessments)
        statuses = tuple(record.assessment.status for record in self.damping.records)
        sheet.append(("Scope", "Item", "Value", "Detail"))
        rows = (
            ("Case", "Run", f"run_{self.run.run_id:06d}", None),
            ("Case", "Scenario", self.profile.scenario.name, None),
            ("Case", "Severity", self.profile.scenario.severity, None),
            ("Case", "Fault / clearance [s]", f"{self.profile.scenario.fault_time_s:.3f} / {self.profile.scenario.clearance_time_s:.3f}", None),
            ("Voltage", "220/500 kV assessed buses", voltage_total, None),
            ("Voltage", "Buses passing all active criteria", voltage_stable, None),
            ("Voltage", "Buses failing one or more criteria", voltage_total - voltage_stable, None),
            ("Damping", "Published active-power transfers", len(statuses), None),
            ("Damping", "Damped", statuses.count("damped"), None),
            ("Damping", "Damped under / growing", statuses.count("damped_under") + statuses.count("growing"), None),
            ("Damping", "Not measurable / not applicable", sum(statuses.count(status) for status in ("not_measurable", "opened", "inactive_pre_fault")), None),
            ("Angular", "COI-referenced machine excursions", len(self.angular.machines), self.angular.disclaimer),
            ("Angular", "Angular stability", "Stable" if self.angular.is_stable else "Unstable", "All machines must remain strictly within ±120°"),
            ("Angular", "Machines within ±120°", sum(machine.is_within_limit for machine in self.angular.machines), None),
        )
        for row in rows:
            sheet.append(row)

    def _write_voltage(self, sheet) -> None:
        sheet.append(("Bus", "Voltage level [kV]", "Assessment", "0.70 pu compliant", "0.70 pu event", "0.70 pu decision time [s]", "0.70 pu duration [s]", "Below 0.80 pu duration [s]", "0.80 pu compliant", "Settlement compliant", "Final peak error [%]", "Final range [%]", "Fault-on area [pu*s]", "Recovery area [pu*s]", "Unified response area [pu*s]"))
        for item in self.voltage.assessments:
            value = item.assessment
            sheet.append((item.bus.name, item.voltage_level_kv, "Stable" if value.is_stable else "Unstable", value.is_crisis_voltage_compliant, value.crisis_decision_event, value.crisis_decision_time_s, value.crisis_decision_duration_s, value.cumulative_low_voltage_duration_s, value.is_low_voltage_duration_compliant, value.is_settled, value.final_settlement_peak_absolute_relative_error * 100, value.final_settlement_peak_to_peak_relative_range * 100, value.fault_on_area_pu_s, value.recovery_area_pu_s, value.unified_response_area_pu_s))

    def _write_transfers(self, sheet) -> None:
        sheet.append(("Transfer", "Pre-fault P [MW]", "Final P [MW]", "Assessment", "Dominant EM mode [Hz]", "Dominant damping [%]", "Critical EM mode [Hz]", "Critical damping [%]", "Note"))
        for record in self.damping.records:
            value = record.assessment
            dominant, critical = value.dominant_mode, value.critical_mode
            sheet.append((value.transfer_name, value.pre_fault_power, value.final_power, value.classification, dominant.frequency_hz if dominant else None, dominant.damping_ratio * 100 if dominant and dominant.damping_ratio is not None else None, critical.frequency_hz if critical else None, critical.damping_ratio * 100 if critical and critical.damping_ratio is not None else None, value.warning or value.comment))

    def _write_modes(self, sheet) -> None:
        sheet.append(("Transfer", "Candidate", "Frequency [Hz]", "Lower band [Hz]", "Upper band [Hz]", "Spectral power", "Dominant", "Assessment", "Terminal direction", "Terminal damping [%]", "Selected damping [%]", "Selected extremum", "Envelope peak time [s]", "Envelope peak amplitude", "Envelope peak extremum", "Tail pairs", "Terminal sequence pairs", "Consecutive pairs", "Note"))
        for record in self.damping.records:
            value, dominant = record.assessment, record.assessment.dominant_mode
            for index, mode in enumerate(value.modes, start=1):
                sheet.append((value.transfer_name, index, mode.frequency_hz, mode.lower_band_hz, mode.upper_band_hz, mode.spectral_power, mode is dominant, mode.classification, mode.terminal_direction, mode.terminal_pair.damping_ratio * 100 if mode.terminal_pair else None, mode.damping_ratio * 100 if mode.damping_ratio is not None else None, mode.selected_extremum, mode.envelope_peak_time_s, mode.envelope_peak_amplitude, mode.envelope_peak_extremum, mode.tail_pair_count, mode.terminal_sequence_pair_count, len(mode.damping_pairs), mode.message))

    def _write_pairs(self, sheet) -> None:
        sheet.append(("Transfer", "Candidate", "Extremum", "Peak 1 time [s]", "Peak 2 time [s]", "A1", "A2", "Period [s]", "Damping [%]", "Peak 1 value", "Peak 2 value"))
        for record in self.damping.records:
            for candidate, mode in enumerate(record.assessment.modes, start=1):
                for pair in mode.damping_pairs:
                    sheet.append((record.assessment.transfer_name, candidate, pair.extremum, pair.first_time_s, pair.second_time_s, pair.first_amplitude, pair.second_amplitude, pair.period_s, pair.damping_ratio * 100, pair.first_value, pair.second_value))

    def _write_angular(self, sheet) -> None:
        sheet.append(("Machine", "Rotor-angle signal", "Inertia H [s]", "S [MVA]", "H·S [MVA*s]", "Minimum excursion [deg]", "Maximum excursion [deg]", "Maximum absolute excursion [deg]", "First ±120° violation [s]", "Within ±120°"))
        for machine in self.angular.machines:
            sheet.append((machine.machine_name, machine.signal_name, machine.inertia_s, machine.nominal_power_mva, machine.inertia_weight_mva_s, machine.minimum_excursion_degrees, machine.maximum_excursion_degrees, machine.maximum_absolute_excursion_degrees, machine.first_limit_violation_time_s, machine.is_within_limit))

    def _write_definitions(self, sheet) -> None:
        voltage, damping = self.profile.voltage_criteria, self.profile.transfer_damping_criteria
        sheet.append(("Scope", "Parameter", "Value", "Unit / meaning"))
        rows = (
            ("Voltage", "Crisis threshold", voltage.crisis_voltage_threshold_pu, "pu"),
            ("Voltage", "Recovery deadline after clearance", voltage.post_clearance_grace_period_s, "s"),
            ("Voltage", "Low-voltage threshold", voltage.low_voltage_threshold_pu, "pu"),
            ("Voltage", "Maximum cumulative time below 0.80 pu", voltage.maximum_cumulative_low_voltage_duration_s, "s"),
            ("Voltage", "Settlement peak-error limit", voltage.maximum_final_absolute_relative_error * 100, "%"),
            ("Voltage", "Settlement range limit", voltage.maximum_final_peak_to_peak_relative_range * 100, "%"),
            ("Damping", "EM frequency band", f"{damping.mode_minimum_frequency_hz:.1f}–{damping.mode_maximum_frequency_hz:.1f}", "Hz"),
            ("Damping", "Minimum damping", damping.minimum_damping_ratio * 100, "%"),
            ("Angular", "Centre of inertia", "Σ(δᵢHᵢSᵢ) / Σ(HᵢSᵢ)", "calculated per time step"),
            ("Angular", "Machine excursion", "δᵢ − δ_eje", "degrees"),
            ("Angular", "Stability criterion", "-120 < Δδᵢ(t) < +120", "strict inequality at every time step and for every machine"),
        )
        for row in rows:
            sheet.append(row)

    @staticmethod
    def _format_sheet(sheet) -> None:
        from openpyxl.styles import Alignment, Font, PatternFill

        fill = PatternFill("solid", fgColor="1F4E78")
        for cell in sheet[1]:
            cell.font = Font(color="FFFFFF", bold=True)
            cell.fill = fill
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions
        for column in sheet.columns:
            width = min(45, max(12, max(len(str(cell.value or "")) for cell in column) + 2))
            sheet.column_dimensions[column[0].column_letter].width = width
            for cell in column[1:]:
                cell.alignment = Alignment(vertical="top", wrap_text=True)

