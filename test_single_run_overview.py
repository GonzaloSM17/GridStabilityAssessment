"""Focused tests for the single-run general-overview signal definitions."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from data.run import MonteCarloRun
from data.signals import (
    Bus,
    PowerTransfer,
    order_500_kv_buses_north_to_south,
    order_power_transfers_north_to_south,
)
from single_run.general_overview import GENERAL_FIGURES, GeneralOverview


class GeneralOverviewDefinitionsTests(unittest.TestCase):
    def test_500_kv_buses_follow_the_declared_north_to_south_order(self):
        buses = tuple(
            Bus(name, name, ())
            for name in (
                "CHARRUA_500",
                "AJAHUEL_500",
                "POLPAICO_500",
                "KIMAL_500",
                "LCHANGOS_500",
                "NPAZUCAR_500",
                "CUMBRE_500",
            )
        )

        ordered = order_500_kv_buses_north_to_south(buses)

        self.assertEqual(
            tuple(bus.name for bus in ordered),
            (
                "KIMAL_500",
                "LCHANGOS_500",
                "CUMBRE_500",
                "NPAZUCAR_500",
                "POLPAICO_500",
                "AJAHUEL_500",
                "CHARRUA_500",
            ),
        )

    def test_power_transfers_follow_the_declared_north_to_south_order(self):
        transfers = (
            PowerTransfer("NC_NPAZUCAR_POLPAICO_C1", "PQ_NC_NPAZUCAR_POLPAICO_C1/P"),
            PowerTransfer("NC_KIMAL_CHANGOS_C1", "PQ_NC_KIMAL_CHANGOS_C1/P"),
            PowerTransfer("NC_CUMBRE_NCARDONES_C1", "PQ_NC_CUMBRE_NCARDONES_C1/P"),
            PowerTransfer("NC_CHANGOS_JADRESIC_C1", "PQ_NC_CHANGOS_JADRESIC_C1/P"),
            PowerTransfer("NC_NCARDONES_NMAITENCILLO_C1", "PQ_NC_NCARDONES_NMAITENCILLO_C1/P"),
            PowerTransfer("NC_NMAITENCILLO_NPAZUCAR_C1", "PQ_NC_NMAITENCILLO_NPAZUCAR_C1/P"),
        )

        ordered = order_power_transfers_north_to_south(transfers)

        self.assertEqual(
            tuple(transfer.name for transfer in ordered),
            (
                "NC_KIMAL_CHANGOS_C1",
                "NC_CHANGOS_JADRESIC_C1",
                "NC_CUMBRE_NCARDONES_C1",
                "NC_NCARDONES_NMAITENCILLO_C1",
                "NC_NMAITENCILLO_NPAZUCAR_C1",
                "NC_NPAZUCAR_POLPAICO_C1",
            ),
        )

    def test_single_case_directory_does_not_require_monte_carlo_naming(self):
        with TemporaryDirectory() as temporary_directory:
            result_directory = Path(temporary_directory) / "SEN2026_case"
            result_directory.mkdir()
            result = MonteCarloRun.from_directory(
                result_directory,
                allow_non_monte_carlo_name=True,
            )

        self.assertEqual(result.run_id, 0)
        self.assertEqual(result.display_name, "SEN2026_case")

    def test_monte_carlo_directory_naming_remains_required_by_default(self):
        with TemporaryDirectory() as temporary_directory:
            result_directory = Path(temporary_directory) / "SEN2026_case"
            result_directory.mkdir()

            with self.assertRaises(ValueError):
                MonteCarloRun.from_directory(result_directory)

    def test_220_kv_voltage_plot_selects_only_rms_voltage_signals(self):
        definition = next(
            item for item in GENERAL_FIGURES if item.filename == "06_220_kv_buses.png"
        )
        voltage_plot = definition.plots[0]
        self.assertTrue(voltage_plot.selector("KIMAL220_MTR/u_rms"))
        self.assertFalse(voltage_plot.selector("KIMAL220_MTR/Frequency_Meter/f"))
        self.assertFalse(voltage_plot.selector("PMGD_CRUCERO_220/PQpark/P"))

    def test_north_center_transfer_plot_selects_only_active_power_signals(self):
        definition = next(
            item
            for item in GENERAL_FIGURES
            if item.filename == "13_500_kv_power_transfers.png"
        )
        north_center_plot = definition.plots[0]
        self.assertTrue(north_center_plot.selector("PQ_NC_CUMBRE_NCARDONES_C1/P"))
        self.assertFalse(north_center_plot.selector("PQ_NC_CUMBRE_NCARDONES_C1/Q"))

    def test_general_overview_definitions_use_requested_units_and_limits(self):
        bus_500 = next(
            item for item in GENERAL_FIGURES if item.filename == "05_500_kv_buses.png"
        )
        self.assertEqual(bus_500.plots[0].y_limits, (0.0, 1.5))
        self.assertTrue(bus_500.plots[0].show_legend)
        self.assertTrue(bus_500.plots[1].show_legend)

        statcom = next(
            item for item in GENERAL_FIGURES if item.filename == "11_statcom_svc_power.png"
        )
        self.assertTrue(statcom.plots[0].show_legend)
        self.assertTrue(statcom.plots[1].show_legend)
        self.assertEqual(bus_500.plots[1].y_limits, (47.5, 52.5))

        der = next(
            item for item in GENERAL_FIGURES if item.filename == "10_der_power.png"
        )
        self.assertEqual(der.title, "DER Power")
        self.assertEqual(der.plots[0].value_scale, 1e-6)

    def test_general_overview_formats_transfer_labels_and_closes_y_limits(self):
        self.assertEqual(
            GeneralOverview._curve_label("PQ_NC_CUMBRE_NCARDONES_C1/P"),
            "CUMBRE_NCARDONES_C1",
        )
        self.assertEqual(GeneralOverview._nice_y_limits(-457.0, 767.0), (-600.0, 800.0))


if __name__ == "__main__":
    unittest.main()
