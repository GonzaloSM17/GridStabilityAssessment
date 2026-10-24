"""Focused tests for the single-run general-overview signal definitions."""

import unittest

from single_run.general_overview import GENERAL_FIGURES, GeneralOverview


class GeneralOverviewDefinitionsTests(unittest.TestCase):
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
