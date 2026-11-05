"""Regression tests for compact Monte Carlo angular evidence retention."""

import unittest

from montecarlo.angular_stability import (
    MAXIMUM_COI_PLOT_SAMPLES,
    MonteCarloAngularStability,
)
from stability.angular import (
    AngularMachineInput,
    assess_angular_stability,
    assess_representative_machine_reference,
)


class MonteCarloAngularCompactTests(unittest.TestCase):
    def test_retains_scalar_evidence_and_a_bounded_coi_plot_trace(self):
        time_values = tuple(index / 100.0 for index in range(2_001))
        machines = (
            AngularMachineInput("M1", "M1/Teta_1_1", 4.0, 100.0),
            AngularMachineInput("M2", "M2/Teta_1_1", 5.0, 200.0),
        )
        angles = {
            "M1": tuple(0.10 * time for time in time_values),
            "M2": tuple(1.00 * time for time in time_values),
        }
        calculation = assess_angular_stability(time_values, angles, machines)
        representative = assess_representative_machine_reference(
            time_values,
            angles,
            machines,
        )

        compact = MonteCarloAngularStability._compact_run_assessment(
            1,
            calculation,
            representative,
        )

        self.assertEqual(compact.analysis, calculation.analysis)
        self.assertEqual(compact.representative_analysis, representative.analysis)
        self.assertLessEqual(len(compact.coi_time_values), MAXIMUM_COI_PLOT_SAMPLES + 1)
        self.assertLessEqual(compact.coi_time_values[-1], 10.0)
        self.assertEqual(
            compact.representative_maximum_excursion_tmax_degrees,
            max(
                machine.maximum_absolute_excursion_degrees
                for machine in representative.analysis.machines
            ),
        )
        self.assertLessEqual(
            compact.representative_maximum_excursion_10s_degrees,
            compact.representative_maximum_excursion_tmax_degrees,
        )
        self.assertFalse(hasattr(compact, "calculation"))
        self.assertFalse(hasattr(compact, "representative_reference"))


if __name__ == "__main__":
    unittest.main()
