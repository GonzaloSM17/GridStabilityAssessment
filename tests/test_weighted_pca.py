"""Validate physical weighting before PCA rather than post-hoc coefficient shares."""
from dataclasses import replace
import unittest
import numpy as np
from sklearn.preprocessing import StandardScaler

from test_spatial_clustering import sample
from montecarlo.spatial_stability.technology_analysis import technology_composition
from montecarlo.spatial_stability.weighted_pca import (
    weighted_profiles, feature_population_weights, fit_weighted_pca,
)


class WeightedPCATests(unittest.TestCase):
    def test_feature_weights_are_actual_population_fractions(self):
        analysis = sample()
        analysis = replace(analysis, points=tuple(
            replace(p, nominal_power_mva=10*p.nominal_power_mva)
            if p.generator_name.startswith("BESS_") else p for p in analysis.points))
        composition = technology_composition(analysis, [c.case_id for c in analysis.cases])
        units = feature_population_weights(composition, "units")
        mva = feature_population_weights(composition, "mva")
        np.testing.assert_allclose(units[:8], .25)
        self.assertAlmostEqual(mva[0], 40/52)
        self.assertAlmostEqual(mva[4], 40/52)
        np.testing.assert_array_equal(mva[8:], [1., 1.])

    def test_weights_survive_scaling_and_scores_reconstruct_from_coefficients(self):
        values = np.random.default_rng(72).normal(size=(40, 10))
        weights = np.array([.1, .2, .3, .4]*2 + [1., 1.])
        model = fit_weighted_pca(values, weights)
        z = StandardScaler().fit_transform(values)
        weighted = z * np.sqrt(weights)
        np.testing.assert_allclose(weighted.var(axis=0), weights)
        expected = (weighted-np.array(model["weighted_center"])) @ np.array(model["components"]).T
        np.testing.assert_allclose(model["scores"], expected, atol=1e-12)
        effective = z @ np.array(model["effective_standardized_coefficients"]).T
        np.testing.assert_allclose(model["scores"], effective, atol=1e-12)
        unweighted = fit_weighted_pca(values, np.ones(10))
        self.assertFalse(np.allclose(model["explained_variance_ratio"],
                                     unweighted["explained_variance_ratio"]))

    def test_unit_profiles_use_simple_means_and_mva_profiles_use_power(self):
        analysis = sample()
        first = analysis.cases[0].case_id
        analysis = replace(analysis, points=tuple(
            replace(p, xi=.6 if p.device_id == "BESS_0" else .9)
            if p.case_id == first else p for p in analysis.points))
        units = weighted_profiles(analysis, [first], "units")
        mva = weighted_profiles(analysis, [first], "mva")
        self.assertAlmostEqual(units.values[0, 0], .75)
        self.assertAlmostEqual(mva.values[0, 0], .825)
        with self.assertRaises(ValueError):
            weighted_profiles(analysis, [first], "other")

    def test_constant_profiles_and_invalid_weights_do_not_create_spurious_axes(self):
        result = fit_weighted_pca(np.ones((8, 10)), np.ones(10))
        self.assertEqual(result["status"], "not_fitted")
        with self.assertRaises(ValueError):
            fit_weighted_pca(np.ones((8, 10)), np.zeros(10))
        with self.assertRaises(ValueError):
            fit_weighted_pca(np.ones((8, 9)), np.ones(10))


if __name__ == "__main__":
    unittest.main()
