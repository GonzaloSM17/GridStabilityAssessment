"""Check case/class weighting, capacity fractions and distribution export."""
from dataclasses import replace
from math import pi
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stability.dynamic_stability import assess_dynamic_stability
from montecarlo.spatial_stability.models import SpatialControllerPoint,SpatialCaseAssessment,SpatialStabilityAnalysis
from montecarlo.spatial_stability.distributions import case_features,technology_features,population_ecdfs,weighted_ecdf,export_distributions


def point(case,device,power,xi,fn):
    ki=(2*pi*fn)**2
    return SpatialControllerPoint(case,1 if case=='a' else 2,device,device,device,0.,0.,power,2*xi*np.sqrt(ki),ki,xi,fn)


def sample():
    points=(point('a','BESS_A',1.,.6,5.),point('a','PFV_B',3.,.8,15.),
            point('b','BESS_A',1.,.9,10.),point('b','PFV_B',3.,.5,20.))
    cases=(SpatialCaseAssessment('a',assess_dynamic_stability(1,'Stable','Damped','Stable')),
           SpatialCaseAssessment('b',assess_dynamic_stability(2,'Unstable','Damped','Stable')))
    return SpatialStabilityAnalysis(points,cases)


class SpatialDistributionTests(unittest.TestCase):
    def test_case_weighted_means_and_joint_capacity_fraction(self):
        rows={r['case_id']:r for r in case_features(sample())}
        self.assertAlmostEqual(rows['a']['weighted_xi'],.75)
        self.assertAlmostEqual(rows['a']['weighted_fn_hz'],12.5)
        self.assertAlmostEqual(rows['a']['low_xi_mva_fraction'],.25)
        self.assertAlmostEqual(rows['a']['high_fn_mva_fraction'],.75)
        self.assertAlmostEqual(rows['a']['joint_low_xi_high_fn_mva_fraction'],0.)
        self.assertAlmostEqual(rows['b']['joint_low_xi_high_fn_mva_fraction'],.75)
        self.assertEqual(rows['a']['status'],'Stable')
        self.assertEqual(rows['b']['status'],'Unstable')

    def test_population_gives_equal_mass_to_each_case_despite_different_mva(self):
        analysis=sample()
        points=tuple(replace(p,nominal_power_mva=p.nominal_power_mva*100) if p.case_id=='b' else p for p in analysis.points)
        cases=tuple(replace(c,assessment=assess_dynamic_stability(c.assessment.run_id,'Stable','Damped','Stable')) for c in analysis.cases)
        curves=[r for r in population_ecdfs(replace(analysis,points=points,cases=cases)) if r['status']=='Stable' and r['metric']=='xi']
        np.testing.assert_allclose([r['value'] for r in curves],[.5,.6,.8,.9])
        np.testing.assert_allclose([r['cumulative_mva_fraction'] for r in curves],[.375,.5,.875,1.])
        self.assertEqual({r['n_cases'] for r in curves},{2})

    def test_ecdf_combines_ties_and_rejects_invalid_weights(self):
        x,y=weighted_ecdf([2,1,1],[2,1,1])
        np.testing.assert_allclose(x,[1,2]);np.testing.assert_allclose(y,[.5,1])
        self.assertEqual(len(weighted_ecdf([])[0]),0)
        with self.assertRaises(ValueError):weighted_ecdf([1],[0])
        with self.assertRaises(ValueError):weighted_ecdf([1],[-1])

    def test_technology_statistics_do_not_mix_cases_or_units(self):
        rows={(r['case_id'],r['technology']):r for r in technology_features(sample())}
        self.assertEqual(len(rows),4)
        self.assertAlmostEqual(rows[('a','BESS')]['weighted_xi'],.6)
        self.assertAlmostEqual(rows[('b','PFV')]['weighted_fn_hz'],20.)
        self.assertEqual(rows[('b','PFV')]['status'],'Unstable')

    def test_duplicate_devices_and_unmatched_labels_are_rejected(self):
        a=sample()
        with self.assertRaisesRegex(ValueError,'Duplicate device'):
            case_features(replace(a,points=(*a.points,a.points[0])))
        with self.assertRaisesRegex(ValueError,'do not match'):
            case_features(replace(a,cases=a.cases[:1]))
        with self.assertRaisesRegex(ValueError,'thresholds'):
            case_features(a,low_xi_threshold=-.7)

    def test_export_handles_missing_class_and_closes_figures(self):
        a=sample();a=replace(a,points=a.points[:2],cases=a.cases[:1])
        before=plt.get_fignums()
        with TemporaryDirectory() as directory:
            paths=export_distributions(a,Path(directory))
            self.assertEqual(len(paths),5)
            self.assertTrue(all(p.is_file() and p.suffix=='.png' for p in paths))
            metadata=json.loads((Path(directory)/'distributions_by_stability/distribution_metadata.json').read_text())
            self.assertEqual(metadata['counts'],{'Stable':1,'Unstable':0,'Not evaluated':0})
            self.assertFalse(list(Path(directory).rglob('*.svg')))
        self.assertEqual(plt.get_fignums(),before)


if __name__=='__main__':unittest.main()
