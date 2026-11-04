"""Validate feature selection, outcome preservation and nested model evaluation."""
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

import numpy as np
from threadpoolctl import threadpool_limits
from test_spatial_clustering import sample
from stability.dynamic_stability import assess_dynamic_stability
from montecarlo.spatial_stability.characterization import (
    FAMILIES, SETS, build_inputs, model_spec, compare_models,
    export_characterization, parameter_associations,
)


class CharacterizationTests(unittest.TestCase):
    def test_nine_indicator_candidate_excludes_only_fn_spread(self):
        self.assertEqual(SETS["candidate9"], tuple(range(9)))
        self.assertEqual(SETS["candidate9_correlation"], (*range(9),10))
        rng = np.random.default_rng(9)
        x = rng.normal(size=(40,11))
        y = (x[:,0]>0).astype(int)
        model, _ = model_spec("logistic", SETS["candidate9"], 42)
        with threadpool_limits(limits=1):
            model.fit(x,y)
            before = model.predict_proba(x)
            changed = x.copy()
            changed[:,9] += 1000
            np.testing.assert_allclose(before,model.predict_proba(changed))

    def test_native_unknowns_are_excluded_and_features_ignore_outcomes(self):
        analysis = sample(32)
        ids,y,x,excluded = build_inputs(analysis)
        changed = replace(analysis, cases=tuple(replace(c, assessment=assess_dynamic_stability(
            c.assessment.run_id, "Stable", "Damped", "Stable")) for c in analysis.cases))
        ids2,y2,x2,_ = build_inputs(changed)
        self.assertEqual(ids,ids2)
        for basis in x:
            np.testing.assert_allclose(x[basis],x2[basis],equal_nan=True)
        first = analysis.cases[0]
        unknown = replace(first, assessment=assess_dynamic_stability(
            first.assessment.run_id,"Not evaluated","Damped","Stable"))
        analysis = replace(analysis,cases=(unknown,*analysis.cases[1:]))
        ids,y,x,excluded = build_inputs(analysis)
        self.assertNotIn(first.case_id,ids)
        self.assertTrue(any(r["case_id"]==first.case_id for r in excluded))

    def test_same_marginals_can_have_different_xi_fn_pairing(self):
        analysis = sample(16)
        points = tuple(replace(p,
            xi=.6 if p.generator_name.endswith("_0") else .9,
            fn_hz=(20. if p.generator_name.endswith("_0") else 2.)
                if p.case_id=="run_000000" else
                (2. if p.generator_name.endswith("_0") else 20.))
            for p in analysis.points)
        ids,y,x,_ = build_inputs(replace(analysis,points=points))
        np.testing.assert_allclose(x["units"][0,:10],x["units"][1,:10])
        self.assertAlmostEqual(x["units"][0,10],-1.)
        self.assertAlmostEqual(x["units"][1,10],1.)

    def test_every_family_fits_and_produces_valid_probabilities(self):
        rng = np.random.default_rng(11)
        x = rng.normal(size=(60,11))
        x[0,10] = np.nan
        y = np.r_[np.zeros(20,dtype=int),np.ones(40,dtype=int)]
        with threadpool_limits(limits=1):
            for family in FAMILIES:
                model,_ = model_spec(family,SETS["candidate9_correlation"],42)
                model.fit(x,y)
                p = model.predict_proba(x)
                self.assertEqual(p.shape,(60,2))
                self.assertTrue(np.isfinite(p).all())
                np.testing.assert_allclose(p.sum(axis=1),1.)

    def test_nested_predictions_cover_each_case_and_selected_thresholds_are_valid(self):
        ids,y,x,_ = build_inputs(sample(32))
        result = compare_models(x,y,families=("logistic",),repeats=2,
                                outer_folds=2,inner_folds=2)
        self.assertEqual(result["status"],"fitted")
        self.assertEqual(result["fold_ids"].shape,(2,32))
        self.assertEqual(len(result["selections"]),8)
        for basis in x:
            p = result["probabilities"][basis]["nested_selection"]
            self.assertEqual(p.shape,(2,32))
            self.assertTrue(np.isfinite(p).all())
            self.assertTrue(np.all((p>=0)&(p<=1)))
            self.assertTrue(np.all((result["thresholds"][basis]>=.05)&
                                   (result["thresholds"][basis]<=.95)))
        self.assertEqual(len(result["blocks"]),16)
        self.assertFalse(compare_models(x,np.ones(32,dtype=int))["status"]=="fitted")

    def test_zero_within_class_variance_does_not_fake_zero_association(self):
        x = np.r_[np.zeros((8,11)),np.ones((8,11))]
        rows = parameter_associations(x,np.r_[np.zeros(8,dtype=int),np.ones(8,dtype=int)],
                                      bootstrap=20)
        self.assertTrue(all(r["hedges_g"] is None for r in rows))

    def test_export_writes_own_folder_and_preserves_existing_graphs(self):
        analysis = sample(32)
        with TemporaryDirectory() as directory:
            base = Path(directory)
            preserved = base/"clustering"/"05_pca_groups_and_stability.png"
            preserved.parent.mkdir()
            preserved.write_bytes(b"unchanged graph")
            paths = export_characterization(analysis,base,families=("logistic",),
                                            repeats=2,outer_folds=2,inner_folds=2)
            self.assertEqual(len(paths),7)
            self.assertTrue(all(p.is_file() and p.suffix==".png" for p in paths))
            self.assertEqual(preserved.read_bytes(),b"unchanged graph")
            out=base/"stability_characterization"
            meta=json.loads((out/"characterization_metadata.json").read_text())
            self.assertEqual(meta["n_cases"],32)
            self.assertEqual(meta["native_criterion"],
                             "Existing voltage + modal damping + angle aggregation, unchanged")
            single=replace(analysis,cases=analysis.cases[:1],
                           points=tuple(p for p in analysis.points if p.case_id==analysis.cases[0].case_id))
            self.assertEqual(export_characterization(single,base),())
            self.assertFalse(list(out.glob("*.png")))
            self.assertTrue(preserved.is_file())


if __name__=="__main__":
    unittest.main()
