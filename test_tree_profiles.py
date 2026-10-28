"""Check class-specific leaf values, native failure counts and tree rendering."""
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest
import numpy as np
from sklearn.tree import DecisionTreeClassifier

from test_spatial_clustering import sample
from stability.dynamic_stability import assess_dynamic_stability
from montecarlo.spatial_stability.characterization import build_inputs
from montecarlo.spatial_stability.tree_profiles import (
    leaf_profiles, leaf_text, export_descriptive_trees,
)


class TreeProfileTests(unittest.TestCase):
    def setUp(self):
        self.analysis = sample(32)
        self.ids,self.y,self.x,_ = build_inputs(self.analysis)
        self.assessments = {c.case_id:c.assessment for c in self.analysis.cases}
        self.tree = DecisionTreeClassifier(max_depth=2,min_samples_leaf=8,random_state=1900)
        self.tree.fit(self.x["mva"][:,:9],self.y)

    def test_leaf_counts_and_class_medians_use_actual_memberships(self):
        summaries,profiles,membership = leaf_profiles(
            self.tree,self.x["mva"],self.y,self.ids,self.assessments)
        self.assertEqual(sum(r["n_cases"] for r in summaries),32)
        self.assertEqual(sum(r["stable"] for r in summaries),16)
        self.assertEqual(sum(r["unstable"] for r in summaries),16)
        self.assertEqual(len({r["case_id"] for r in membership}),32)
        applied = self.tree.apply(self.x["mva"][:,:9])
        for row in profiles:
            mask = applied == row["node"]
            j = list(__import__("montecarlo.spatial_stability.clustering",fromlist=["FEATURES"]).FEATURES).index(row["feature"])
            for label,status in ((0,"stable"),(1,"unstable")):
                values = self.x["mva"][mask & (self.y == label),j]
                if len(values):
                    self.assertAlmostEqual(row["median_"+status],np.median(values))
                else:
                    self.assertIsNone(row["median_"+status])
                    self.assertIsNone(row["delta_unstable_minus_stable"])
        text = leaf_text(summaries[0],[r for r in profiles if r["node"]==summaries[0]["node"]])
        self.assertIn("Estables:",text)
        self.assertIn("xi BESS",text)
        self.assertIn("Fallas / evaluados",text)
        self.assertNotIn("value =",text)

    def test_native_failures_can_overlap_and_unknown_evidence_is_not_a_pass(self):
        assessments = dict(self.assessments)
        identifier = self.ids[-1]
        assessments[identifier] = assess_dynamic_stability(
            31,"Unstable","Damped","Not evaluated")
        summaries,_,_ = leaf_profiles(
            self.tree,self.x["mva"],self.y,self.ids,assessments)
        self.assertEqual(sum(r["voltage_failures_unstable"] for r in summaries),1)
        self.assertEqual(sum(r["angle_evaluated_unstable"] for r in summaries),15)
        self.assertEqual(sum(r["angle_failures_unstable"] for r in summaries),15)
        self.assertEqual(sum(r["damping_failures_unstable"] for r in summaries),0)

    def test_native_labels_cannot_be_replaced_by_leaf_predictions(self):
        wrong = dict(self.assessments)
        wrong[self.ids[-1]] = assess_dynamic_stability(31,"Stable","Damped","Stable")
        with self.assertRaisesRegex(ValueError,"native"):
            leaf_profiles(self.tree,self.x["mva"],self.y,self.ids,wrong)

    def test_export_generates_readable_versions_and_preserves_unrelated_png(self):
        with TemporaryDirectory() as directory:
            out = Path(directory)
            preserved = out/"05_pca_groups_and_stability.png"
            preserved.write_bytes(b"preserved")
            paths = export_descriptive_trees(out,self.x,self.y,self.ids,self.assessments)
            self.assertEqual({p.name for p in paths},{
                "04_descriptive_trees.png","04_descriptive_trees_units.png",
                "04_descriptive_trees_mva.png"})
            self.assertTrue(all(p.is_file() for p in paths))
            self.assertEqual(preserved.read_bytes(),b"preserved")
            metadata = json.loads((out/"tree_profile_metadata.json").read_text())
            self.assertEqual(metadata["case_ids"],list(self.ids))
            self.assertTrue((out/"tree_leaf_profiles.csv").is_file())
            self.assertTrue((out/"tree_leaf_membership.csv").is_file())


if __name__=="__main__":
    unittest.main()
