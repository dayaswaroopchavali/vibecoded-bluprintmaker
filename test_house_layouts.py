import copy
import unittest

import house_layouts as hl


class HouseLayoutTests(unittest.TestCase):
    def test_suite_coverage_and_seed_metadata(self):
        suite=hl.generate_suite(1234)
        self.assertEqual(len(suite["layouts"]),30)
        for stage in hl.STAGES:
            rows=[x for x in suite["layouts"] if x["stage"]==stage]
            self.assertEqual(len({x["house_id"] for x in rows}),10)
            self.assertTrue(all(len(set(x["capture_seeds"]))==3 for x in rows))

    def test_single_room_contains_rectangular_and_nonconvex_floors(self):
        rows=[hl.generate_house("single_room",i,i*311+15) for i in range(40)]
        self.assertEqual({x["topology"] for x in rows},{"rectangle","l_shape"})

    def test_overlap_handles_coincident_rectangles_and_shared_vertices(self):
        square=[[0,0],[2,0],[2,2],[0,2]]
        self.assertTrue(hl._interior_overlap(square,square))
        self.assertFalse(hl._interior_overlap(square,[[2,2],[4,2],[4,4],[2,4]]))
        self.assertFalse(hl._interior_overlap(square,[[2,0],[4,0],[4,2],[2,2]]))

    def test_multiroom_means_three_living_rooms_plus_connector_and_varied_topologies(self):
        rows=[hl.generate_house("multiroom",i,i*137+5) for i in range(30)]
        self.assertGreaterEqual(len({r["topology"] for r in rows}),2)
        for row in rows:
            self.assertEqual(len(row["rooms"]),4)
            self.assertEqual(sum(r["room_id"]!="connector" for r in row["rooms"]),3)
            self.assertTrue(hl.validate_layout(row))

    def test_multifloor_has_connector_network_per_level_and_unvalidated_stairs(self):
        row=hl.generate_house("multifloor",1,876)
        self.assertEqual(len({r["floor_elevation_m"] for r in row["rooms"]}),2)
        for z in {r["floor_elevation_m"] for r in row["rooms"]}:
            self.assertEqual(sum(r["floor_elevation_m"]==z for r in row["rooms"]),3)
        self.assertFalse(row["capture_ready"])
        self.assertTrue(all(s["status"]=="physically_unvalidated" for s in row["stairs"]))
        self.assertTrue(hl.validate_layout(row))

    def test_rejects_bad_stair_and_wall_records(self):
        row=hl.generate_house("multifloor",1,876); bad=copy.deepcopy(row)
        bad["stairs"][0]["footprint_xy_m"]=[[100,100],[101,100],[101,101],[100,101]]
        with self.assertRaisesRegex(ValueError,"outside connected rooms"):
            hl.validate_layout(bad)
        bad=copy.deepcopy(row); bad["rooms"][0]["walls"][0]["end_xy_m"]=[99,99]
        with self.assertRaisesRegex(ValueError,"match actual polygon edges"):
            hl.validate_layout(bad)

    def test_rejects_nonfinite_metric_values_and_invalid_doors(self):
        row=hl.generate_house("multiroom",1,77); bad=copy.deepcopy(row)
        bad["rooms"][0]["ceiling_height_m"]=float("nan")
        with self.assertRaisesRegex(ValueError,"finite and positive"):
            hl.validate_layout(bad)
        bad=copy.deepcopy(row); bad["doorways"][0]["height_m"]=99
        with self.assertRaisesRegex(ValueError,"door height"):
            hl.validate_layout(bad)

    def test_determinism_and_seed_diversity(self):
        self.assertEqual(hl.generate_house("multiroom",2,55),hl.generate_house("multiroom",2,55))
        self.assertNotEqual(hl.generate_house("multiroom",2,55)["rooms"],hl.generate_house("multiroom",2,56)["rooms"])

    def test_rejects_invalid_polygons_disconnected_graph(self):
        with self.assertRaisesRegex(ValueError,"self-intersects"):
            hl.validate_polygon([[0,0],[2,2],[0,2],[2,0]])
        row=hl.generate_house("multiroom",1,98); bad=copy.deepcopy(row); bad["doorways"].pop()
        with self.assertRaisesRegex(ValueError,"disconnected"):
            hl.validate_layout(bad)


if __name__=="__main__": unittest.main()
