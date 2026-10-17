import unittest

from core.place_catalog import load_place_catalog, place_options


class PlaceCatalogTests(unittest.TestCase):
    def test_istat_catalog_has_unique_liguria_places_with_genova(self):
        places = load_place_catalog()
        self.assertEqual(len(places), 234)
        self.assertEqual(len(places), len(set(places)))
        self.assertEqual(places["010025"]["name"], "Genova")
        self.assertEqual(places["010025"]["province"], "Genova")
        self.assertGreaterEqual(places["010025"]["lat"], -90)
        self.assertLessEqual(places["010025"]["lat"], 90)
        self.assertGreaterEqual(places["010025"]["lon"], -180)
        self.assertLessEqual(places["010025"]["lon"], 180)

    def test_searchable_place_options_include_requested_municipalities_and_provinces(self):
        options = {place["name"]: place["province"] for place in place_options()}
        self.assertEqual(options["Genova"], "Genova")
        self.assertEqual(options["Sestri Levante"], "Genova")
        self.assertEqual(options["Santa Margherita Ligure"], "Genova")


if __name__ == "__main__":
    unittest.main()