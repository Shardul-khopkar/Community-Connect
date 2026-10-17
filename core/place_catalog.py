import csv
import math
from functools import lru_cache
from pathlib import Path


PLACE_CATALOG_PATH = Path(__file__).resolve().parent.parent / "data" / "liguria_places.csv"
_REQUIRED_COLUMNS = {"place_id", "name", "province", "lat", "lon", "source_year"}


@lru_cache(maxsize=1)
def load_place_catalog(path: str = str(PLACE_CATALOG_PATH)) -> dict[str, dict]:
    places = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as catalog_file:
        reader = csv.DictReader(catalog_file)
        if not reader.fieldnames or not _REQUIRED_COLUMNS <= set(reader.fieldnames):
            raise ValueError("Place catalog is missing required columns")
        for row in reader:
            place_id = row["place_id"].strip()
            if not place_id or place_id in places:
                raise ValueError("Place catalog contains an empty or duplicate place ID")
            latitude = float(row["lat"])
            longitude = float(row["lon"])
            if not math.isfinite(latitude) or not -90 <= latitude <= 90:
                raise ValueError(f"Invalid latitude for place {place_id}")
            if not math.isfinite(longitude) or not -180 <= longitude <= 180:
                raise ValueError(f"Invalid longitude for place {place_id}")
            places[place_id] = {
                "place_id": place_id,
                "name": row["name"].strip(),
                "province": row["province"].strip(),
                "lat": latitude,
                "lon": longitude,
                "source_year": row["source_year"].strip(),
            }
    if not places:
        raise ValueError("Place catalog is empty")
    return places


def place_options() -> list[dict]:
    return sorted(load_place_catalog().values(), key=lambda place: (place["province"], place["name"], place["place_id"]))