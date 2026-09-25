from dataclasses import dataclass
from pathlib import Path


RAW_DATA_DIR = Path("data/raw/california_2024")
PROCESSED_DATA_DIR = Path("data/processed")


@dataclass(frozen=True)
class RegionSpec:
    key: str
    county: str
    incident_areas: tuple[str, ...]

    @property
    def dataset_name(self) -> str:
        return f"{self.county}_2024"

    @property
    def data_dir(self) -> Path:
        return PROCESSED_DATA_DIR / f"{self.key}_2024"

    @property
    def artifacts_dir(self) -> Path:
        return Path(f"artifacts_{self.key}")

    @property
    def config_path(self) -> Path:
        return Path("configs/default.json")


REGIONS = {
    "sacramento": RegionSpec(
        key="sacramento",
        county="Sacramento",
        incident_areas=(
            "South Sac",
            "North Sac",
            "South Sac FSP",
            "East Sac",
            "North Sac FSP",
            "SACC",
        ),
    ),
    "fresno": RegionSpec(
        key="fresno",
        county="Fresno",
        incident_areas=("Fresno", "Coalinga", "FRFSP", "FRCC", "FR1", "FR"),
    ),
    "kern": RegionSpec(
        key="kern",
        county="Kern",
        incident_areas=("Bakersfield", "Fort Tejon", "Mojave", "Buttonwillow", "BF", "BFCC"),
    ),
    "san_francisco": RegionSpec(
        key="san_francisco",
        county="San Francisco",
        incident_areas=("San Francisco", "San Francisco FSP"),
    ),
}


def get_region(key: str) -> RegionSpec:
    normalized = str(key).strip().lower()
    try:
        return REGIONS[normalized]
    except KeyError as exc:
        raise ValueError(f"Unknown region {key!r}; expected one of {sorted(REGIONS)}") from exc
