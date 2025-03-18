"""
recipe for creating a stac collection for the cmems marine insitu tac

https://doi.org/10.48670/moi-00036
"""

import hashlib
import pathlib

import apache_beam as beam
import fsspec
import pandas as pd
import pystac
import rich_click as click
import yaml
from beam_pyspark_runner.pyspark_runner import PySparkRunner
from pangeo_forge_recipes.transforms import OpenURLWithFSSpec, OpenWithXarray
from rich.console import Console
from stac_insitu.geometry import extract_geometry
from tlz.itertoolz import concat

from stac_recipes.patterns import FilePattern
from stac_recipes.readers import open_collections
from stac_recipes.transforms import CreateStacItem, ToPgStac

collection_root_id = "INSITU_GLO_PHYBGCWAV_DISCRETE_MYNRT_013_030"


def cached_glob(fs, glob: str, *, cache_root: pathlib.Path, cache=True):
    m = hashlib.sha256()
    m.update(glob.encode())
    hash_ = m.hexdigest()

    cache_path = cache_root.joinpath(hash_).with_suffix(".parquet")
    if cache_path.exists() and cache:
        df = pd.read_parquet(str(cache_path))
        return df["urls"].to_list()

    urls = sorted(fs.glob(glob))
    if cache:
        df = pd.DataFrame({"urls": urls})
        df.to_parquet(cache_path)

    return urls


def reencode_surrogates(ds):
    def fix_dict(attrs):
        return {name: fix_value(value) for name, value in attrs.items()}

    def fix_value(value):
        if not isinstance(value, str):
            return value
        return value.encode("utf-8", "surrogateescape").decode("utf-8")

    ds_ = ds.copy()

    for k, v in ds_.variables.items():
        v.attrs = fix_dict(v.attrs)
    ds_.attrs = fix_dict(ds_.attrs)

    return ds_


def normalize_datetime(string):
    timestamp = pd.to_datetime(string)

    return timestamp.isoformat()


def generate_stac_item(ds):
    url = ds.encoding["source"]

    category = url.rsplit("/", maxsplit=3)[1]

    collection_id = f"{collection_root_id}-{category}"
    item_id = ds.attrs["id"]

    bbox_strings = [
        ds.attrs["geospatial_lon_min"],
        ds.attrs["geospatial_lat_min"],
        ds.attrs["geospatial_lon_max"],
        ds.attrs["geospatial_lat_max"],
    ]

    try:
        bbox = list(map(float, bbox_strings))
    except ValueError as e:
        raise ValueError(ds.attrs, bbox_strings) from e
    geometry, time = extract_geometry(
        ds.squeeze(), tolerance=0.001, x="LONGITUDE", y="LATITUDE", time="TIME"
    )

    properties = {
        "start_datetime": None,
        "end_datetime": None,
        "collection": collection_id,
    }
    if time is not None:
        properties["datetimes"] = time

    stac_extensions = [
        "https://stac-extensions.github.io/moving-features/v1.0.0/schema.json",
    ]

    item = pystac.Item(
        item_id,
        geometry=geometry,
        bbox=bbox,
        datetime=None,
        properties=properties,
        stac_extensions=stac_extensions,
    )
    item.add_asset(
        "public",
        pystac.Asset(href=url, media_type="application/netcdf"),
    )
    item.links.append(pystac.Link(rel="collection", target=collection_id))
    item.collection_id = collection_id

    return item


def create_collections(pipeline, database_config, collections_path):
    return (
        pipeline
        | beam.Create(open_collections(collections_path))
        | ToPgStac(database_config, type="collection")
    )


def create_items(
    pipeline, categories, data_root, cache_root, database_config, storage_kwargs
):
    fs = fsspec.filesystem("http", **storage_kwargs)
    urls = list(
        concat(
            [
                cached_glob(
                    fs,
                    f"{data_root}/{category}/202205/*.nc",
                    cache=True,
                    cache_root=cache_root,
                )
                for category in categories
            ]
        )
    )
    pattern = FilePattern.from_sequence(urls, file_type="netcdf4")

    return (
        pipeline
        | beam.Create(pattern.items())
        | OpenURLWithFSSpec(open_kwargs=storage_kwargs)
        | OpenWithXarray(
            xarray_open_kwargs={"decode_timedelta": True}, file_type=pattern.file_type
        )
        | CreateStacItem(
            template=generate_stac_item,
            preprocess=reencode_surrogates,
            xstac_kwargs={
                "reference_system": "epsg:4326",
                "x_dimension": "LONGITUDE",
                "y_dimension": "LATITUDE",
            },
        )
        | ToPgStac(database_config, type="item")
    )


@click.command()
@click.argument("config_file", type=click.File(mode="r"))
def main(config_file):
    console = Console()

    recipe_root = pathlib.Path(__file__).parent
    console.log(f"running recipe at {recipe_root}")

    data_root = "https://data-marineinsitu.ifremer.fr/glo_multiparameter_nrt/monthly"

    console.log(f"creating items for data at {data_root}")
    runtime_config = yaml.safe_load(config_file)
    database_config = runtime_config["pgstac"]
    storage_kwargs = runtime_config.get("storage_kwargs", {})
    cache_root = pathlib.Path(runtime_config["cache_root"])

    collections_path = recipe_root / "collections.yaml"

    categories = [
        id.rsplit("-", maxsplit=1)[1] for id, _ in open_collections(collections_path)
    ]

    console.log("creating collections")
    with beam.Pipeline(runner=PySparkRunner()) as p:
        create_collections(p, database_config, collections_path)
    console.log("finished creating collections")

    console.log("creating items")
    with beam.Pipeline(runner=PySparkRunner()) as p:
        create_items(
            p, categories, data_root, cache_root, database_config, storage_kwargs
        )
    console.log("finished creating items")


if __name__ == "__main__":
    main()
