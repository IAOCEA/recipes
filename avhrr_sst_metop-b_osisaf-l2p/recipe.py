"""
recipe for creating a stac collection for the avhrr metop-b dataset
"""

import pathlib

import apache_beam as beam
import numpy as np
import pandas as pd
import pystac
import rich_click as click
import shapely
import yaml
from beam_pyspark_runner.pyspark_runner import PySparkRunner
from pangeo_forge_recipes.transforms import OpenURLWithFSSpec, OpenWithXarray
from rich.console import Console
from tlz.functoolz import curry

from stac_recipes.patterns import FilePattern
from stac_recipes.readers import open_collections
from stac_recipes.transforms import CreateStacItem, ToPgStac

collection_id = "AVHRR_SST_METOP_B-OSISAF-L2P-v1.0"


def normalize_datetime(string):
    timestamp = pd.to_datetime(string)

    return timestamp.isoformat()


def generate_url(root, time):
    constant = "OSISAF-L2P_GHRSST-SSTsubskin-AVHRR_SST_METOP_B-sstmgr_metop01"
    fname = "-".join(
        [
            f"{time:%Y%m%d%H%M%S}",
            f"{constant}_{time:%Y%m%d}_{time:%H%M%S}",
            "v02.0-fv01.0.nc",
        ]
    )

    return f"{root}/{time.year}/{time.dayofyear:03d}/{fname}"


def generate_stac_item(ds):
    url = ds.encoding["source"]

    collection_id = ds.attrs["id"]
    item_id = ds.attrs["uuid"]

    bbox = [
        float(ds.attrs["westernmost_longitude"]),
        float(ds.attrs["southernmost_latitude"]),
        float(ds.attrs["easternmost_longitude"]),
        float(ds.attrs["northernmost_latitude"]),
    ]

    # TODO: extract the true geometry from the coordinates
    geometry = shapely.geometry.mapping(shapely.box(*bbox))

    properties = {
        "start_datetime": normalize_datetime(ds.attrs["time_coverage_start"]),
        "end_datetime": normalize_datetime(ds.attrs["time_coverage_end"]),
    }

    # TODO: figure out which stac extensions to use
    item = pystac.Item(
        id=item_id, geometry=geometry, bbox=bbox, datetime=None, properties=properties
    )
    item.add_asset("data", pystac.Asset(href=url, media_type="application/netcdf"))
    item.links.append(pystac.Link(rel="collection", target=collection_id))
    item.collection_id = collection_id

    return item


def postprocess_item(item, ds):
    return item


def create_collections(pipeline, database_config, collections_path):
    return (
        pipeline
        | beam.Create(open_collections(collections_path))
        | ToPgStac(database_config, type="collection")
    )


def create_items(pipeline, data_root, database_config, storage_kwargs):
    # full range:
    # dates = pd.date_range("2016-01-19T08:07:03", "2024-12-31T23:59:59", freq="3min")
    # subset:
    missing = np.array(
        [
            "2022-05-01 00:10:03",
            "2022-05-11 08:28:03",
            "2022-05-20 16:52:03",
            "2022-05-20 21:55:03",
            "2022-05-21 03:19:03",
            "2022-05-24 14:52:03",
            "2022-05-24 14:55:03",
            "2022-05-24 14:58:03",
            "2022-05-24 15:01:03",
            "2022-05-25 08:10:03",
            "2022-05-26 14:19:03",
            "2022-05-27 05:07:03",
            "2022-05-27 12:49:03",
            "2022-05-28 13:43:03",
            "2022-05-29 13:46:03",
            "2022-05-30 08:31:03",
            "2022-05-30 08:34:03",
            "2022-05-30 08:37:03",
        ],
        dtype="datetime64",
    )
    dates = pd.date_range(
        "2022-05-01T00:01:03", "2022-05-31T23:59:59", freq="3min"
    ).drop(labels=missing)

    pattern = FilePattern(curry(generate_url, data_root), dates, file_type="netcdf4")
    return (
        pipeline
        | beam.Create(pattern.items())
        | OpenURLWithFSSpec(open_kwargs=storage_kwargs)
        | OpenWithXarray(
            xarray_open_kwargs={"decode_timedelta": True}, file_type=pattern.file_type
        )
        | CreateStacItem(
            template=generate_stac_item,
            postprocess=postprocess_item,
            xstac_kwargs={
                "reference_system": "epsg:4326",
                "x_dimension": "ni",
                "y_dimension": "nj",
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

    data_root = "https://osi-saf.ifremer.fr/sst/l2p/global/avhrr_metop_b"

    console.log(f"creating items for data at {data_root}")
    runtime_config = yaml.safe_load(config_file)
    database_config = runtime_config["pgstac"]
    storage_kwargs = runtime_config.get("storage_kwargs", {})

    collections_path = recipe_root / "collections.yaml"

    console.log("creating collections")
    with beam.Pipeline(runner=PySparkRunner()) as p:
        create_collections(p, database_config, collections_path)
    console.log("finished creating collections")

    console.log("creating items")
    with beam.Pipeline(runner=PySparkRunner()) as p:
        create_items(p, data_root, database_config, storage_kwargs)
    console.log("finished creating items")


if __name__ == "__main__":
    main()
