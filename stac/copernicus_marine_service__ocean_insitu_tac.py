import json
import pathlib
import re

import attrs
import dask
import dask.bag
import fsspec
import numpy as np
import pystac
import shapely
import xarray as xr
import xstac
from rich.console import Console
from rich.progress import track
from toolz.dicttoolz import itemmap, keyfilter, valmap
from toolz.functoolz import compose_left, curry, identity, juxt
from toolz.itertoolz import first, groupby, nth, second

console = Console()

category_names = {
    "bo": "bottles",
    "ct": "conductivity, temperature, and depth sensors (CTD)",
    "db": "drifting buoys",
    "fb": "ferrybox",
    "gl": "gliders",
    "hf": "high frequency radars",
    "ml": "mini loggers",
    "mo": "moorings",
    "pf": "profilers",
    "rf": "river flows",
    "sd": "saildrones",
    "sm": "sea mammals",
    "tg": "tide gauges",
    "ts": "thremosalinometer",
    "tx": "thermistor chains",
    "xb": "expendable bathythermographs (XBT)",
}


def category_name(abbrev):
    return category_names.get(abbrev.lower(), abbrev)


@attrs.define
class Parts:
    region = attrs.field()
    data_type = attrs.field()
    category = attrs.field()
    platform = attrs.field()
    time = attrs.field()

    @classmethod
    def from_url(cls, url):
        fname = url.rsplit("/", maxsplit=1)[-1].removesuffix(".nc")

        match = re.match(
            r"^(?P<region>[A-Z]+)_(?P<data_type>[A-Z]+)_(?P<category>[A-Z]+)_(?P<platform>.{2,})_(?P<time>[0-9]+)",
            fname,
        )
        if match is None:
            raise ValueError(f"unexpected filename: {url}")

        return cls(**match.groupdict())

    @property
    def item_id(self):
        return "-".join(
            [self.region, self.data_type, self.category, self.platform, self.time]
        )


@attrs.define
class CollectionParts:
    root = attrs.field()
    dataset = attrs.field()
    aggregation = attrs.field()
    category = attrs.field(default=None)

    @classmethod
    def from_url(cls, url):
        parts = url.rstrip("/").rsplit("/", maxsplit=3)

        return cls(*parts)

    @property
    def collection_id(self):
        return "-".join([self.dataset, self.aggregation, self.category]).lower()


def create_geometry(coords):
    if coords.shape[1] != 2:
        raise ValueError(
            f"invalid coordinates or invalid order of dimensions: {coords.shape}"
        )
    elif coords.size == 2:
        return shapely.Point(coords)
    else:
        return shapely.LineString(coords)


def simplify_geometry(geometry, tolerance, point_threshold):
    # using the convex hull makes sense for point-like and areas
    # TODO: find a better way to detect actual line geometries
    if geometry.convex_hull.area < point_threshold:
        simplified = geometry.convex_hull
        if simplified.geom_type == "Point":
            simplified = simplified.buffer(tolerance)
    else:
        simplified = geometry.simplify(tolerance)

    return simplified


def create_item(fs, url, open_kwargs=None):
    open_kwargs = open_kwargs or {}

    parts = Parts.from_url(url)

    bbox_attrs = [
        "geospatial_lon_min",
        "geospatial_lat_min",
        "geospatial_lon_max",
        "geospatial_lat_max",
    ]

    with xr.open_dataset(fs.open(url), **open_kwargs) as ds:
        bbox = [float(ds.attrs[name]) for name in bbox_attrs]

        # TODO: split along antemeridian to be valid geojson
        coords = np.stack([ds["LONGITUDE"].values, ds["LATITUDE"].values], axis=-1)
        geometry = create_geometry(coords)
        simplified = simplify_geometry(geometry, tolerance=0.001, point_threshold=1e-4)
        geojson = json.loads(shapely.to_geojson(simplified))
        template = pystac.Item(
            parts.item_id,
            geometry=geojson,
            bbox=bbox,
            datetime=None,
            properties={"start_datetime": None, "end_datetime": None},
        )
        item = xstac.xarray_to_stac(
            xstac.fix_attrs(ds),
            template,
            x_dimension="LONGITUDE",
            y_dimension="LATITUDE",
            reference_system="epsg:4326",
        )

    extra_fields = {"xarray:open_kwargs": {"engine": "h5netcdf"}}
    item.add_asset(
        "https",
        pystac.Asset(
            href=url, media_type="application/netcdf", extra_fields=extra_fields
        ),
    )

    item.validate()

    return item


def create_collection(
    dataset_id, abbrev, title, items, *, description="{{ description }}"
):
    collection_id = f"{dataset_id}-{abbrev.lower()}"

    min_time = min(item.properties["start_datetime"] for item in items)
    max_time = max(item.properties["end_datetime"] for item in items)

    extent = pystac.Extent(
        spatial=pystac.SpatialExtent(bboxes=[[-180, -90, 180, 90]]),
        temporal=pystac.TemporalExtent.from_dict({"interval": [[min_time, max_time]]}),
    )

    providers = []
    keywords = []
    extra_fields = {}

    col = pystac.Collection(
        collection_id,
        title=title,
        description=description,
        extent=extent,
        keywords=keywords,
        extra_fields=extra_fields,
        providers=providers,
        license="proprietary",
    )
    col.set_self_href("collection.json")
    col.add_items(items)
    col.validate()
    col.remove_links(pystac.RelType.SELF)
    # col.remove_links(pystac.RelType.ROOT)

    return col


def extract_category(url, *, pathsep="/", sep="_"):
    fname = url.rsplit(pathsep, maxsplit=1)[-1]
    parts = fname.split(sep, maxsplit=4)

    try:
        return parts[2]
    except IndexError:
        raise ValueError(f"unknown filename: {fname}")


def collection_from_urls(prefix, urls):
    print(urls)


def create_item_(fs, url, open_kwargs=None):
    try:
        return create_item(fs, url, open_kwargs=open_kwargs)
    except ValueError as e:
        raise ValueError(f"failed to open {url}") from e


def save(item, path):
    path = pathlib.Path(path)
    d = item.to_dict(include_self_link=False)
    s = json.dumps(d)

    if not path.parent.exists():
        path.parent.mkdir(exist_ok=True, parents=True)
    path.write_text(s)

    return path


@curry
def starcall(f, args, **kwargs):
    return f(*args, **kwargs)


def create_outpath(root, category, url):
    parts = Parts.from_url(url)
    item_id = parts.item_id

    return f"{root}/{category}/{item_id}.json"


if __name__ == "__main__":
    fs = fsspec.filesystem("http")
    root = (
        "https://data-marineinsitu.ifremer.fr/cmems_obs-ins_glo_phybgcwav_mynrt_na_irr"
    )
    dataset_id = "cmems_obs-ins_glo_phybgcwav_mynrt_na_irr"

    local_root = pathlib.Path(__file__).parent
    url_cache = local_root / "data_paths.json"
    if not url_cache.exists():
        with console.status("fetching file urls..."):
            urls = fs.glob(f"{root}/monthly/**/*.nc")
        with console.status("caching urls to disk..."):
            url_cache.write_text(json.dumps(urls))
    else:
        with console.status("reading urls from disk..."):
            urls = json.loads(url_cache.read_text())

    from distributed import Client, LocalCluster

    outroot = local_root.parent
    item_root = outroot.joinpath(f"items/{dataset_id}")
    catalog_root = outroot / "by_category"

    # create items
    # with LocalCluster(n_workers=12) as cluster:
    #     client = cluster.get_client()

    #     print("dashboard link:", client.dashboard_link)
    #     url_bag = dask.bag.from_sequence(urls, partition_size=100)
    #     categorized = url_bag.map(juxt(extract_category, identity)).filter(
    #         compose_left(first, lambda cat: cat not in ["HF"])
    #     )

    #     categorized_urls = categorized.map(
    #         juxt(second, starcall(curry(create_outpath, outroot)))
    #     ).filter(compose_left(second, lambda p: not pathlib.Path(p).exists()))

    #     items = categorized_urls.map(
    #         juxt(compose_left(first, curry(create_item_, fs, open_kwargs={})), second)
    #     )

    #     _ = items.starmap(save).compute()

    # create collections by category
    fs = fsspec.filesystem("file")
    item_paths = fs.find(item_root)
    categorized = list(
        map(juxt(curry(extract_category, sep="-"), identity), item_paths)
    )
    groups = groupby(first, categorized)
    items = valmap(
        compose_left(
            juxt(
                curry(map, compose_left(second, pystac.Item.from_file)),
                len,
                compose_left(curry(map, first), set, first),
            ),
            lambda x: track(x[0], total=x[1], description=f"loading files for {x[2]}"),
        ),
        groups,
    )
    collections = itemmap(
        juxt(
            first,
            compose_left(
                juxt(
                    lambda _: dataset_id,
                    first,
                    compose_left(first, category_name),
                    compose_left(second, list),
                ),
                starcall(create_collection),
            ),
        ),
        items,
    )
    catalog = pystac.Catalog(id=dataset_id, description="cmems marine in-situ")
    collection_root = catalog_root / dataset_id.upper()
    for col in track(
        collections.values(), total=len(collections), description="writing collections"
    ):
        if col.id.split("-")[-1].lower() in ["DB", "PF"]:
            continue
        href = collection_root / col.id.upper()
        col.normalize_hrefs(str(href))
        # col.validate_all()
        # col.set_root(catalog)
        col.save(catalog_type=pystac.CatalogType.SELF_CONTAINED)

    # by_category = urls.foldby(
    #     curry(extract_category, f"{root}/monthly"),
    #
    # ).compute()
    # col = create_collection(fs, f"{root}/monthly", "MO")
    # console.print(col.to_dict())
