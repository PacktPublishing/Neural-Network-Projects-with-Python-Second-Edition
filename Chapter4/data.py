import torch
import numpy as np
import typing as tt
import pathlib
from torch.utils.data import Dataset
from pyarrow import feather
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import kagglehub as kh

IDX_COL = "idx"
TARGET_COL = "fare_amount"
MAX_CHUNK_SIZE = 10000

LANDMARKS = {
    "jfk_airport": (40.6446, -73.7797),
    "lag_airport": (40.7766, -73.8743),
    "new_airport": (40.6885, -74.1769),
    "center":      (40.7128, -74.0060),
    "manhattan":   (40.7767, -73.9713),
    "brooklyn":    (40.6782, -73.9442),
    "queens":      (40.7282, -73.7949),
    "bronx":       (40.8370, -73.8654),
    "staten":      (40.5790, -74.1515),
}


TBatch = tt.Tuple[torch.Tensor, torch.Tensor]


class FeatherChunksDataset(Dataset):
    def __init__(self, file_name: pathlib.Path):
        self.file_name = file_name
        self.table = feather.read_table(file_name, memory_map=True)
        self.batches = self.table.to_batches(max_chunksize=MAX_CHUNK_SIZE)
        self.columns = list(self.table.schema.names)
        self.columns.remove(IDX_COL)
        self.columns.remove(TARGET_COL)

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, batch_idx: int) -> TBatch:
        batch = self.batches[batch_idx]
        x_np = np.column_stack([
            batch[col].to_numpy()
            for col in self.columns
        ]).astype(np.float32, copy=False)
        y_np = batch[TARGET_COL].to_numpy().astype(np.float32)
        return torch.from_numpy(x_np), torch.from_numpy(y_np)


def collate_batches(items):
    xs, ys = zip(*items)
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


def get_dataset() -> ds.Dataset:
    path = kh.competition_download('new-york-city-taxi-fare-prediction')
    return ds.dataset(path + "/train.csv", format="csv")


def parse_datetime(expr: pc.Expression) -> pc.Expression:
    return pc.strptime(pc.replace_substring(expr, " UTC", ""),
                       format="%Y-%m-%d %H:%M:%S", unit='s')


def filter_fare_amount(e: ds.Expression) -> ds.Expression:
    return (e > 1) & (e < 200)


def coord_filter(expr: pc.Expression) -> pc.Expression:
    lon = (expr > -78) & (expr < -70)
    lat = (expr > 36)  & (expr < 46)
    return pc.is_valid(expr) & (lon | lat)


def coord_transform(expr: pc.Expression, swap_min: int, swap_max: int,
                    swap_expr: pc.Expression) -> pc.Expression:
    return pc.if_else(
        (expr > swap_min) & (expr < swap_max),
        swap_expr,
        expr
    ).cast(pa.float32())


def add_index_column(table: pa.Table) -> pa.Table:
    col = pa.array(range(table.shape[0]), type=pa.uint32())
    return table.add_column(0, IDX_COL, col)


def get_cleaned_table(dataset: ds.Dataset, head: tt.Optional[int] = None) -> pa.Table:
    plon = ds.field('pickup_longitude')
    plat = ds.field('pickup_latitude')
    dlon = ds.field('dropoff_longitude')
    dlat = ds.field('dropoff_latitude')

    cols = {
        'fare_amount': ds.field('fare_amount').cast(pa.float16()),
        'passenger_count': pc.if_else(
            ds.field('passenger_count') == 0,
            1, ds.field('passenger_count')
        ).cast(pa.uint8()),
        'pickup_datetime': parse_datetime(ds.field('pickup_datetime')),
        'pickup_longitude': coord_transform(plon, 36, 46, plat),
        'pickup_latitude': coord_transform(plat, -79, -70, plon),
        'dropoff_longitude': coord_transform(dlon, 36, 46, dlat),
        'dropoff_latitude': coord_transform(dlat, -79, -70, dlon),
    }
    expr_filter = (
            filter_fare_amount(ds.field('fare_amount')) &
            (ds.field("passenger_count") <= 9) &
            coord_filter(plon) &
            coord_filter(plat) &
            coord_filter(dlon) &
            coord_filter(dlat) &
            (pc.sign(plat) != pc.sign(plon))
    )
    if head is not None:
        table = dataset.head(head, columns=cols, filter=expr_filter)
    else:
        table = dataset.to_table(columns=cols, filter=expr_filter)
    return add_index_column(table)


# Feature engineering
def compute_distance(lat1, lon1,
                     lat2, lon2, earth_radius = 6371.0) -> ds.Expression:
    # convert to radians
    r_lat1 = lat1 * np.pi / 180
    r_lon1 = lon1 * np.pi / 180
    r_lat2 = lat2 * np.pi / 180
    r_lon2 = lon2 * np.pi / 180
    d_lat = r_lat2 - r_lat1
    d_lon = r_lon2 - r_lon1
    sin_dlat2 = pc.sin(d_lat / 2)
    sin_dlon2 = pc.sin(d_lon / 2)
    a = sin_dlat2*sin_dlat2 + pc.cos(r_lat1) * pc.cos(r_lat2) * sin_dlon2 * sin_dlon2
    sqrt_a = pc.sqrt(a)
    sqrt_1_a = pc.sqrt(ds.scalar(1) - a)
    c = pc.atan2(sqrt_a, sqrt_1_a) * 2
    return (c * earth_radius).cast(pa.float16())


def one_hot_encode(col_name: str, table: pa.Table,
                   drop_col: bool = True) -> pa.Table:
    col = table[col_name]
    vc = col.value_counts()
    res = table
    for key in vc.field(0):
        name = f"{col_name}_{key}"
        res = res.append_column(name, pc.equal(col, key))
    if drop_col:
        res = res.drop_columns(col_name)
    return res


def get_features_v1(dataset: ds.Dataset) -> pa.Table:
    plon = ds.field('pickup_longitude')
    plat = ds.field('pickup_latitude')
    dlon = ds.field('dropoff_longitude')
    dlat = ds.field('dropoff_latitude')

    res_plon = coord_transform(plon, 36, 46, plat)
    res_plat = coord_transform(plat, -79, -70, plon)
    res_dlon = coord_transform(dlon, 36, 46, dlat)
    res_dlat = coord_transform(dlat, -79, -70, dlon)
    res_dtime = parse_datetime(ds.field('pickup_datetime'))

    res_dist = compute_distance(
        res_plat, res_plon, res_dlat, res_dlon)

    cols = {
        'fare_amount': ds.field('fare_amount').cast(pa.float16()),
        'passenger_count': pc.if_else(
            ds.field('passenger_count') == 0,
            1, ds.field('passenger_count')
        ).cast(pa.uint8()),

        'travel_distance': res_dist,
        'trip_dow': pc.day_of_week(res_dtime).cast(pa.uint8()),
        'trip_year': pc.year(res_dtime).cast(pa.uint16()),
        'trip_month': pc.month(res_dtime).cast(pa.uint8()),
        'trip_day': pc.day(res_dtime).cast(pa.uint8()),
        'trip_hour': pc.hour(res_dtime).cast(pa.uint8()),
    }

    for land_name, (lat, lon) in LANDMARKS.items():
        for pt_name, (elat, elon) in (
                ("dropoff", (res_dlat, res_dlon)),
                ("pickup", (res_plat, res_plon))
        ):
            cols[f"{pt_name}_{land_name}_dist"] = compute_distance(
                elat, elon, ds.scalar(lat), ds.scalar(lon)
            )

    expr_filter = (
            filter_fare_amount(ds.field('fare_amount')) &
            (ds.field("passenger_count") <= 9) &
            coord_filter(plon) &
            coord_filter(plat) &
            coord_filter(dlon) &
            coord_filter(dlat) &
            (pc.sign(plat) != pc.sign(plon)) &
            pc.less(res_dist.cast(pa.float32()), 50)
    )

    table_full = dataset.to_table(columns=cols, filter=expr_filter)
    return add_index_column(table_full)


def normalize_table(table: pa.Table) -> pa.Table:
    res_dict = {}

    for col in table.column_names:
        col_data = table[col]
        if col == IDX_COL or col == TARGET_COL:
            res_dict[col] = col_data
            continue
        col_type = col_data.type
        if pa.types.is_boolean(col_type):
            col_data = col_data.cast(pa.uint8())
            col_type = pa.float16()
        # there is a limited support of half float, need to cast
        if col_type == pa.float16():
            col_data = col_data.cast(pa.float32())
        m_mean = pc.mean(col_data)
        n = pc.divide(pc.subtract(col_data, m_mean), pc.stddev(col_data))
        if col_type == pa.float16():
            n = n.cast(col_type)
        else:
            n = n.cast(pa.float32())
        res_dict[col] = n
    return pa.table(res_dict)
