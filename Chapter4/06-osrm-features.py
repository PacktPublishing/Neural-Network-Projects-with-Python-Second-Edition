# tool to efficiently retrieve routing features from local OSRM instance
import pathlib
import argparse
import requests
import tqdm
import time
import typing as tt
from concurrent import futures
from timer_context import TimerContext

import data
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.feather as pf

ROUTER_PATH = "/route/v1/driving/"

COLS = (
    "idx", "pickup_latitude", "pickup_longitude",
    "dropoff_latitude", "dropoff_longitude"
)


def query_route_osmr(
    session: requests.Session,
    host: str,
    src_lat: float, src_lon: float,
    dst_lat: float, dst_lon: float,
) -> tt.Optional[dict]:
    url = "http://" + host + ROUTER_PATH + f"{src_lon},{src_lat};{dst_lon},{dst_lat}"
    res = session.get(url)
    if res.status_code != 200:
        # some points almost immediate to each other, then osrm fails
        return {
            'distance': 0.0,
            'duration': 0.0,
        }
    data = res.json()
    route = data['routes'][0]
    result = {
        'distance': route['distance'],
        'duration': route['duration'],
    }
    return result


def batch_query_route(host: str, batch: pa.RecordBatch, out_path: str) -> tt.List[int]:
    res_idx, res_dist, res_dur = [], [], []
    data_dict = batch.select(COLS).to_pydict()
    errors = []
    with requests.Session() as session:
        for idx, plat, plon, dlat, dlon in zip(*map(lambda n: data_dict[n], COLS)):
            # retries
            for _ in range(10):
                try:
                    res = query_route_osmr(session, host, plat, plon, dlat, dlon)
                    break
                except requests.exceptions.RequestException:
                    time.sleep(0.5)
                    continue
            if res is None:
                errors.append(idx)
                continue
            res_idx.append(idx)
            res_dist.append(res['distance'])
            res_dur.append(res['duration'])
    if not errors:
        table = pa.table({
            "idx": res_idx,
            "route_distance": res_dist,
            "route_duration": res_dur,
        })
        pf.write_feather(table, out_path)
    return errors


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost:5050", help="Host and port of OSRM instance, default=localhost:5050")
    parser.add_argument("-n", "--num-workers", type=int, default=12, help="Count of parallel workers, default=12")
    parser.add_argument("-o", "--output", required=True, help="Name of the feather file to write")
    parser.add_argument("--head", type=int, help="If given, process only this amount of rows, default=No limit")
    parser.add_argument("-t", "--tmp-dir", default="tmp-osrm", help="Name of the temporary dir, default=tmp-osrm")
    args = parser.parse_args()

    tmp_path = pathlib.Path(args.tmp_dir)
    tmp_path.mkdir(parents=True, exist_ok=True)

    dataset = data.get_dataset()
    table_full = data.get_cleaned_table(dataset, head=args.head)

    batches = table_full.to_batches()
    print(f"Table shape={table_full.shape}, batches={len(batches)}")

    with futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        jobs, done = [], []
        with TimerContext() as timer:
            for idx, batch in enumerate(tqdm.tqdm(batches)):
                out_name = tmp_path / f"tmp_{idx:06d}.feather"
                if out_name.exists():
                    continue
                job = executor.submit(batch_query_route, args.host, batch, str(out_name))
                jobs.append(job)
                if len(jobs) == args.num_workers:
                    res = futures.wait(jobs, return_when=futures.FIRST_COMPLETED)
                    done.extend(res.done)
                    jobs = list(res.not_done)
                    for j in res.done:
                        indices = j.result()
                        if indices:
                            print("Indices: ", str(indices))
                            break
            if jobs:
                print(f"Finally waiting for {len(jobs)} jobs")
                res = futures.wait(jobs)
                done.extend(res.done)
            print(f"Done in {timer.duration}")
            for j in done:
                error_indices = j.result()
                if not error_indices:
                    continue
                print("Errors: " + str(error_indices))
    files = list(tmp_path.glob("tmp_*.feather"))
    if len(files) == len(batches):
        print("Combining feather files into " + args.output)
        out_batches = []
        for f in files:
            t = pf.read_table(f)
            out_batches.extend(t.to_batches())
        out_t = pa.Table.from_batches(out_batches)
        print("Final shape: ", out_t.shape)
        pf.write_feather(out_t, args.output)
