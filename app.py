#!/usr/bin/env python3
"""
Download NASA GISTEMP v4, re-reference 2000–present monthly anomalies to the
1901-2000 mean, and write lazy-load shards for the Leaflet timeline.

Usage:
    python3 -m pip install -r requirements.txt
    python3 app.py
    python3 app.py --force
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr

GISTEMP_URL = (
    "https://data.giss.nasa.gov/pub/gistemp/gistemp1200_GHCNv4_ERSSTv5.nc.gz"
)
SCRIPT_DIR = Path(__file__).resolve().parent
GZ_PATH = SCRIPT_DIR / "gistemp1200_GHCNv4_ERSSTv5.nc.gz"
NC_PATH = SCRIPT_DIR / "gistemp1200_GHCNv4_ERSSTv5.nc"
META_PATH = SCRIPT_DIR / "anomalies.json"
LATEST_PATH = SCRIPT_DIR / "anomalies-latest.bin"
SHARD_DIR = SCRIPT_DIR / "anomalies"

BASELINE_START = "1901-01-01"
BASELINE_END = "2000-12-31"
SERIES_START = "2000-01-01"
NATIVE_BASELINE = "1951-1980"
TARGET_BASELINE = "1901-2000"
SCALE = 0.1
FILL = np.int16(-32768)


def download_gistemp(force: bool = False) -> Path:
    if force:
        for path in (GZ_PATH, NC_PATH):
            if path.exists():
                path.unlink()
                print(f"Removed cached file: {path.name}")

    if not GZ_PATH.exists():
        print(f"Downloading GISTEMP v4 from {GISTEMP_URL}")
        try:
            with requests.get(GISTEMP_URL, stream=True, timeout=120) as response:
                response.raise_for_status()
                total = int(response.headers.get("content-length") or 0)
                downloaded = 0
                with GZ_PATH.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 256):
                        if not chunk:
                            continue
                        handle.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            pct = 100 * downloaded / total
                            print(
                                f"  {downloaded / 1e6:.1f} / {total / 1e6:.1f} MB ({pct:.0f}%)",
                                end="\r",
                            )
            print(f"\nSaved {GZ_PATH.name}")
        except requests.RequestException as exc:
            if GZ_PATH.exists():
                GZ_PATH.unlink()
            raise RuntimeError(f"Failed to download GISTEMP: {exc}") from exc
    else:
        print(f"Using cached download: {GZ_PATH.name}")

    if force or not NC_PATH.exists():
        print(f"Decompressing {GZ_PATH.name} -> {NC_PATH.name}")
        try:
            with gzip.open(GZ_PATH, "rb") as src, NC_PATH.open("wb") as dest:
                shutil.copyfileobj(src, dest)
        except OSError as exc:
            raise RuntimeError(f"Failed to decompress GISTEMP NetCDF: {exc}") from exc
    else:
        print(f"Using cached NetCDF: {NC_PATH.name}")

    return NC_PATH


def lon_to_180(lon: np.ndarray) -> np.ndarray:
    return ((lon.astype(np.float64) + 180.0) % 360.0) - 180.0


def area_weighted_mean(da: xr.DataArray) -> xr.DataArray:
    weights = np.cos(np.deg2rad(da["lat"]))
    return da.weighted(weights).mean(dim=("lat", "lon"), skipna=True)


def quantize(values: np.ndarray) -> np.ndarray:
    scaled = np.rint(values / SCALE)
    scaled = np.clip(scaled, -32767, 32767)
    out = np.where(np.isfinite(values), scaled, FILL).astype(np.int16)
    return np.ascontiguousarray(out)


def write_int16(path: Path, array: np.ndarray) -> None:
    path.write_bytes(np.ascontiguousarray(array, dtype=np.int16).tobytes())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build lazy-load GISTEMP departure shards for the timeline map."
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--start", default=SERIES_START, help="First month YYYY-MM-DD")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        nc_path = download_gistemp(force=args.force)
        print(f"Opening {nc_path.name}...")
        with xr.open_dataset(nc_path) as ds:
            if "tempanomaly" not in ds:
                raise RuntimeError(
                    f"Expected variable 'tempanomaly'; found {list(ds.data_vars)}"
                )
            da = ds["tempanomaly"]
            print(
                f"Loaded tempanomaly from {pd.Timestamp(da.time.values[0]):%Y-%m} "
                f"to {pd.Timestamp(da.time.values[-1]):%Y-%m}"
            )

            print(f"Computing {TARGET_BASELINE} calendar-month climatology...")
            baseline = da.sel(time=slice(BASELINE_START, BASELINE_END))
            climatology = baseline.groupby("time.month").mean("time")
            recent = da.sel(time=slice(args.start, None))
            if recent.time.size == 0:
                raise RuntimeError(f"No GISTEMP months found at or after {args.start}.")
            print(
                f"Re-referencing {recent.time.size} months "
                f"({pd.Timestamp(recent.time.values[0]):%Y-%m} to "
                f"{pd.Timestamp(recent.time.values[-1]):%Y-%m})..."
            )
            anomaly = recent.groupby("time.month") - climatology

            lons = lon_to_180(anomaly["lon"].values)
            order = np.argsort(lons)
            anomaly = anomaly.isel(lon=order)
            lons = lons[order]
            lats = anomaly["lat"].values.astype(np.float64)

            globals_da = area_weighted_mean(anomaly)
            times = [pd.Timestamp(t).strftime("%Y-%m") for t in anomaly.time.values]
            global_anomaly = [
                None if not np.isfinite(v) else round(float(v), 2)
                for v in globals_da.values
            ]

            values = anomaly.transpose("time", "lat", "lon").values.astype(np.float32)
            quantized = quantize(values)

        nlat, nlon = quantized.shape[1], quantized.shape[2]
        SHARD_DIR.mkdir(parents=True, exist_ok=True)
        shards = {}
        years = sorted({stamp[:4] for stamp in times})
        for year in years:
            idxs = [i for i, stamp in enumerate(times) if stamp.startswith(year)]
            path = SHARD_DIR / f"y{year}.bin"
            write_int16(path, quantized[idxs[0] : idxs[-1] + 1])
            shards[year] = f"anomalies/y{year}.bin"
            print(f"Wrote {path.name} ({len(idxs)} months, {path.stat().st_size / 1024:.0f} KB)")

        for leftover in SHARD_DIR.glob("y*.bin"):
            year = leftover.stem[1:]
            if year not in shards:
                leftover.unlink()
                print(f"Removed stale shard {leftover.name}")

        write_int16(LATEST_PATH, quantized[-1])
        print(f"Wrote {LATEST_PATH.name} ({LATEST_PATH.stat().st_size / 1024:.1f} KB)")

        meta = {
            "baseline": TARGET_BASELINE,
            "native_baseline": NATIVE_BASELINE,
            "source": "NASA GISTEMP v4 (GHCN-M v4 + ERSST v5, 1200 km smoothing)",
            "source_url": GISTEMP_URL,
            "scale": SCALE,
            "fill": int(FILL),
            "times": times,
            "global_anomaly": global_anomaly,
            "grid": {
                "lat0": float(lats[0]),
                "lon0": float(lons[0]),
                "step": float(round(lats[1] - lats[0], 4)) if nlat > 1 else 2.0,
                "nlat": int(nlat),
                "nlon": int(nlon),
            },
            "latest": {"time": times[-1], "file": LATEST_PATH.name},
            "shards": shards,
        }
        META_PATH.write_text(json.dumps(meta, separators=(",", ":")), encoding="utf-8")
        print(f"Wrote {META_PATH.name} ({META_PATH.stat().st_size / 1024:.1f} KB, {len(times)} months)")
        print("View the map with:")
        print(f"  python3 -m http.server 8000 --directory {SCRIPT_DIR}")
        print("  then open http://localhost:8000/index.html")
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
