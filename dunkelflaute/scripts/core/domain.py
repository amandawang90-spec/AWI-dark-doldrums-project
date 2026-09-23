"""
Germany (and, later, Korea) domain masks on the TCo1279-DART native grid.

The land/sea split and the grid's lat/lon are both taken from
wind/data/results/tco1279_era5_regrid_index.npz, precomputed by the wind
reconstruction (ERA5 lsm regridded onto TCo1279, nearest-neighbour). The
solar and wind reconstructed files share exactly this grid and cell
ordering (verified bit-for-bit identical lat/lon arrays), so the same
mask indexes both.

The grid's cell index is spatially coherent (not a shuffled/global-random
order), so a bounding box selects a *contiguous* range of cell indices.
Reading that contiguous range touches only the few file chunks it spans,
which is ~20x faster than fancy-indexing a scattered boolean mask directly
against the netCDF variable (the chunk layout is [n_time, 100000], i.e.
each chunk already holds the full time axis, so any read into a chunk
pays for decompressing all of it -- better to read one contiguous slice
than scatter across many chunks).
"""
import numpy as np

REGRID_INDEX = "wind/data/results/tco1279_era5_regrid_index.npz"

# Bounding boxes from the project proposal, section 3.
BBOX_GERMANY = dict(lat_min=47.0, lat_max=55.0, lon_min=6.0, lon_max=15.0)
BBOX_KOREA = dict(lat_min=33.0, lat_max=39.0, lon_min=124.0, lon_max=132.0)


class DomainMasks:
    """Full-grid lat/lon/land plus a bounding-box's onshore/offshore masks."""

    def __init__(self, bbox, regrid_index_path=REGRID_INDEX):
        d = np.load(regrid_index_path)
        self.lat = d["tco_lat"]
        self.lon = d["tco_lon"]
        self.land = d["land"]

        self.bbox = (
            (self.lat >= bbox["lat_min"]) & (self.lat <= bbox["lat_max"]) &
            (self.lon >= bbox["lon_min"]) & (self.lon <= bbox["lon_max"])
        )
        self.onshore = self.bbox & self.land
        self.offshore = self.bbox & ~self.land

        idx = np.where(self.bbox)[0]
        self.lo, self.hi = int(idx.min()), int(idx.max()) + 1
        # Masks local to the [lo:hi) contiguous read window.
        self.bbox_local = self.bbox[self.lo:self.hi]
        self.onshore_local = self.onshore[self.lo:self.hi]
        self.offshore_local = self.offshore[self.lo:self.hi]

    def read_window(self, ncvar):
        """Read the contiguous [lo:hi) cell window for one time-varying netCDF variable."""
        return ncvar[:, self.lo:self.hi]
