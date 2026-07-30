"""
Simple centroid-based multi-object tracker.

Why not a heavier tracker (DeepSORT, ByteTrack) for the base project?
A centroid tracker is easy to understand, has no extra dependencies, and
works well for our use case: single-direction conveyor belt motion where
objects move predictably frame-to-frame. It's also a great teaching tool
before moving to production-grade trackers.

Algorithm:
1. For each new frame's detections, compute centroids.
2. Match each new centroid to the closest existing tracked object
   (within MAX_TRACKING_DISTANCE).
3. Unmatched existing objects get a "disappeared" counter incremented.
4. Objects disappeared for too long are deregistered.
5. Unmatched new centroids become new tracked objects.

Complexity: O(N*M) per frame (N = existing tracks, M = new detections),
using a simple distance matrix. For a conveyor with a handful of products
in frame at once, this is more than fast enough.
"""

from collections import OrderedDict
import numpy as np

from src.utils.config import MAX_TRACKING_DISTANCE, MAX_DISAPPEARED_FRAMES
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CentroidTracker:
    def __init__(self, max_disappeared: int = MAX_DISAPPEARED_FRAMES,
                 max_distance: int = MAX_TRACKING_DISTANCE):
        self.next_object_id = 0
        self.objects = OrderedDict()       # object_id -> centroid (x, y)
        self.disappeared = OrderedDict()    # object_id -> frames missing
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, detections_centroids):
        """
        detections_centroids: list of (x, y) tuples for the current frame.
        Returns: OrderedDict of {object_id: centroid} for currently tracked objects.
        """
        # No detections this frame -> mark everyone as disappeared
        if len(detections_centroids) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        input_centroids = np.array(detections_centroids)

        # No existing tracks -> register everything as new
        if len(self.objects) == 0:
            for centroid in input_centroids:
                self.register(tuple(centroid))
            return self.objects

        object_ids = list(self.objects.keys())
        object_centroids = np.array(list(self.objects.values()))

        # Distance matrix: rows = existing objects, cols = new detections
        distances = np.linalg.norm(
            object_centroids[:, np.newaxis] - input_centroids[np.newaxis, :], axis=2
        )

        # Greedy matching: smallest distances first
        rows = distances.min(axis=1).argsort()
        cols = distances.argmin(axis=1)[rows]

        used_rows, used_cols = set(), set()

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if distances[row, col] > self.max_distance:
                continue  # too far apart, don't associate

            object_id = object_ids[row]
            self.objects[object_id] = tuple(input_centroids[col])
            self.disappeared[object_id] = 0

            used_rows.add(row)
            used_cols.add(col)

        unused_rows = set(range(distances.shape[0])) - used_rows
        unused_cols = set(range(distances.shape[1])) - used_cols

        # Existing objects that found no match this frame
        for row in unused_rows:
            object_id = object_ids[row]
            self.disappeared[object_id] += 1
            if self.disappeared[object_id] > self.max_disappeared:
                self.deregister(object_id)

        # New detections that matched no existing object -> new track
        for col in unused_cols:
            self.register(tuple(input_centroids[col]))

        return self.objects
