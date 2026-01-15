import json

import numpy as np

from vidur.types.event_type import EventType
from vidur.types.replica_id import ReplicaId


class JsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ReplicaId):
            return str(obj)
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, EventType):
            return str(obj)
        # Handle numpy types
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
