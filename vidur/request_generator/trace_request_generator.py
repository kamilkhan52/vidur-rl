import json
import logging
from typing import Optional

import pandas as pd

from vidur.config import TraceRequestGeneratorConfig
from vidur.entities import Request
from vidur.request_generator.base_request_generator import BaseRequestGenerator

logger = logging.getLogger(__name__)


class TraceRequestGenerator(BaseRequestGenerator):
    """
    Reads a trace csv file containing request arrival time, its prompt and completion token values to generate
    inter-request times, number of tokens.
    """

    def __init__(self, config: TraceRequestGeneratorConfig):
        super().__init__(config)

        self.num_requests_generated = 0
        # load into a pd dataframe
        self.trace_df = pd.read_csv(config.trace_file)

        # scale prefill and decode tokens
        self.trace_df["num_prefill_tokens"] = (
            self.trace_df["num_prefill_tokens"] * config.prefill_scale_factor
        )
        self.trace_df["num_decode_tokens"] = (
            self.trace_df["num_decode_tokens"] * config.decode_scale_factor
        )

        # make sure all the prefill and decode counts are integers
        self.trace_df["num_prefill_tokens"] = self.trace_df[
            "num_prefill_tokens"
        ].astype(int)
        self.trace_df["num_decode_tokens"] = self.trace_df["num_decode_tokens"].astype(
            int
        )

        # make sure that there is at least one prefill and decode token
        self.trace_df["num_prefill_tokens"] = self.trace_df["num_prefill_tokens"].clip(
            lower=1
        )
        self.trace_df["num_decode_tokens"] = self.trace_df["num_decode_tokens"].clip(
            lower=1
        )

        # Filter out requests that exceed max_tokens (if specified)
        # This must happen BEFORE setting num_requests to ensure correct count
        if config.max_tokens is not None:
            total_tokens = self.trace_df["num_prefill_tokens"] + self.trace_df["num_decode_tokens"]
            num_before = len(self.trace_df)
            self.trace_df = self.trace_df[total_tokens <= config.max_tokens].reset_index(drop=True)
            num_filtered = num_before - len(self.trace_df)
            if num_filtered > 0:
                logger.warning(
                    f"Filtered {num_filtered} requests exceeding max_tokens={config.max_tokens}"
                )

        # Optional arrival jitter (trace seconds, before time scaling). Spreads each
        # request uniformly over [t, t + jitter) with the generator's seeded RNG and
        # re-sorts stably. With jitter <= the trace's bucket width this only reorders
        # requests inside a bucket. Paired across schedulers when the seed is shared.
        if config.arrival_jitter and config.arrival_jitter > 0:
            offsets = self._random_number_generator.uniform(
                0.0, config.arrival_jitter, size=len(self.trace_df)
            )
            self.trace_df["arrived_at"] = self.trace_df["arrived_at"] + offsets
            self.trace_df = self.trace_df.sort_values(
                "arrived_at", kind="mergesort"
            ).reset_index(drop=True)
            logger.info(
                f"Applied arrival jitter U[0, {config.arrival_jitter}) s (seed {config.seed})"
            )

        # Set num_requests AFTER filtering to ensure we have enough requests
        self.num_requests = (
            config.num_requests
            if config.num_requests is not None
            else len(self.trace_df)
        )
        if self.num_requests > len(self.trace_df):
            logger.warning(
                f"Requested {self.num_requests} requests but only {len(self.trace_df)} available after filtering. "
                f"Using {len(self.trace_df)} requests."
            )
            self.num_requests = len(self.trace_df)

        # rescale the time to change QPS
        self.trace_df["arrived_at"] = (
            self.trace_df["arrived_at"] * config.time_scale_factor
        )

        # Preprocess block_hash_ids, block_size
        if "block_hash_ids" in self.trace_df.columns:
            self.trace_df["block_hash_ids"] = self.trace_df["block_hash_ids"].apply(
                json.loads
            )
        else:
            self.trace_df["block_hash_ids"] = None

        if "block_size" not in self.trace_df.columns:
            self.trace_df["block_size"] = None

        # Shim session_id to None if not present
        if "session_id" not in self.trace_df.columns:
            self.trace_df["session_id"] = None
        else:
            self.trace_df["session_id"] = self.trace_df["session_id"].astype(int)

        logger.info(
            f"Loaded trace file {config.trace_file} with {len(self.trace_df)} requests"
        )
        # compute pd ratio and log the 25, 50, 75, 90, 95, 99 percentiles
        pd_ratio = (
            self.trace_df["num_prefill_tokens"] / self.trace_df["num_decode_tokens"]
        )
        logger.debug(
            f"Prompt/decode token ratio stats\n:{pd_ratio.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99])}"
        )

        # Pre-extract columns to numpy arrays/lists for fast indexing
        # This avoids expensive pandas iloc operations in the hot path
        self._arrived_at = self.trace_df["arrived_at"].to_numpy()
        self._num_prefill_tokens = self.trace_df["num_prefill_tokens"].to_numpy()
        self._num_decode_tokens = self.trace_df["num_decode_tokens"].to_numpy()
        self._block_hash_ids = self.trace_df["block_hash_ids"].tolist()
        self._block_size = self.trace_df["block_size"].tolist()
        self._session_id = self.trace_df["session_id"].tolist()

        # Free the DataFrame to save memory (we no longer need it)
        del self.trace_df

    def max_request_total_tokens(self) -> int:
        """Largest prefill+decode token count among the requests that will be replayed."""
        n = self.num_requests
        if n == 0:
            return 0
        return int((self._num_prefill_tokens[:n] + self._num_decode_tokens[:n]).max())

    def get_next_request_arrival_time(self) -> Optional[float]:
        if self.num_requests_generated >= self.num_requests:
            return None

        return float(self._arrived_at[self.num_requests_generated])

    def get_next_request(self) -> Optional[Request]:
        if self.num_requests_generated >= self.num_requests:
            return None

        idx = self.num_requests_generated
        request = Request(
            arrived_at=float(self._arrived_at[idx]),
            num_prefill_tokens=int(self._num_prefill_tokens[idx]),
            num_decode_tokens=int(self._num_decode_tokens[idx]),
            block_hash_ids=self._block_hash_ids[idx],
            block_size=self._block_size[idx],
            session_id=self._session_id[idx],
        )
        self.num_requests_generated += 1
        return request
