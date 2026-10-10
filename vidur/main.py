import os
import sys

# Plots are only saved to files. A GUI backend (macOS default) can hang in AppKit initialisation when the
# simulator runs as a background subprocess (seen 2026-10-10), so never open one.
os.environ.setdefault("MPLBACKEND", "Agg")

from vidur.config import SimulationConfig
from vidur.simulator import Simulator
from vidur.utils.random import set_seeds


def main() -> None:
    config: SimulationConfig = SimulationConfig.create_from_cli_args()

    set_seeds(config.seed)

    simulator = Simulator(config)
    simulator.run()

    # Learning schedulers (RL_CACHE_AWARE, LEARNED) persist their network at the end. Saved first, and outputs are
    # written even if the save fails, so neither failure loses the other's result (either still exits non-zero).
    save = getattr(simulator._scheduler, "save_final_checkpoint", None)
    try:
        if callable(save):
            save()
    finally:
        simulator.write_output()

    if simulator.stuck_requests:  # end state written to stuck_requests.json; never exit 0 with lost requests
        sys.exit(3)


if __name__ == "__main__":
    main()
