from vidur.config import SimulationConfig
from vidur.simulator import Simulator
from vidur.utils.random import set_seeds


def main() -> None:
    config: SimulationConfig = SimulationConfig.create_from_cli_args()

    set_seeds(config.seed)

    simulator = Simulator(config)
    simulator.run()
    
    # Learning schedulers (RL_CACHE_AWARE, LEARNED) persist their network at the end.
    save = getattr(simulator._scheduler, "save_final_checkpoint", None)
    if callable(save):
        save()


if __name__ == "__main__":
    main()
