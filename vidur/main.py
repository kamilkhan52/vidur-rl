from vidur.config import SimulationConfig
from vidur.simulator import Simulator
from vidur.utils.random import set_seeds


def main() -> None:
    config: SimulationConfig = SimulationConfig.create_from_cli_args()

    set_seeds(config.seed)

    simulator = Simulator(config)
    simulator.run()
    
    # Save final RL checkpoint if training mode
    from vidur.scheduler.global_scheduler.rl_cache_aware_global_scheduler import RLCacheAwareGlobalScheduler
    if isinstance(simulator._scheduler, RLCacheAwareGlobalScheduler):
        simulator._scheduler.save_final_checkpoint()


if __name__ == "__main__":
    main()
