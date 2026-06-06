from __future__ import annotations

import yaml
import logging
from .interfaces import AlgorithmContext, MarketDataRequest
from ..data.repository import DataRepository
from ..connectors.registry import MARKET_CONNECTOR_REGISTRY
from ..algorithms.registry import get_algorithm_class

logger = logging.getLogger(__name__)

class Walbot:
    def __init__(self, config_path: str = "config/connectors.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.repository = DataRepository(self.config, MARKET_CONNECTOR_REGISTRY)
        self.algorithms = self._init_algorithms()

    def _init_algorithms(self):
        algo_configs = self.config.get("algorithms", {})
        algos = []
        for algorithm_id, algorithm_config in algo_configs.items():
            try:
                algos.append(get_algorithm_class(algorithm_id).from_config(algorithm_config or {}))
            except KeyError:
                logger.warning("Ignoring unregistered algorithm %s", algorithm_id)
        return algos

    def run_once(self):
        logger.info("Starting bot iteration...")
        
        for algo in self.algorithms:
            # 1. Fetch required data for the algorithm
            # (In a real scenario, the algo would define its requirements)
            request = MarketDataRequest(
                symbols=algo.config.get("symbols", []),
                timeframe="1d",
                category="market_data"
            )
            bars = self.repository.fetch_market_bars(request)
            
            # 2. Build Context
            context = AlgorithmContext(
                config=algo.config,
                bars_by_symbol=bars,
                equity=100000.0, # Mock equity
                account_id="mock_account"
            )
            
            # 3. Generate signals
            signals = algo.generate_signals(context)
            
            # 4. Process signals
            for signal in signals:
                logger.info(f"Signal from {algo.algorithm_id}: {signal}")
                # Here you would call the OrderService to execute the trade

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    bot = Walbot()
    bot.run_once()
