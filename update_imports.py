
import os
import re

MAPPING = {
    'alpaca_client': 'brokerages.alpaca_client',
    'alpha_vantage': 'connectors.sentiment.alpha_vantage',
    'api_app': 'api.api_app',
    'api_payloads': 'api.api_payloads',
    'backtest': 'execution.backtest',
    'bot_runtime': 'core.bot_runtime',
    'config': 'core.config',
    'controls': 'api.controls',
    'dual_momentum_optimizer': 'algorithms.equities.dual_momentum_optimizer',
    'duckdb_store': 'data.duckdb_store',
    'live_runner': 'execution.live_runner',
    'logging_utils': 'common.logging_utils',
    'notifications': 'common.notifications',
    'orders': 'core.orders',
    'portfolio': 'core.portfolio',
    'provider_cache': 'data.provider_cache',
    'signals': 'data.signals.signals',
    'social': 'data.social',
    'state_store': 'data.state_store',
    'strategy_models': 'core.strategy_models',
    'universe': 'data.universe',
    'universe_selector': 'data.universe_selector',
    'web_app': 'api.web_app',
    # 'data' is a special case. It's now a package src.data
}

def update_content(content, file_path):
    # Update absolute imports: from src.OLD import -> from src.NEW import
    # and: import src.OLD -> import src.NEW
    # and: from src import OLD -> from src.SUBDIR import OLD
    for old, new in MAPPING.items():
        # from src.alpaca_client import ...
        content = re.sub(rf'from src\.{old}(\s| import)', rf'from src.{new}\1', content)
        # from src.alpaca_client(newline)
        content = re.sub(rf'from src\.{old}\n', rf'from src.{new}\n', content)
        # import src.alpaca_client
        content = re.sub(rf'import src\.{old}(\s|$)', rf'import src.{new}\1', content)
        # from src import alpaca_client
        if '.' in new:
            subdir, name = new.rsplit('.', 1)
            content = re.sub(rf'from src import {old}\b', rf'from src.{subdir} import {name} as {old}', content)
        else:
            # Special case for data if it was just 'data'
            pass

    # Update relative imports in src/ files
    if file_path.startswith('src/'):
        for old, new in MAPPING.items():
            content = re.sub(rf'from \.+\b{old}\b(\s| import)', rf'from src.{new}\1', content)
            
        for old, new in MAPPING.items():
            content = re.sub(rf'from \.+ import {old}\b', rf'from src.{new} import {old}', content)

    # Specific fix for src/brokerages/registry.py
    if file_path == 'src/brokerages/registry.py':
        content = content.replace('from .alpaca import AlpacaBrokerage', 'from .providers.alpaca import AlpacaBrokerage')

    return content

def main():
    for root, dirs, files in os.walk('src'):
        for file in files:
            if file.endswith('.py'):
                path = os.path.join(root, file)
                with open(path, 'r') as f:
                    content = f.read()
                new_content = update_content(content, path)
                if new_content != content:
                    with open(path, 'w') as f:
                        f.write(new_content)
                    print(f"Updated {path}")

    for root, dirs, files in os.walk('tests'):
        for file in files:
            if file.endswith('.py'):
                path = os.path.join(root, file)
                with open(path, 'r') as f:
                    content = f.read()
                new_content = update_content(content, path)
                if new_content != content:
                    with open(path, 'w') as f:
                        f.write(new_content)
                    print(f"Updated {path}")

if __name__ == '__main__':
    main()
