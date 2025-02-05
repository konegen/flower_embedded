import yaml


def load_config(yaml_path="config.yaml"):
    """Lädt die Konfigurationsdatei im YAML-Format."""
    with open(yaml_path, "r") as file:
        config = yaml.safe_load(file)
    return config.get("flwr_app_config", {})  # Nur den relevanten Abschnitt zurückgeben
