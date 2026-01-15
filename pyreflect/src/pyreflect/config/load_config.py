from pathlib import Path
import yaml

_default_config_files = ["settings.yaml", "settings.yml", "settings.json"]

def load_config(root):
    """

    :param root:
    :return: Path | None
        returns a Path if there is a settings.yaml or settings.yml file.
        Otherwise, returns None.
    """
    config_path = _search_for_config_in_root_dir(root)
    config_data = _parse(config_path)
    return config_data

def _search_for_config_in_root_dir(root:str|Path) ->Path | None:
    root = Path(root)
    for file in _default_config_files:
        if(root/file).is_file():
            return root/file

    return None

def _parse(config_path:str | None) -> dict:
    with open(config_path, "r",encoding="utf-8") as f:
        return yaml.safe_load(f)
