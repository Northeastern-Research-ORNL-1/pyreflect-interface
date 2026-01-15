class DataPathMissingError(ValueError):
    """Data for model training missing error."""

    def __init__(self, path)->None:
        msg = f"Data path is required for model training"
        super().__init__(msg)

class ConfigMissingKeyError(ValueError):
    """Settings.yml missing required key."""

    def __init__(self, keys)->None:
        msg = f"Settings.yml missing required key: {keys}"
        super().__init__(msg)

