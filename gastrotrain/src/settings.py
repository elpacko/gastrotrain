import toml


def __init__():
    pass


def get_settings():
    parsed_toml = toml.load("../etc/settings.toml")
    return parsed_toml
