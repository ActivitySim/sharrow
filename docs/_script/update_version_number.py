import os
from pathlib import Path

from ruamel.yaml import YAML

import sharrow as sh

print(f"updating version number to {sh.__version__}")

config_file = Path(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "_config.yml",
    )
)

yaml = YAML(typ="rt")  # default, if not specfied, is 'rt' (round-trip)
content = yaml.load(config_file)
content["title"] = f"v{sh.__version__}"
yaml.dump(content, config_file)
