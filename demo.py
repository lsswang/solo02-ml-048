import json
from pathlib import Path
import align
import mosaic


if __name__ == "__main__":
    cases = json.loads((Path(__file__).parent / "fixtures/cases.json").read_text())
    fit = align.fit(**cases["before_registration"])
    print(json.dumps({"fit": fit, "mosaic": mosaic.compose(
        cases["coupling_request"], [fit["matrix"], cases["identity"]])}, indent=2))
