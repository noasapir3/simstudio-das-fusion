import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Set macOS dock / menu-bar name to "Optical Fiber SIM"
if sys.platform == "darwin":
    try:
        from Foundation import NSBundle  # type: ignore
        bundle = NSBundle.mainBundle()
        info = bundle.localizedInfoDictionary() or bundle.infoDictionary()
        if info is not None:
            info["CFBundleName"] = "Optical Fiber SIM"
    except Exception:
        pass

from simstudio.gui.app import run  # GUI layer is now under simstudio/gui/

if __name__ == "__main__":
    run()
