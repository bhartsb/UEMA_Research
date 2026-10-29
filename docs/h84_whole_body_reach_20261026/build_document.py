"""Current report builder; rejected earlier builder retained in review evidence."""
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).with_name("revise_document.py")), run_name="__main__")
