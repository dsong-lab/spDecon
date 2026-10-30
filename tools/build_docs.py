"""Stage executed notebooks and build the GitHub Pages site."""
from pathlib import Path
import shutil,subprocess,sys
root=Path(__file__).resolve().parents[1]
dest=root/'docs/source/notebooks'
dest.mkdir(exist_ok=True)
for path in (root/'tutorials').glob('*.ipynb'):
    shutil.copy2(path,dest/path.name)
subprocess.run([sys.executable,'-m','sphinx','-W','--keep-going','-b','html',str(root/'docs/source'),str(root/'docs/build/html')],check=True)
(root/'docs/build/html/.nojekyll').touch()
