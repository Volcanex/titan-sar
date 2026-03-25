"""Execute all CPU notebooks to save outputs, using nbconvert."""
import subprocess
import sys
from pathlib import Path

notebooks = [
    'notebooks/01_data_acquisition.ipynb',
    'notebooks/02_preprocessing_and_tiling.ipynb',
    'notebooks/03_exploratory_data_analysis.ipynb',
    'notebooks/04_traditional_ml_baseline.ipynb',
    'notebooks/05_resolution_sanity_test.ipynb',
]

for nb in notebooks:
    print(f'\n{"="*60}')
    print(f'Executing: {nb}')
    print(f'{"="*60}')
    result = subprocess.run(
        [sys.executable, '-m', 'jupyter', 'nbconvert',
         '--to', 'notebook', '--execute',
         '--ExecutePreprocessor.timeout=1800',
         '--ExecutePreprocessor.kernel_name=python3',
         '--inplace', nb],
        capture_output=True, text=True, timeout=2400,
    )
    if result.returncode == 0:
        print(f'  OK: {nb}')
    else:
        print(f'  FAILED: {nb}')
        # Print last few lines of stderr
        err_lines = result.stderr.strip().split('\n')
        for line in err_lines[-10:]:
            print(f'    {line}')
