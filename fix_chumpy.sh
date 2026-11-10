#!/bin/bash

# 1. Get the site-packages path for the current environment
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
CHUMPY_DIR="$SITE_PACKAGES/chumpy"

echo "Detected site-packages at: $SITE_PACKAGES"

if [ ! -d "$CHUMPY_DIR" ]; then
    echo "Error: chumpy directory not found. Please run 'pip install chumpy' first."
    exit 1
fi

# 2. Patch chumpy/__init__.py (Line 11 NumPy imports)
INIT_FILE="$CHUMPY_DIR/__init__.py"
echo "Patching $INIT_FILE..."

# Replace the specific import line with a modern-compatible block
sed -i 's/from numpy import bool, int, float, complex, object, unicode, str, nan, inf/import numpy as np\nbool = np.bool_\nint = np.int_\nfloat = np.float_\ncomplex = np.complex_\nobject = np.object_\nunicode = np.str_\nstr = np.str_\nfrom numpy import nan, inf/' "$INIT_FILE"

# 3. Patch chumpy/ch.py (inspect.getargspec -> inspect.getfullargspec)
CH_FILE="$CHUMPY_DIR/ch.py"
echo "Patching $CH_FILE..."

# Global replace of getargspec with getfullargspec
sed -i 's/inspect.getargspec/inspect.getfullargspec/g' "$CH_FILE"

echo "------------------------------------------------"
echo "Patches applied successfully!"
echo "Verifying..."
python -c "import chumpy; print('Verification: Chumpy imported without errors.')"