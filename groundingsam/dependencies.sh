# Install dependencies
pip install -q -e .
pip uninstall -y supervision
pip install -q supervision==0.6.0