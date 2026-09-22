"""Explicit process-local TypeSafe credentials; no machine defaults or logging."""
import os
from pathlib import Path


def credential_status():
    """Report availability without reading a credential file or returning its path."""
    filename = os.environ.get('TYPESAFE_API_KEY_FILE')
    if filename:
        return {'source': 'TYPESAFE_API_KEY_FILE', 'available': Path(filename).expanduser().is_file()}
    return {'source': 'TYPESAFE_API_KEY', 'available': bool(os.environ.get('TYPESAFE_API_KEY'))}


def read_key():
    filename = os.environ.get('TYPESAFE_API_KEY_FILE')
    if filename:
        try:
            key = Path(filename).expanduser().read_text(encoding='utf-8-sig').strip()
        except OSError:
            raise ValueError('Cannot read TYPESAFE_API_KEY_FILE') from None
    else:
        key = os.environ.get('TYPESAFE_API_KEY', '').strip()
    if key.startswith('TYPESAFE_API_KEY='):
        key = key.split('=', 1)[1].strip().strip('"\'')
    if not key or any(c.isspace() for c in key):
        raise ValueError('Set TYPESAFE_API_KEY_FILE or TYPESAFE_API_KEY to a single API key')
    return key
