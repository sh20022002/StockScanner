"""
Serialisation for trained models.

Deserialising a model is arbitrary code execution: joblib, like pickle, will
happily instantiate whatever the payload names. The old code called
`pickle.loads()` directly on documents fetched from MongoDB, so anyone who could
write to the database — which, unauthenticated and published on 0.0.0.0, was
anyone who could reach the port — had remote code execution.

Two changes:

  * Payloads carry an HMAC-SHA256 signature. Set MODEL_STORE_KEY and loads()
    refuses anything not signed with that key, so a tampered document cannot be
    deserialised at all.
  * Without a key configured, the digest is a plain checksum: it detects
    corruption but proves nothing about origin, and loads() says so loudly.

Fix the database credentials too — this narrows the window, it does not remove
the fact that a model blob is executable content.
"""
import hashlib
import hmac
import io
import os
import warnings

import joblib

FORMAT = 'joblib-1'


def _key() -> bytes | None:
    secret = os.getenv('MODEL_STORE_KEY')
    return secret.encode('utf-8') if secret else None


def _digest(blob: bytes) -> tuple[str, bool]:
    """Return (hex digest, signed). Signed digests are HMACs, not bare hashes."""
    key = _key()
    if key:
        return hmac.new(key, blob, hashlib.sha256).hexdigest(), True
    return hashlib.sha256(blob).hexdigest(), False


def dumps(model) -> dict:
    """
    Serialise a model into a storable record.

    Returns:
        {'format': str, 'blob': bytes, 'digest': str, 'signed': bool}
    """
    buf = io.BytesIO()
    joblib.dump(model, buf)
    blob = buf.getvalue()
    digest, signed = _digest(blob)
    return {'format': FORMAT, 'blob': blob, 'digest': digest, 'signed': signed}


def loads(record: dict):
    """
    Verify and deserialise a model record produced by dumps().

    Raises:
        ValueError: The record is malformed, or its digest does not match.
    """
    if not isinstance(record, dict):
        raise ValueError('model record must be a dict')

    blob = record.get('blob')
    if not isinstance(blob, (bytes, bytearray)):
        raise ValueError('model record has no binary blob')

    if record.get('format') != FORMAT:
        raise ValueError(f"unsupported model format: {record.get('format')!r}")

    stored = record.get('digest')
    if not stored:
        raise ValueError('model record is missing its digest — refusing to load')

    expected, signed = _digest(bytes(blob))
    # Constant-time compare so a mismatch cannot be probed byte by byte.
    if not hmac.compare_digest(str(stored), expected):
        raise ValueError(
            'model digest mismatch — the stored model was modified, or was '
            'signed with a different MODEL_STORE_KEY. Refusing to deserialise.'
        )

    if not signed:
        warnings.warn(
            'MODEL_STORE_KEY is not set: the model digest is an unauthenticated '
            'checksum, so it detects corruption but not tampering. Anyone able '
            'to write to the model store can still execute code here.',
            RuntimeWarning, stacklevel=2,
        )

    return joblib.load(io.BytesIO(blob))
