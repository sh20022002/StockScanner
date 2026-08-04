"""
Tests for model_store — the replacement for raw pickle.loads() on database
documents, which was straightforward remote code execution given the database
had no authentication.

Run with: pytest server/tests -v
"""
import os
import sys
import warnings

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import model_store


class Dummy:
    """A small picklable payload standing in for a fitted model."""

    def __init__(self, value):
        self.value = value

    def __eq__(self, other):
        return isinstance(other, Dummy) and other.value == self.value


@pytest.fixture
def signed(monkeypatch):
    monkeypatch.setenv('MODEL_STORE_KEY', 'a' * 64)


@pytest.fixture
def unsigned(monkeypatch):
    monkeypatch.delenv('MODEL_STORE_KEY', raising=False)


class TestRoundTrip:
    def test_signed_round_trip(self, signed):
        record = model_store.dumps(Dummy(41))
        assert record['signed'] is True
        assert model_store.loads(record) == Dummy(41)

    def test_unsigned_round_trip_warns(self, unsigned):
        record = model_store.dumps(Dummy(7))
        assert record['signed'] is False
        with pytest.warns(RuntimeWarning, match='MODEL_STORE_KEY'):
            assert model_store.loads(record) == Dummy(7)

    def test_signed_load_does_not_warn(self, signed):
        record = model_store.dumps(Dummy(1))
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            assert model_store.loads(record) == Dummy(1)


class TestIntegrity:
    def test_tampered_blob_is_rejected(self, signed):
        record = model_store.dumps(Dummy(1))
        record['blob'] = record['blob'] + b'\x00'
        with pytest.raises(ValueError, match='digest mismatch'):
            model_store.loads(record)

    def test_tampered_digest_is_rejected(self, signed):
        record = model_store.dumps(Dummy(1))
        record['digest'] = '0' * 64
        with pytest.raises(ValueError, match='digest mismatch'):
            model_store.loads(record)

    def test_missing_digest_is_rejected(self, signed):
        record = model_store.dumps(Dummy(1))
        del record['digest']
        with pytest.raises(ValueError, match='missing its digest'):
            model_store.loads(record)

    def test_record_signed_with_another_key_is_rejected(self, monkeypatch):
        monkeypatch.setenv('MODEL_STORE_KEY', 'k1' * 16)
        record = model_store.dumps(Dummy(1))
        monkeypatch.setenv('MODEL_STORE_KEY', 'k2' * 16)
        with pytest.raises(ValueError, match='digest mismatch'):
            model_store.loads(record)

    def test_unsigned_record_rejected_once_a_key_is_set(self, monkeypatch):
        """
        An attacker who wrote a plain-sha256 record cannot have it loaded by an
        instance that has a signing key configured.
        """
        monkeypatch.delenv('MODEL_STORE_KEY', raising=False)
        record = model_store.dumps(Dummy(1))
        monkeypatch.setenv('MODEL_STORE_KEY', 'k' * 32)
        with pytest.raises(ValueError, match='digest mismatch'):
            model_store.loads(record)


class TestMalformedRecords:
    def test_rejects_non_dict(self, signed):
        with pytest.raises(ValueError, match='must be a dict'):
            model_store.loads(b'raw pickle bytes')

    def test_rejects_missing_blob(self, signed):
        with pytest.raises(ValueError, match='no binary blob'):
            model_store.loads({'format': model_store.FORMAT, 'digest': 'x'})

    def test_rejects_unknown_format(self, signed):
        record = model_store.dumps(Dummy(1))
        record['format'] = 'pickle-legacy'
        with pytest.raises(ValueError, match='unsupported model format'):
            model_store.loads(record)

    def test_rejects_bare_pickle_payload(self, signed):
        """A legacy raw-pickle document must not be deserialised on trust."""
        import pickle
        with pytest.raises(ValueError):
            model_store.loads({'blob': pickle.dumps(Dummy(1))})
