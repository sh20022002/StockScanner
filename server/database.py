"""
All database actions.

Connection settings come from the environment — see .env.example. There are no
credentials in this file: it previously carried a real password in a comment,
which is now in the git history and should be treated as compromised and rotated.

Models are stored via model_store, not raw pickle.
"""
import os
from urllib.parse import quote_plus

from pymongo import MongoClient

import model_store

# Default to loopback. The old default published the database on every interface
# with no authentication at all.
DB_HOST = os.getenv('DB_HOST', '127.0.0.1')
DB_PORT = int(os.getenv('DB_PORT', 27017))
DB_NAME = os.getenv('DB_NAME', 'SmartTraid')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_AUTH_SOURCE = os.getenv('DB_AUTH_SOURCE', 'admin')

_client = None


def _uri() -> str:
    if DB_USER and DB_PASSWORD:
        user = quote_plus(DB_USER)
        pwd  = quote_plus(DB_PASSWORD)
        return (f'mongodb://{user}:{pwd}@{DB_HOST}:{DB_PORT}/'
                f'{quote_plus(DB_NAME)}?authSource={quote_plus(DB_AUTH_SOURCE)}')
    return f'mongodb://{DB_HOST}:{DB_PORT}/'


def get_client() -> MongoClient:
    """
    Lazily connect, so importing this module does not require a running database.

    The old module-level MongoClient() meant anything that imported database.py
    — including the test suite — tried to open a socket at import time.
    """
    global _client
    if _client is None:
        if not (DB_USER and DB_PASSWORD):
            print('[database] DB_USER/DB_PASSWORD not set — connecting without '
                  'authentication. Do not do this outside a local sandbox.')
        _client = MongoClient(_uri(), serverSelectionTimeoutMS=5000)
    return _client


def get_db():
    return get_client()[DB_NAME]


def _companies():
    return get_db()['stocks']


def _models():
    return get_db()['models']


# ---------------------------------------------------------------------------
# Companies
# ---------------------------------------------------------------------------

def remove_from_db(symbol):
    """Removes a stock from the database based on its symbol."""
    return _companies().delete_one({'symbol': symbol}).deleted_count


def save_company(company):
    """Saves a company's information to the database."""
    _companies().update_one(
        {'symbol': company.symbol},
        {'$set': {
            'name':              company.compeny_name,
            'symbol':            company.symbol,
            'Gics_Sector':       company.GICS_Sector,
            'Gics_Sub_Industry': company.GICS_Sub_Industry,
            'CIK':               company.CIK,
            'Founded':           company.Founded,
            'Location':          company.Location,
        }},
        upsert=True,
    )
    return True


def get_company(symbol):
    """Returns the company information for a given symbol."""
    return _companies().find_one({'symbol': symbol})


def get_companies():
    """
    Returns all companies in the database, as a list.

    The old version assigned to the name it was reading from
    (`compenies = compenies.find()`), making the global a local and raising
    UnboundLocalError on every single call.
    """
    return list(_companies().find())


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def save_hmm_model(symbol, interval, model, update):
    """Stores (or replaces) the HMM model for a symbol and interval."""
    _models().update_one(
        {'kind': 'hmm', 'symbol': symbol, 'interval': interval},
        {'$set': {'model': model_store.dumps(model), 'last_update': update}},
        upsert=True,
    )
    return True


def get_hmm_model(symbol, interval):
    """Returns the stored HMM record for a symbol and interval, or None."""
    return _models().find_one({'kind': 'hmm', 'symbol': symbol, 'interval': interval})


# update_hmm_model was identical to save_hmm_model once the upsert was in place,
# and its own version passed a single argument to update_one() — which raises
# TypeError, so it could never have updated anything.
update_hmm_model = save_hmm_model


def save_model(symbol, interval, model, update):
    """
    Stores the next-close regression model for a symbol and interval.

    Keyed per symbol, not per interval. The old "master model" scheme kept one
    model per interval and refitted it on whichever symbol was asked for last,
    so every prediction used a model trained on some other company.
    """
    _models().update_one(
        {'kind': 'regressor', 'symbol': symbol, 'interval': interval},
        {'$set': {'model': model_store.dumps(model),
                  'last_update': update,
                  'trained_on': symbol}},
        upsert=True,
    )
    return True


def get_model(symbol, interval):
    """Returns the stored regression record for a symbol and interval, or None."""
    return _models().find_one({'kind': 'regressor', 'symbol': symbol, 'interval': interval})


update_model = save_model


def load_model(record):
    """Verify and deserialise the model inside a stored record."""
    if not record or 'model' not in record:
        return None
    return model_store.loads(record['model'])
