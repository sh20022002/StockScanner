'''all database actions'''

import pymongo
import os
import pickle
from functools import wraps
from pymongo import MongoClient

from pymongo.errors import OperationFailure
from urllib.parse import quote_plus

# Fetch environment variables
db_host = os.getenv('DB_HOST', 'db')
db_port = int(os.getenv('DB_PORT', 27017))
db_user = os.getenv('DB_USER', 'root')
db_password = os.getenv('DB_PASSWORD', 'te13@t$3t')
db_name = os.getenv('DB_NAME', 'SmartTraid')

# parse the URI
db_user = quote_plus(db_user)
db_password = quote_plus(db_password)
db_name = quote_plus(db_name)

# Create the MongoDB client
mongo_uri = f"mongodb://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}?authSource=admin"
client = MongoClient(mongo_uri)

mydb = client.SmartTraid
compenies = mydb['stocks']
models = mydb['models']


#users functions



def remove_from_db(symbol):
    '''Removes a stock from the database based on its symbol.'''
    compenies.delete_one({'symbol': symbol})



def save_compeny(company):
    '''Saves a company's information to the database.'''
    compenies.insert_one({'name': company.compeny_name,
                          'symbol': company.symbol,
                          'Gics_Sector': company.GICS_Sector,
                          'Gics_Sub_Industry': company.GICS_Sub_Industry,
                          'CIK': company.CIK,
                          'Founded': company.Founded,
                          'Location': company.Location,
                          'price': company.price,
                          'sentiment': company.sentiment,
                          'summary': company.summery,})
                        #   'model_1h': company.hourly,
                        #   'model_1d': company.daily,
                        #   'last_update': company.last_update})
    return True

def get_compeny(symbol):
    '''Returns the company information for a given symbol.'''
    compeny = compenies.find_one({'symbol': symbol})
    return compeny

def get_compenies():
    '''Returns all companies in the database.'''
    compenies = compenies.find()
    return compenies

def save_model(symbol, interval, pickled_model, update):
    '''Saves the HMM model for a given stock symbol and interval.'''
    models.insert_one({'interval': interval, 'model': pickled_model,'traind':{'symbol': symbol, 'last_update': update}})
    return True

def save_hmm_model(symbol, interval, pickled_model, update):
    '''Saves the HMM model for a given stock symbol and interval.'''
    models.insert_one({'symbol': symbol, 'interval': interval, 'model': pickled_model, 'last_update': update})
    return True

def get_hmm_model(symbol, interval):
    '''Returns the HMM model for a given stock symbol and interval.'''
    model = models.find_one({'symbol': symbol, 'interval': interval})
    return model

def update_hmm_model(symbol, interval, pickled_model, update):
    '''Updates the hmm model for a given stock symbol and interval.'''
    models.update_one({'symbol': symbol, 'interval': interval, '$set':{ 'model': pickled_model, 'last_update': update}})
    return True

def get_model(interval):
    '''Returns the master model for a given interval.'''
    model = models.find_one({'interval': interval})
    return model

def update_model(symbol, interval, pickled_model, update):
    '''Updates the master model for a given interval.'''
    models.update_one({'interval': interval, '$set': {'model': pickled_model}, 'traind': {'symbol': symbol, '$set': {'last_update': update}}})
    return True