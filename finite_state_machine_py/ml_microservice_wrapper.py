
import requests

microservice_url = "http://localhost:8090"


def get_type_of_sentence(user_input):
    r = requests.get(microservice_url + '/type_of_sentence', params={'sentence':user_input})
    return r.json()['type']

def get_entities(user_input):
    r = requests.get(microservice_url + '/entities', params={'sentence':user_input})
    return r.json()['entities']

def get_tags(user_input):
    r = requests.get(microservice_url + '/tag', params={'sentence':user_input})
    return r.json()['pos']

def get_sentiment(user_input):
    r = requests.get(microservice_url + '/sentiment', params={'sentence':user_input})
    return r.json()['sentiment']

def get_phrase_similarity(user_input):
    r = requests.get(microservice_url + '/similarity', params={'sentence':user_input})
    return r.json()['similarity']

def parse(user_input):
    r = requests.get(microservice_url + '/similarity', params={'sentence':user_input})
    return r.json()
