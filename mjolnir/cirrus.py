"""
Reasonable facsimile's of queries from CirrusSearch, and utils
to make those queries against an elasticsearch cluster.
"""

import requests


def _bulk_success(response):
    """Return true if all requests in a bulk request succeeded

    Parameters
    ----------
    response : requests.models.Response

    Returns
    -------
    bool
    """
    parsed = response.json()
    if 'responses' not in parsed:
        return False
    for result in parsed['responses']:
        if result['status'] != 200:
            return False
    return True


def make_request(session, url, url_list, bulk_query, num_retries=5, reuse_url=False):
    failures = 0
    while True:
        try:
            result = session.get(url + '/_msearch', data=bulk_query)
            if _bulk_success(result):
                return url, result
            last_ex = RuntimeError('Too many failures or no urls left')
        except requests.ConnectionError as e:
            last_ex = e
        failures += 1
        if failures >= num_retries:
            raise last_ex
        if not reuse_url:
            if len(url_list) == 0:
                raise last_ex
            # TODO: This is only desirable if url_list is a list of actual
            # servers. If the url_list is a loadbalancer like LVS then we
            # want to keep using the existing url.
            url = url_list.pop()


def full_text_query(query):
    return {
        "bool": {
            "minimum_should_match": 1,
            "should": [
                {
                    "bool": {
                        "filter": [
                            {
                                "bool": {
                                    "should": [
                                        {
                                            "match": {
                                                "all": {
                                                    "query": query,
                                                    "operator": "AND"
                                                }
                                            }
                                        },
                                        {
                                            "match": {
                                                "all.plain": {
                                                    "query": query,
                                                    "operator": "AND"
                                                }
                                            }
                                        }
                                    ]
                                }
                            }
                        ],
                        "should": [
                            {
                                "multi_match": {
                                    "query": query,
                                    "boost": 0.3,
                                    "minimum_should_match": "1",
                                    "type": "most_fields",
                                    "fields": [
                                        "title.plain^1",
                                        "title^3"
                                    ]
                                }
                            },
                            {
                                "multi_match": {
                                    "query": query,
                                    "boost": 0.05,
                                    "minimum_should_match": "1",
                                    "type": "most_fields",
                                    "fields": [
                                        "category.plain^1",
                                        "category^3"
                                    ]
                                }
                            },
                            {
                                "multi_match": {
                                    "query": query,
                                    "boost": 0.05,
                                    "minimum_should_match": "1",
                                    "type": "most_fields",
                                    "fields": [
                                        "heading.plain^1",
                                        "heading^3"
                                    ]
                                }
                            },
                            {
                                "multi_match": {
                                    "query": query,
                                    "boost": 0.05,
                                    "minimum_should_match": "1",
                                    "type": "most_fields",
                                    "fields": [
                                        "auxiliary_text.plain^1",
                                        "auxiliary_text^3"
                                    ]
                                }
                            },
                            {
                                "multi_match": {
                                    "query": query,
                                    "boost": 0.5,
                                    "minimum_should_match": "1",
                                    "type": "most_fields",
                                    "fields": [
                                        "file_text.plain^1",
                                        "file_text^3"
                                    ]
                                }
                            },
                            {
                                "dis_max": {
                                    "queries": [
                                        {
                                            "multi_match": {
                                                "query": query,
                                                "boost": 0.27,
                                                "minimum_should_match": "1",
                                                "type": "most_fields",
                                                "fields": [
                                                    "redirect.title.plain^1",
                                                    "redirect.title^3"
                                                ]
                                            }
                                        },
                                        {
                                            "multi_match": {
                                                "query": query,
                                                "boost": 0.2,
                                                "minimum_should_match": "1",
                                                "type": "most_fields",
                                                "fields": [
                                                    "suggest"
                                                ]
                                            }
                                        }
                                    ]
                                }
                            },
                            {
                                "dis_max": {
                                    "queries": [
                                        {
                                            "multi_match": {
                                                "query": query,
                                                "boost": 0.6,
                                                "minimum_should_match": "1",
                                                "type": "most_fields",
                                                "fields": [
                                                    "text.plain^1",
                                                    "text^3"
                                                ]
                                            }
                                        },
                                        {
                                            "multi_match": {
                                                "query": query,
                                                "boost": 0.5,
                                                "minimum_should_match": "1",
                                                "type": "most_fields",
                                                "fields": [
                                                    "opening_text.plain^1",
                                                    "opening_text^3"
                                                ]
                                            }
                                        }
                                    ]
                                }
                            }
                        ],
                        "disable_coord": True
                    }
                },
                {
                    "multi_match": {
                        "fields": [
                            "all_near_match^2"
                        ],
                        "query": query,
                    }
                }
            ]
        }
    }


def rescore():
    return {
        "window_size": 8192,
        "query": {
            "query_weight": 1,
            "rescore_query_weight": 1,
            "score_mode": "total",
            "rescore_query": {
                "function_score": {
                    "score_mode": "sum",
                    "boost_mode": "sum",
                    "functions": [
                        {
                            "script_score": {
                                "script": {
                                    "inline": "pow(doc['popularity_score'].value , 0.8) / ( pow(doc['popularity_score'].value, 0.8) + pow(8.0E-6,0.8))", # noqa:E501
                                    "lang": "expression"
                                }
                            },
                            "weight": 3
                        },
                        {
                            "script_score": {
                                "script": {
                                    "inline": "pow(doc['incoming_links'].value , 0.7) / ( pow(doc['incoming_links'].value, 0.7) + pow(30,0.7))", # noqa:E501
                                    "lang": "expression"
                                }
                            },
                            "weight": 10
                        }
                    ]
                }
            }
        }
    }
