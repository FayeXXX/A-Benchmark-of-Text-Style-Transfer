from elasticsearch import Elasticsearch

# es = Elasticsearch(hosts="http://127.0.0.1:9200")
es = Elasticsearch([{'host':'localhost','port':9200,'scheme':'http'}])
result = es.indices.create(index='news', ignore=400)
print(result)