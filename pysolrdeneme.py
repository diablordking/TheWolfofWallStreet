import scorched

si = scorched.SolrInterface("http://localhost:8983/solr/wolf/")

for result in si.query("Apple").execute():
  print result["news"]