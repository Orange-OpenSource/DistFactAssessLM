# To be used to migrate environment from wikifactdiff env (https://github.com/Orange-OpenSource/WikiFactDiff) to wikieval env
# This has to be executed inside mongosh
use wiki
db.wikidata_old_json.renameCollection('wikidata__20210104__ALMOST_RAW')
db.wikidata_old_prep.renameCollection('wikidata__20210104__PREPROCESSED')
db.wikidata_new_json.renameCollection('wikidata__20230227__ALMOST_RAW')
db.wikidata_new_prep.renameCollection('wikidata__20230227__PREPROCESSED')