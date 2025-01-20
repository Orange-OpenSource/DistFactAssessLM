# To be used to migrate environment from wikifactdiff env (https://github.com/Orange-OpenSource/WikiFactDiff) to wikieval env
# $OLDSTORAGE is the old wikifactdiff storage for intermediate folder (called STORAGE_FOLDER) 
mv $OLDSTORAGE/mongodb_storage $OLDSTORAGE/storage_rdf_to_counterfact/
cd $OLDSTORAGE/storage_rdf_to_counterfact/
mv new_wikidata.json.bz2 20230227_wikidata.json.bz2
mv old_wikidata.json.bz2 20210104_wikidata.json.bz2
