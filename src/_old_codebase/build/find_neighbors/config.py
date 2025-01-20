from _old_codebase.build.config import STORAGE_FOLDER
import os.path as osp


EMBEDDINGS_FOLDER = osp.join(STORAGE_FOLDER, "gpt_encode_entity_labels")
TFIDF_WIKIPEDIA_INDEX_FOLDER = osp.join(STORAGE_FOLDER, "tfidf_index")
TFIDF_FULL_INDEX_FOLDER = osp.join(STORAGE_FOLDER, "tfidf_full_index")