import os
import re
import json
import faiss
import numpy as np
from ImageCaptioning import create_metadata_db
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-miniLM-L6-v2', local_files_only=True)

def get_embeddings(sentence):
    return model.encode(sentence)

def load_captions(metadata_file="metadata.json"):
    try:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        return [item["caption"] for item in metadata] # return Document
    except:
        raise Exeption("medata file not Found, running saveEmbadding")
        saveEmbaddings()    


def saveEmbaddings(Document=None, file="Embaddings.index", images_dir="images", metadata_file="metadata.json"):
    if Document is None:
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            print(f"Loaded existing metadata database from {metadata_file} ({len(metadata)} entries)")
        except Exception:
            print(f"Metadata file {metadata_file} not found or corrupted. Generating new metadata database using BLIP model...")
            metadata = create_metadata_db(images_dir, metadata_file)
        Document = [item["caption"] for item in metadata]
        
    embeddings = get_embeddings(Document)
    embaddings = np.array(embeddings).astype('float32')
    index = faiss.IndexFlatIP(384)
    index.add(embaddings)
    print(index)
 
    faiss.write_index(index, file)


def loadEmbaddings(file="Embaddings.index"):
    index = faiss.read_index(file)
    print(index)
    return index


if __name__ == "__main__":
    text = "children playing in the afternoon"
    embedding = get_embeddings(text)
    print(embedding)
    saveEmbaaddings()