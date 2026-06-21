# def old():

#     import json
#     embeddings = {}

#     with open('captions.txt', 'r') as f:
#         caption = f.read().strip()
#         for line in caption.splitlines():
#             embeddings[line] = get_embeddings(line).tolist()
#             # embeddings[line] = len(line)
#     print(embeddings)

#     with open('embeddings.json', 'w') as f:
#         json.dump(embeddings, f, indent=4)


#     json_file_path = 'embeddings.json'
#     with open(json_file_path, 'r') as f:
#         embeddings = json.load(f)
#     print(embeddings)

#     import numpy as np

#     quary = "children playing in the afternoon"
#     quary_embedding = get_embeddings(quary).reshape(1, -1)
#     similarities = {}
#     for caption, embedding in embeddings.items():
#         embedding = np.array(embedding).reshape(1, -1)
#         print(embedding)
#         similarity = cosine_similarity(quary_embedding, embedding)[0][0]
#         similarities[caption] = similarity
#    print(similarities)

    # index = faiss.read_index("Embaddings.index")

    # quary = "children playing in the afternoon"
    # quary_embedding = get_embeddings(quary).reshape(1, -1).astype('float32')

    # D, I = index.search(quary_embedding, k=5)
    # print("Distances:", D)
    # print("Indices:", I)

import embaddings
import faiss
import json
import os
from dotenv import load_dotenv

def load_metadata(metadata_file="metadata.json"):
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def getCaptions(I=[0]):
    metadata = load_metadata()
    if metadata:
        return [metadata[i]["caption"] for i in I if 0 <= i < len(metadata)]
    raise Exeption("metadata file not Found")

def imagePath(I=[0]):
    metadata = load_metadata()
    if metadata:
        return [metadata[i]["image_path"] for i in I if 0 <= i < len(metadata)]
    raise Exeption("metadata file not Found")

def searchImages(quary, index=embaddings.loadEmbaddings("Embaddings.index"), k=3):
    quary_embedding = embaddings.get_embeddings(quary).reshape(1, -1).astype('float32')
    D, I = index.search(quary_embedding, k=k)   # for siliarity we already specify to use inner product(cosin similarity)
    return D, I, getCaptions(I[0]), imagePath(I[0])


if __name__ == "__main__":
    load_dotenv()
    images_dir = os.getenv("IMAGES_DIR","images")
    metadata_file = os.getenv("METADATA_FILE","metadata.json")
    Embaddings_index_file = os.getenv("EMBADDINGS_DB","Embaddings.index")

    embaddings.saveEmbaddings(Document=None, images_dir=images_dir, metadata_file=metadata_file)
    index = embaddings.loadEmbaddings(Embaddings_index_file)
    
    while True:
        quary = input("Enter a caption to search (or 'exit' to quit): ")
        if quary.lower() == 'exit':
            break
        D, I, captions, images = searchImages(quary, index)
        print("Distances:", D)
        print("Indices in FAISS:", I)
        print("Captions", captions)
        print("imagePath", images)