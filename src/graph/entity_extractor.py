import spacy
nlp = spacy.load("en_core_web_sm")  # web_trf 

def extract_entities(chunk_text: str):
    doc = nlp(chunk_text)
    entities = []
    for ent in doc.ents:
        entities.append((ent.text, ent.label_))
    return entities

def extract_relations(chunk_text: str):
    # Simple heuristic relation extraction: subject-verb-object from dependency parse
    doc = nlp(chunk_text)
    rels = []
    for sent in doc.sents:
        subj = None
        obj = None
        verb = None
        for token in sent:
            if token.dep_.endswith("subj"):
                subj = token.text
            if token.dep_.endswith("obj"):
                obj = token.text
            if token.pos_ == "VERB":
                verb = token.lemma_
        if subj and obj and verb:
            rels.append((subj, verb, obj))
    return rels
