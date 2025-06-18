from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util
import inflect

inflect_engine = inflect.engine()
# Load animal name extractor (FLAN-T5)
model_name = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model_request_animals = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def extract_animals_from_query(query):
    prompt = f"List the animal names mentioned in the query: {query}"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model_request_animals.generate(**inputs, max_length=32)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Convert to singular, clean up
    raw_list = [x.strip().lower() for x in response.replace('[', '').replace(']', '').split(',')]
    animal_list = [inflect_engine.singular_noun(animal) if inflect_engine.singular_noun(animal) else animal for animal in raw_list]
    
    return animal_list

# Load intent classifier (MPNet)
model_action = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
labels = ["counting", "existence", "location"]

def classify_query_intent(query):
    query_embedding = model_action.encode(query, convert_to_tensor=True)
    label_embeddings = model_action.encode(labels, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, label_embeddings)[0]
    best_label = labels[scores.argmax()]
    return {label: (label == best_label) for label in labels}