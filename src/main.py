from vision import detect_animals
from language import extract_animals_from_query, classify_query_intent
from reasoning import load_facts, get_count, does_exist
from utils import plot_multiple_bboxes


image_path = "1692db0fa150891f.jpg"
query = "where are camels in image"


# Vision pipeline
animal_data, animal_counts = detect_animals(image_path)


# Language pipeline
query_animals = extract_animals_from_query(query)
intent_flags = classify_query_intent(query)

# Load Prolog knowledge base
load_facts(animal_data, animal_counts)


if intent_flags.get("counting"):
    for animal in query_animals:
        print(f"{animal.capitalize()} count: {get_count(animal)}")

elif intent_flags.get("existence"):
    for animal in query_animals:
        exists = does_exist(animal)
        print(f"{animal.capitalize()} exists? {'Yes' if exists else 'No'}")

elif intent_flags.get("location"):
    to_plot = [a for a in query_animals if does_exist(a)]
    if to_plot:
        plot_multiple_bboxes(image_path, to_plot)
    else:
        print("No queried animals found in the image.")