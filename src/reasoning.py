import os

os.environ["SWI_HOME_DIR"] = "C:\\Program Files\\swipl" 

from pyswip import Prolog

prolog = Prolog()

def load_facts(animal_data, animal_counts):
    prolog.retractall("animal(_,_)")
    prolog.retractall("animal_bbox(_,_,_,_,_)")
    prolog.retractall("animal_exists(_)")

    for animal, count in animal_counts.items():
        prolog.assertz(f"animal({animal}, {count})")
        prolog.assertz(f"animal_exists({animal}) :- animal({animal}, C), C > 0")

    for obj in animal_data:
        a = obj['class']
        x1, y1, x2, y2 = obj['bbox']
        prolog.assertz(f"animal_bbox({a}, {x1}, {y1}, {x2}, {y2})")

def get_count(animal):
    query = list(prolog.query(f"animal({animal.lower()}, C)"))
    return query[0]["C"] if query else 0

def does_exist(animal):
    return bool(list(prolog.query(f"animal_exists({animal.lower()})")))

def get_bboxes(animal):
    query = list(prolog.query(f"animal_bbox({animal.lower()}, X1, Y1, X2, Y2)"))
    return [(float(q["X1"]), float(q["Y1"]), float(q["X2"]), float(q["Y2"])) for q in query]
