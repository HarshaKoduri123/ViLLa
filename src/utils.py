import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from reasoning import get_bboxes, does_exist

def plot_multiple_bboxes(image_path, animal_list):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Cannot load image from {image_path}")
        return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots()
    ax.imshow(img)

    for animal in animal_list:
        bboxes = get_bboxes(animal)
        if not bboxes:
            print(f"No locations found for {animal}")
            continue
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     linewidth=2, edgecolor='green', facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1 - 5, animal.capitalize(), color='green', fontsize=8)

    plt.axis('off')
    plt.tight_layout()
    plt.show()
