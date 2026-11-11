import json
from pathlib import Path

from PIL import Image

YOLO_DIR = Path("dataset_yolo")
COCO_DIR = Path("dataset_coco") / "annotations"
SPLITS = ["train", "val", "test"]


CATEGORIAS = [{"id": 0, "name": "plate", "supercategory": "plate"}]

COCO_DIR.mkdir(parents=True, exist_ok=True)

for split in SPLITS:
    images, annotations = [], []
    ann_id = 0

    for img_id, img_path in enumerate(sorted((YOLO_DIR / "images" / split).iterdir())):
        with Image.open(img_path) as im:
            W, H = im.size

        images.append({"id": img_id, "file_name": img_path.name, "width": W, "height": H})

        # Cada línea YOLO: clase xc yc w h (normalizadas en [0, 1])
        label_path = YOLO_DIR / "labels" / split / f"{img_path.stem}.txt"
        lineas = label_path.read_text().splitlines() if label_path.exists() else []

        for linea in lineas:
            _, xc, yc, w, h = map(float, linea.split())

            # COCO: [x esquina superior izquierda, y esquina superior izquierda, ancho, alto] en píxeles
            bw, bh = w * W, h * H
            x, y = xc * W - bw / 2, yc * H - bh / 2

            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": 0,
                "bbox": [round(x, 2), round(y, 2), round(bw, 2), round(bh, 2)],
                "area": round(bw * bh, 2),
                "iscrowd": 0,
            })
            ann_id += 1

    coco = {"images": images, "annotations": annotations, "categories": CATEGORIAS}
    salida = COCO_DIR / f"instances_{split}.json"
    salida.write_text(json.dumps(coco))

    print(f"{split:5s}: {len(images):5d} imágenes, {len(annotations):5d} matrículas -> {salida}")