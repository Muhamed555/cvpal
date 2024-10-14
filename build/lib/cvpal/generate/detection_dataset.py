import os
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from diffusers import StableDiffusionPipeline, DDIMScheduler
from openai import OpenAI
import requests
from io import BytesIO
import json
import random
import yaml
import datetime
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed


class DetectionDataset:
    def __init__(self, model="stable-diffusion", openai_api_key=None):
        self.model = model
        self.openai_api_key = openai_api_key
        self.output_folder = "detection_dataset"
        self.images_folder = os.path.join(self.output_folder, "images")
        self.labels_folder = os.path.join(self.output_folder, "labels")
        self.null_files = []

        for folder in [self.output_folder, self.images_folder, self.labels_folder]:
            if not os.path.exists(folder):
                os.makedirs(folder)

        # Initialize OwlViT for object detection
        self.processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        self.detector = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

        # Check for GPU availability
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.detector = self.detector.to(self.device)

        if model == "stable-diffusion":
            model_id = "CompVis/stable-diffusion-v1-4"
            self.num_inference_steps = 50

            scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")

            self.pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                scheduler=scheduler,
                torch_dtype=torch.float16,
                safety_checker=None
            )
            self.pipe = self.pipe.to(self.device)
        elif model == "dalle" and openai_api_key is None:
            raise ValueError("OpenAI API key is required for DALL-E model")

    def generate(self, prompt, num_images=1, height=512, width=512, seed=42, labels=None, output_type="yolo",
                 overwrite=True):
        if labels is None or len(labels) == 0:
            raise ValueError("Please provide at least one label for object detection.")

        if overwrite and os.path.exists(self.output_folder):
            shutil.rmtree(self.output_folder)
            os.makedirs(self.output_folder)
            os.makedirs(self.images_folder)
            os.makedirs(self.labels_folder)

        existing_images = len([f for f in os.listdir(self.images_folder) if f.endswith('.jpg')])

        def generate_and_process_image(i):
            if self.model == "stable-diffusion":
                generator = torch.Generator(device=self.device).manual_seed(seed + i)
                with torch.autocast(device_type=self.device.type):
                    image = self.pipe(
                        prompt,
                        height=height,
                        width=width,
                        generator=generator,
                        num_inference_steps=self.num_inference_steps
                    ).images[0]
            elif self.model == "dalle":
                client = OpenAI(api_key=self.openai_api_key)
                response = client.images.generate(
                    model="dall-e-3",
                    prompt=prompt,
                    n=1,
                    size=f"{width}x{height}"
                )
                image_url = response.data[0].url
                image = Image.open(BytesIO(requests.get(image_url).content))

            # Perform object detection
            inputs = self.processor(text=[labels], images=image, return_tensors="pt").to(self.device)
            outputs = self.detector(**inputs)
            target_sizes = torch.Tensor([image.size[::-1]]).to(self.device)
            results = self.processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes,
                                                                   threshold=0.1)

            # Save image
            image_path = os.path.join(self.images_folder, f"image_{i + existing_images}.jpg")
            image.save(image_path)

            detections = []
            yolo_annotations = []
            coco_annotations = []

            for box, score, label in zip(results[0]["boxes"], results[0]["scores"], results[0]["labels"]):
                xmin, ymin, xmax, ymax = box.tolist()
                detection = {
                    "label": labels[label],
                    "score": score.item(),
                    "box": [xmin, ymin, xmax, ymax]
                }
                detections.append(detection)

                # YOLO format: <class> <x_center> <y_center> <width> <height>
                x_center = (xmin + xmax) / (2 * width)
                y_center = (ymin + ymax) / (2 * height)
                box_width = (xmax - xmin) / width
                box_height = (ymax - ymin) / height
                yolo_annotations.append(f"{label.item()} {x_center} {y_center} {box_width} {box_height}")

                # COCO format
                coco_annotations.append({
                    "image_id": i + existing_images,
                    "category_id": label.item(),
                    "bbox": [xmin, ymin, xmax - xmin, ymax - ymin],
                    "area": (xmax - xmin) * (ymax - ymin),
                    "segmentation": [],
                    "iscrowd": 0
                })

            if output_type == "yolo":
                # Save YOLO annotations
                yolo_path = os.path.join(self.labels_folder, f"image_{i + existing_images}.txt")
                with open(yolo_path, 'w') as f:
                    f.write("\n".join(yolo_annotations))

            return detections, coco_annotations

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = [executor.submit(generate_and_process_image, i) for i in range(num_images)]

            all_detections = []
            all_coco_annotations = []
            for future in as_completed(futures):
                detections, coco_annotations = future.result()
                all_detections.extend(detections)
                all_coco_annotations.extend(coco_annotations)

        # Assign annotation IDs
        for i, ann in enumerate(all_coco_annotations):
            ann["id"] = i

        # Update or create package and dataset information
        package_info = {
            "name": "CvPal",
            "version": "1.0.0",
            "github": "https://github.com/Muhamed555/CvPal/",
            "company": "Vision Full Space",
            "website": "https://visionfullspace.com/"
        }

        yaml_path = os.path.join(self.output_folder, "data.yaml")
        if os.path.exists(yaml_path) and not overwrite:
            with open(yaml_path, 'r') as f:
                yaml_config = yaml.safe_load(f)
            dataset_info = yaml_config.get("dataset_info", {})
            dataset_info["num_images"] = num_images + existing_images
            dataset_info["labels"] = list(set(dataset_info.get("labels", []) + labels))
            dataset_info["date_updated"] = datetime.datetime.now().isoformat()
        else:
            dataset_info = {
                "name": f"dataset_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}",
                "description": f"Dataset generated using {self.model} model with prompt: {prompt}",
                "num_images": num_images + existing_images,
                "image_size": f"{width}x{height}",
                "labels": labels,
                "date_created": datetime.datetime.now().isoformat(),
                "date_updated": datetime.datetime.now().isoformat()
            }

        if output_type == "coco":
            # Save COCO JSON
            coco_data = {
                "images": [{"id": i, "file_name": f"image_{i}.jpg", "width": width, "height": height} for i in
                           range(num_images + existing_images)],
                "annotations": all_coco_annotations,
                "categories": [{"id": i, "name": label} for i, label in enumerate(labels)],
                "info": {**package_info, **dataset_info}
            }
            coco_path = os.path.join(self.output_folder, "annotations.json")
            with open(coco_path, 'w') as f:
                json.dump(coco_data, f, indent=2)

        # Save or update YAML config for YOLO
        yaml_config = {
            "path": self.output_folder,
            # "train": "images/train",
            # "val": "images/val",
            # "test": "images/test",
            "names": dataset_info["labels"],
            "nc": len(dataset_info["labels"]),
            "cvpal": package_info,
            "dataset_info": dataset_info
        }
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_config, f, sort_keys=False)

        print(
            f"Generated {num_images} new images, total {num_images + existing_images} images with detections in {self.output_folder}")
        print(f"Output type: {output_type}")

    def isnull(self):
        print("Files with no detections:")
        for file in self.null_files:
            print(os.path.join(self.images_folder, file))

    def dropna(self):
        for file in self.null_files:
            os.remove(os.path.join(self.images_folder, file))
            label_file = os.path.join(self.labels_folder, file.replace('.jpg', '.txt'))
            if os.path.exists(label_file):
                os.remove(label_file)
        print(f"Removed {len(self.null_files)} files with no detections")
        self.null_files = []

    def add_labels(self, labels=[]):
        for file in os.listdir(self.images_folder):
            if file.endswith('.jpg'):
                image_path = os.path.join(self.images_folder, file)
                label_path = os.path.join(self.labels_folder, file.replace('.jpg', '.txt'))

                image = Image.open(image_path)
                inputs = self.processor(text=[labels], images=image, return_tensors="pt").to(self.device)
                outputs = self.detector(**inputs)
                target_sizes = torch.Tensor([image.size[::-1]]).to(self.device)
                results = self.processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes,
                                                                       threshold=0.1)

                new_annotations = []
                for box, score, label in zip(results[0]["boxes"], results[0]["scores"], results[0]["labels"]):
                    xmin, ymin, xmax, ymax = box.tolist()
                    x_center = (xmin + xmax) / (2 * image.width)
                    y_center = (ymin + ymax) / (2 * image.height)
                    box_width = (xmax - xmin) / image.width
                    box_height = (ymax - ymin) / image.height
                    new_annotations.append(f"{label.item()} {x_center} {y_center} {box_width} {box_height}")

                with open(label_path, 'a') as f:
                    f.write("\n" + "\n".join(new_annotations))

        # Update YAML file with new labels
        yaml_path = os.path.join(self.output_folder, "data.yaml")
        if os.path.exists(yaml_path):
            with open(yaml_path, 'r') as f:
                yaml_config = yaml.safe_load(f)

            current_labels = yaml_config['names']
            updated_labels = list(set(current_labels + labels))
            yaml_config['names'] = updated_labels
            yaml_config['nc'] = len(updated_labels)
            yaml_config['dataset_info']['labels'] = updated_labels
            yaml_config['dataset_info']['date_updated'] = datetime.datetime.now().isoformat()

            with open(yaml_path, 'w') as f:
                yaml.dump(yaml_config, f, sort_keys=False)

        print(f"Added new labels to existing dataset in {self.labels_folder} and updated data.yaml")

    def show_samples(self, num_samples=5, annotation_type="yolo"):
        image_files = [f for f in os.listdir(self.images_folder) if f.endswith('.jpg')]
        if num_samples > len(image_files):
            num_samples = len(image_files)

        sample_files = random.sample(image_files, num_samples)

        fig, axs = plt.subplots(num_samples, 1, figsize=(10, 10 * num_samples))
        if num_samples == 1:
            axs = [axs]

        if annotation_type == "coco":
            coco_path = os.path.join(self.output_folder, "annotations.json")
            with open(coco_path, 'r') as f:
                coco_data = json.load(f)
            annotations_by_image = {ann['image_id']: ann for ann in coco_data['annotations']}
            categories = {cat['id']: cat['name'] for cat in coco_data['categories']}

        for i, file in enumerate(sample_files):
            image_path = os.path.join(self.images_folder, file)

            image = Image.open(image_path)
            axs[i].imshow(image)

            if annotation_type == "yolo":
                label_path = os.path.join(self.labels_folder, file.replace('.jpg', '.txt'))
                if os.path.exists(label_path):
                    with open(label_path, 'r') as f:
                        annotations = f.readlines()

                    for ann in annotations:
                        label, x_center, y_center, width, height = map(float, ann.strip().split())
                        x_center *= image.width
                        y_center *= image.height
                        width *= image.width
                        height *= image.height

                        rect = patches.Rectangle(
                            (x_center - width / 2, y_center - height / 2),
                            width, height, linewidth=2, edgecolor='r', facecolor='none'
                        )
                        axs[i].add_patch(rect)
                        axs[i].text(
                            x_center - width / 2, y_center - height / 2,
                            f"Label: {int(label)}",
                            color='white', fontsize=12, bbox=dict(facecolor='red', alpha=0.5)
                        )

            elif annotation_type == "coco":
                image_id = int(file.split('.')[0].split('_')[-1])
                if image_id in annotations_by_image:
                    ann = annotations_by_image[image_id]
                    bbox = ann['bbox']
                    x, y, w, h = bbox
                    rect = patches.Rectangle(
                        (x, y), w, h, linewidth=2, edgecolor='r', facecolor='none'
                    )
                    axs[i].add_patch(rect)
                    axs[i].text(
                        x, y,
                        f"Label: {categories[ann['category_id']]}",
                        color='white', fontsize=12, bbox=dict(facecolor='red', alpha=0.5)
                    )

            axs[i].set_title(f"Sample {i + 1}")

        plt.tight_layout()
        plt.show()
