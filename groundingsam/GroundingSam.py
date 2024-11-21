import supervision as sv
import os
import torch
from typing import List
import cv2
from tqdm.notebook import tqdm
from .groundingdino.util.inference import Model

import numpy as np
from .segment_anything import sam_model_registry, SamPredictor

# Global Variables
GROUNDING_DINO_CONFIG_PATH = "/content/SegmaVisionPro/groundingsam/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH = "/content/SegmaVisionPro/groundingsam/weights/groundingdino_swint_ogc.pth"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
assert(os.path.isfile(GROUNDING_DINO_CONFIG_PATH)), "GroundingDINO config file not found!"
assert(os.path.isfile(GROUNDING_DINO_CHECKPOINT_PATH)), "GroundingDINO checkpoint file not found!"
grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)

SAM_ENCODER_VERSION = "vit_h"
SAM_CHECKPOINT_PATH = "/content/SegmaVisionPro/groundingsam/weights/sam_vit_h_4b8939.pth"
assert(os.path.isfile(SAM_CHECKPOINT_PATH)), "SAM checkpoint file not found!"
sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(device=DEVICE)
sam_predictor = SamPredictor(sam)

# Function to enhance class names and get
def enhance_class_name(class_names: List[str]) -> List[str]:
  return [
      f"all {class_name}s"
      for class_name
      in class_names
  ]

# Function to replace base class detections with new classes
# ------- base class=new class, replace !after! detections
def annotate_new_class(class_names: List[str], new_class: List[str]) -> List[str]:
    return enhance_class_name(class_names) + enhance_class_name(new_class)



def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)


class GroundingSam:
    def __init__(self, classes, images_dir="./data/", annotations_dir="./annotations/", images_extensions=['jpg', 'jpeg', 'png']):
        self.classes = classes
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.images_extensions = images_extensions

        self.image_paths = sv.list_files_with_extensions(
            directory=self.images_dir,
            extensions=self.images_extensions)

        self.detections = {}
        self.images = {}
        self.annotations = {}

    def _calculate_detections(self, BOX_TRESHOLD=0.35, TEXT_TRESHOLD=0.25, class_enhancer=enhance_class_name):
        for image_path in tqdm(self.image_paths):
            image_name = image_path.name
            image_path = str(image_path)
            image = cv2.imread(image_path)

            detections = grounding_dino_model.predict_with_classes(
                image=image,
                classes=class_enhancer(class_names=self.classes),
                box_threshold=BOX_TRESHOLD,
                text_threshold=TEXT_TRESHOLD
            )
            detections = detections[detections.class_id != None]
            self.images[image_name] = image
            self.detections[image_name] = detections
            self.annotations[image_name] = detections

    def get_detections(self, BOX_TRESHOLD=0.35, TEXT_TRESHOLD=0.25, class_enhancer=enhance_class_name):
        if not self.detections:  # If detections haven't been calculated, do so
            self._calculate_detections(BOX_TRESHOLD, TEXT_TRESHOLD, class_enhancer)
        return self.detections

  def get_masks(self, BOX_TRESHOLD=0.35, TEXT_TRESHOLD=0.25, class_enhancer=enhance_class_name):
      if not self.detections:  # If detections haven't been calculated, do so
          self._calculate_detections(BOX_TRESHOLD, TEXT_TRESHOLD, class_enhancer)
  
      # Compute masks and store them in self.annotations
      for image_name, detections in self.detections.items():
          image = self.images[image_name]
          detections.mask = segment(
              sam_predictor=sam_predictor,
              image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
              xyxy=detections.xyxy
          )
          self.annotations[image_name] = detections
  
      # Plotting logic
      plot_images = []
      plot_titles = []
  
      box_annotator = sv.BoxAnnotator()
      mask_annotator = sv.MaskAnnotator()
  
      for image_name, detections in self.annotations.items():
          image = self.images[image_name]
          plot_images.append(image)
          plot_titles.append(image_name)
  
          labels = [
              f"{self.classes[class_id]} {confidence:0.2f}"
              for _, _, confidence, class_id, _
              in detections
          ]
          annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
          annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
          plot_images.append(annotated_image)
          title = " ".join(set([
              self.classes[class_id]
              for class_id
              in detections.class_id
          ]))
          plot_titles.append(title)
  
      sv.plot_images_grid(
          images=plot_images,
          titles=plot_titles,
          grid_size=(len(self.annotations), 2),
          size=(2 * 4, len(self.annotations) * 4)
      )


    def annotate_images(self):
        plot_images = []
        plot_titles = []

        box_annotator = sv.BoxAnnotator()

        for image_name, detections in self.detections.items():
            image = self.images[image_name]
            plot_images.append(image)
            plot_titles.append(image_name)

            labels = [
                f"{self.classes[class_id]} {confidence:0.2f}"
                for _, _, confidence, class_id, _
                in detections]
            annotated_image = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)
            plot_images.append(annotated_image)
            title = " ".join(set([
                self.classes[class_id]
                for class_id
                in detections.class_id
            ]))
            plot_titles.append(title)

        sv.plot_images_grid(
            images=plot_images,
            titles=plot_titles,
            grid_size=(len(self.detections), 2),
            size=(2 * 4, len(self.detections) * 4)
        )

    def save_as_pascal_voc(self, min_image_area_percentage=0.002, max_image_area_percentage=0.80, approximation_percentage=0.75):
        dataset = sv.Dataset(
            classes=self.classes,
            images=self.images,
            annotations=self.annotations
        )
        dataset.as_pascal_voc(
            annotations_directory_path=self.annotations_dir,
            min_image_area_percentage=min_image_area_percentage,
            max_image_area_percentage=max_image_area_percentage,
            approximation_percentage=approximation_percentage
        )


class AutomaticLabel(GroundingSam):
    def __init__(self, base_classes, new_classes=None, images_dir="./data/", annotations_dir="./annotations/", images_extensions=['jpg', 'jpeg', 'png']):
        """
        Initialize AutomaticLabel with base and new classes.
        Detection is performed on base_classes, and annotation/saving is performed on new_classes.
        """
        self.base_classes = base_classes  # Classes used for detection
        self.new_classes = new_classes or []  # Classes used for annotation and saving
        self.classes = base_classes  # Start with base classes for detection
        
        # Initialize the parent class
        super().__init__(classes=self.base_classes, images_dir=images_dir, annotations_dir=annotations_dir, images_extensions=images_extensions)

    def _map_detections_to_new_classes(self, detections, base_to_new_mapping):
        """
        Map detections from base classes to new classes based on a mapping.
        
        Args:
            detections: Original detections.
            base_to_new_mapping: Dictionary mapping base class indices to new class indices.
        
        Returns:
            Updated detections with new class annotations.
        """
        for detection in detections:
            class_id = detection.class_id
            if class_id in base_to_new_mapping:
                detection.class_id = base_to_new_mapping[class_id]
        return detections

    def _calculate_detections(self, BOX_TRESHOLD=0.35, TEXT_TRESHOLD=0.25, class_enhancer=enhance_class_name):
        """
        Override detection logic to detect based on base classes but annotate based on new classes.
        """
        # Prepare a mapping from base to new class indices
        base_to_new_mapping = {i: i for i in range(len(self.base_classes))}  # Default mapping if no new classes
        
        if self.new_classes:
            base_to_new_mapping = {
                base_idx: new_idx
                for base_idx, new_idx in zip(range(len(self.base_classes)), range(len(self.new_classes)))
            }
        
        for image_path in tqdm(self.image_paths):
            image_name = image_path.name
            image_path = str(image_path)
            image = cv2.imread(image_path)

            # Detect objects using base classes
            detections = grounding_dino_model.predict_with_classes(
                image=image,
                classes=enhance_class_name(self.base_classes),
                box_threshold=BOX_TRESHOLD,
                text_threshold=TEXT_TRESHOLD
            )
            detections = detections[detections.class_id != None]

            # Map detections to new classes
            detections = self._map_detections_to_new_classes(detections, base_to_new_mapping)
            
            self.images[image_name] = image
            self.detections[image_name] = detections
            self.annotations[image_name] = detections

    def save_as_pascal_voc_with_new_class(self, min_image_area_percentage=0.002, max_image_area_percentage=0.80, approximation_percentage=0.75):
        """
        Save annotations in Pascal VOC format using new classes.
        """
        dataset = sv.Dataset(
            classes=self.new_classes,  # Use new classes for saving
            images=self.images,
            annotations=self.annotations
        )
        dataset.as_pascal_voc(
            annotations_directory_path=self.annotations_dir,
            min_image_area_percentage=min_image_area_percentage,
            max_image_area_percentage=max_image_area_percentage,
            approximation_percentage=approximation_percentage
        )
