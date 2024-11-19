from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer, BlipForConditionalGeneration, BlipProcessor
from PIL import Image
import torch
import os

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Synchronizer class
class Synchronizer:
    def __init__(self, detection_data):
        # Store detection data, mapping images to detected labels and bounding boxes
        self.detection_data = detection_data

    def sync_with_prompt_generator(self, prompt_generator):
        # Process each image's detection data for the prompt generator
        for image_name, data in self.detection_data.items():
            labels = data["labels"]
            bounding_boxes = data["bounding_boxes"]
            prompt_generator.set_context(image_name, labels, bounding_boxes)


# PromptGenerator class
class PromptGenerator:
    
    def __init__(self, image_directory="./data/", images_extensions=['jpg', 'jpeg', 'png'], models_to_use=None):
        # Initialize available models
        self.available_models = {
            "blip": self.generate_blip_prompts,
            "vit_gpt2": self.generate_vit_gpt2_prompts
        }
        
        # Set the models to use (if None, use all available models)
        self.models_to_use = models_to_use if models_to_use else list(self.available_models.keys())
        
        # Initialize the processors and models only for selected ones
        self.models = {
            "blip": BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device),
            "vit_gpt2": VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning").to(device)
        }
        
        self.processors = {
            "blip": BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base"),
            "vit_gpt2": ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        }

        self.tokenizers = {
            "vit_gpt2": AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        }

        self.image_directory = image_directory
        self.images_extensions = images_extensions
        self.prompt_pool = {}
        self.context_data = {}  # Store context data for each image

    def set_context(self, image_name, labels, bounding_boxes):
        # Set context data for specific image to guide prompt generation
        self.context_data[image_name] = {
            "labels": labels,
            "bounding_boxes": bounding_boxes
        }

    def load_images(self):
        images = []
        image_names = []
        for image_name in os.listdir(self.image_directory):
            if image_name.split('.')[-1].lower() in self.images_extensions:
                image_path = os.path.join(self.image_directory, image_name)
                try:
                    i_image = Image.open(image_path)
                    if i_image.mode != "RGB":
                        i_image = i_image.convert(mode="RGB")
                    images.append(i_image)
                    image_names.append(image_name)
                except Exception as e:
                    print(f"Error opening image {image_name}: {e}")
                    continue

        return images, image_names

    def generate_blip_prompts(self, images, num_prompts, max_length):
        prompts_dict = {}

        # Preprocess images and get input tensor
        inputs = self.processors["blip"](images=images, return_tensors="pt", padding=True).to(device)
        pixel_values = inputs['pixel_values']
        attention_mask = inputs.get('attention_mask', None)  # Get attention mask if available

        # Ensure pad_token_id is set correctly
        if self.models["blip"].config.pad_token_id is None:
            self.models["blip"].config.pad_token_id = self.models["blip"].config.eos_token_id

        for idx in range(len(images)):
            prompts = []
            for _ in range(num_prompts):
                output_ids = self.models["blip"].generate(
                    pixel_values[idx].unsqueeze(0),  # Add batch dimension for a single image
                    attention_mask=attention_mask[idx].unsqueeze(0) if attention_mask is not None else None,  # Attention mask
                    max_length=max_length,
                    num_beams=1,
                    do_sample=True,
                    top_k=50,
                    temperature=1.0
                )
                pred = self.processors["blip"].tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
                prompts.append(pred)
            prompts_dict[idx] = prompts
        return prompts_dict

    def generate_vit_gpt2_prompts(self, images, num_prompts, max_length):
        prompts_dict = {}

        # Preprocess images and get input tensor with attention mask
        inputs = self.processors["vit_gpt2"](images=images, return_tensors="pt", padding=True).to(device)
        pixel_values = inputs['pixel_values']
        attention_mask = inputs.get('attention_mask', None)  # Get attention mask if available

        # Ensure pad_token_id is set correctly
        if self.models["vit_gpt2"].config.pad_token_id is None:
            self.models["vit_gpt2"].config.pad_token_id = self.models["vit_gpt2"].config.eos_token_id

        for idx in range(len(images)):
            prompts = []
            for _ in range(num_prompts):
                output_ids = self.models["vit_gpt2"].generate(
                    pixel_values[idx].unsqueeze(0),  # Add batch dimension for a single image
                    attention_mask=attention_mask[idx].unsqueeze(0) if attention_mask is not None else None,  # Handle mask
                    max_length=max_length,
                    num_beams=1,
                    do_sample=True,
                    top_k=50,
                    temperature=1.0
                )
                pred = self.tokenizers["vit_gpt2"].decode(output_ids[0], skip_special_tokens=True).strip()
                prompts.append(pred)
            prompts_dict[idx] = prompts
        return prompts_dict
        
    def generate_prompts(self, num_prompts=5, max_length=15):
        images, image_names = self.load_images()
        final_prompt_pool = {}

        # For each selected model, generate prompts and concatenate the results
        for model_name in self.models_to_use:
            if model_name in self.available_models:
                model_prompts = self.available_models[model_name](images, num_prompts, max_length)
                for idx, prompts in model_prompts.items():
                    if image_names[idx] not in final_prompt_pool:
                        final_prompt_pool[image_names[idx]] = []
                    final_prompt_pool[image_names[idx]].extend(prompts)

        return final_prompt_pool