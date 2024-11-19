import nltk
from tqdm import tqdm

def initialize_nltk_resources():
    nltk_resources = [
        "punkt",  # Sentence and word tokenizer
        "stopwords",  # Stop words for filtering
        "averaged_perceptron_tagger",  # Part-of-speech tagging
        "cmudict",  # CMU Pronouncing Dictionary for syllable counts
        "punkt_tab",
        "averaged_perceptron_tagger_eng",
    ]
    print("Downloading required NLTK resources...")

    for resource in tqdm(nltk_resources, desc="Downloading NLTK Resources"):
        nltk.download(resource, quiet=True, force=True)  # Force download to ensure resources are refreshed

    print("All NLTK resources are ready!")

# Call the function to initialize NLTK resources
initialize_nltk_resources()