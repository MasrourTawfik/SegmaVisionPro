import language_tool_python
import random
import csv
from PIL import Image
import json

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords,cmudict
from nltk.tree import Tree
import string

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



class PromptAnalyzer():
    def __init__(self, prompts_dict):
        self.prompts_dict = prompts_dict  
        self.prompts_list = []
        for image_name, prompts_list in self.prompts_dict.items():
            self.prompts_list = prompts_list

######################################################################################################################
############################################# main prompts processor #################################################
######################################################################################################################


    def process_prompts(self, readability=False, complexity_criterion="vocabulary_richness", reference_prompts=None):
        '''
        Process prompts to determine the best ones based on readability, complexity criteria, or relevance.

        This method processes each set of prompts associated with an image by:
        - Calculating similarity between prompts and optionally removing similar prompts.
        - Evaluating prompts based on the specified complexity criterion, readability, or relevance.
        - Sorting the prompts based on the chosen criterion.
        - Selecting the top three prompts for each image.

        Parameters:
        readability (bool): Whether to evaluate prompts based on readability (default is False).
        complexity_criterion (str): The complexity criterion to use for evaluation. 
                                    Options are "vocabulary_richness", "lexical_density", "parse_tree_depth", "relevance" (default is "vocabulary_richness").
        reference_prompts (list): A list of reference prompts to compare against for relevance. Required if complexity_criterion is "relevance".
        '''
        results = []
        for image_name, prompts_list in self.prompts_dict.items():
            self.prompts_list = prompts_list
            self.prompts_similarity(remove_similar=True)  
            
            if complexity_criterion == "vocabulary_richness":
                scores = self.vocabulary_richness()
                sorted_prompts = sorted(zip(self.prompts_list, scores), key=lambda x: -x[1])
            
            elif complexity_criterion == "lexical_density":
                scores = self.lexical_density()
                sorted_prompts = sorted(zip(self.prompts_list, scores), key=lambda x: -x[1])
            
            elif complexity_criterion == "parse_tree_depth":
                scores = self.parse_tree_depth()
                sorted_prompts = sorted(zip(self.prompts_list, scores), key=lambda x: -x[1])
            
            elif complexity_criterion == "relevance":
                if reference_prompts is None:
                    raise ValueError("reference_prompts must be provided when using relevance as the complexity criterion.")
                scores = self.relevance(reference_prompts)
                sorted_prompts = sorted(zip(self.prompts_list, scores), key=lambda x: -x[1])
            
            elif readability:
                scores = self.prompt_readability()
                sorted_prompts = sorted(zip(self.prompts_list, scores), key=lambda x: x[1])

            # Handle fewer than three prompts
            top_prompts = [prompt[0] for prompt in sorted_prompts[:3]]  
            while len(top_prompts) < 3:
                top_prompts.append("N/A")  # Fill in missing prompts with "N/A"

            results.append({
                'image_name': image_name,
                'best_prompt1': top_prompts[0],
                'best_prompt2': top_prompts[1],
                'best_prompt3': top_prompts[2]
            })

        self.write_to_json(results)

    def write_to_json(self, results):
        """
        Write results to a JSON file.
        """
        with open('prompt_results.json', 'w', encoding='utf-8') as json_file:
            json.dump(results, json_file, indent=4, ensure_ascii=False)
        print("Results saved to 'prompt_results.json'")



######################################################################################################################
########################################### Initial Prompts Processing ###############################################
######################################################################################################################


    def prompt_processing(self):
        '''
        Process prompts by removing punctuation, tokenizing, and filtering out stop words.

        This method processes each prompt in `self.prompts_list` by:
        - Removing punctuation.
        - Tokenizing the prompt into words.
        - Filtering out stop words.
        - Calculating the length of the filtered prompt.
        - Identifying unique words in the filtered prompt.

        Returns:
        tuple: A tuple containing the following lists:
            - prompts_unpunctuated (list of str): Prompts with punctuation removed.
            - prompts_filtered (list of list of str): Tokenized and filtered prompts.
            - prompts_length (list of int): Length of each filtered prompt.
            - unique_words_list (list of set of str): Unique words in each filtered prompt.
        '''
        stop_words = set(stopwords.words('english'))
        punct_table = str.maketrans('', '', string.punctuation)
        prompts_unpunctuated = []
        prompts_filtered = []
        prompts_length = []
        unique_words_list = []

        for prompt in self.prompts_list:
            prompt_unpunctuated = prompt.translate(punct_table)
            words = word_tokenize(prompt_unpunctuated)
            prompt_filtered = [word for word in words if word.lower() not in stop_words]
            prompt_length = len(prompt_filtered)
            unique_words = set(prompt_filtered)

            prompts_unpunctuated.append(prompt_unpunctuated)
            prompts_filtered.append(prompt_filtered)
            prompts_length.append(prompt_length)
            unique_words_list.append(unique_words)

        return prompts_unpunctuated, prompts_filtered, prompts_length, unique_words_list


######################################################################################################################
############################################### Similarity Removal ###################################################
######################################################################################################################


    def prompts_similarity(self, remove_similar=False, threshold=0.7):
        """
        This function calculates the similarity between prompts and optionally removes similar prompts.

        Parameters:
        remove_similar (bool): If True, removes prompts that are similar to each other based on the threshold.
        threshold (float): The similarity threshold above which prompts are considered similar. Default is 0.7.

        Returns:
        numpy.ndarray: A matrix of similarity scores if remove_similar is False.
        list: The updated list of prompts if remove_similar is True.

        Explanation:
        - The function first processes the prompts using `self.prompt_processing()` and obtains the unpunctuated prompts.
        - It uses TfidfVectorizer to convert the prompts into a TF-IDF matrix.
        - The cosine similarity between the TF-IDF vectors is calculated to create a similarity matrix.
        - If `remove_similar` is False, the function returns the similarity matrix.
        - If `remove_similar` is True, it identifies prompts that are similar based on the given threshold.
        - It creates a set of indices of similar prompts.
        - If there are multiple similar prompts, it randomly removes all but one from the list of prompts.
        - It returns the updated list of prompts.
        - If there are not enough similar prompts to remove, it prints a message and returns the original list of prompts.
        """
        prompts_unpunctuated, _, _, _ = self.prompt_processing()
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(prompts_unpunctuated)
        similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

        if not remove_similar:
            return similarity_matrix
        else:
            similar_prompts = set()
            for i in range(similarity_matrix.shape[0]):
                for j in range(i + 1, similarity_matrix.shape[1]):
                    if similarity_matrix[i][j] > threshold:
                        similar_prompts.update([i, j])

            similar_prompts = list(similar_prompts)
            m = len(similar_prompts)
            if m > 1:
                prompts_to_remove = [self.prompts_list[i] for i in random.sample(similar_prompts, m - 1)]
                self.prompts_list = [prompt for prompt in self.prompts_list if prompt not in prompts_to_remove]
                return self.prompts_list
            else:
                print("Not enough similar prompts to remove items according to threshold.")
                print("Returning original prompts list.")
                return self.prompts_list

######################################################################################################################
################################################ Complexity Metrics ##################################################
######################################################################################################################

    def vocabulary_richness(self):
        """
        This function calculates the vocabulary richness score for each prompt.

        Returns:
        list: A sorted list of vocabulary richness scores for all prompts.

        Explanation:
        - The function processes the prompts using `self.prompt_processing()` and obtains the filtered prompts.
        - It initializes an empty list `vocabulary_richness_scores` to store the scores.
        - For each filtered prompt:
        - It calculates the length of the prompt.
        - It determines the number of unique words in the prompt by converting it to a set.
        - It calculates the vocabulary richness as the ratio of the number of unique words to the total length of the prompt.
        - If the prompt length is zero, it assigns a richness score of 0.
        - It appends the calculated richness score to the `vocabulary_richness_scores` list.
        - Finally, it returns the list of vocabulary richness scores, sorted in ascending order.
        """
        vocabulary_richness_scores = []
        _, prompts_filtered, _, _ = self.prompt_processing()

        for filtered_prompt in prompts_filtered:
            prompt_length = len(filtered_prompt)
            unique_words_number = set(filtered_prompt)
            vocabulary_richness = len(unique_words_number) / prompt_length if prompt_length > 0 else 0
            vocabulary_richness_scores.append(vocabulary_richness)

        return sorted(vocabulary_richness_scores)
    

    def lexical_density(self):
        """
        This function calculates the lexical density for each prompt.

        Returns:
        list: A sorted list of lexical density scores for all prompts.

        Explanation:
        - The function processes the prompts using `self.prompt_processing()` and obtains the filtered prompts.
        - It defines a set of content word tags (`content_words`) which includes nouns, verbs, adjectives, and adverbs.
        - It initializes an empty list `lexical_densities` to store the scores.
        - For each filtered prompt:
        - It tags each word in the prompt with its part of speech using `nltk.pos_tag()`.
        - It counts the number of content words in the prompt by checking if the tag of each word is in `content_words`.
        - It calculates the lexical density as the ratio of the number of content words to the total length of the prompt.
        - If the prompt length is zero, it assigns a lexical density score of 0.
        - It appends the calculated lexical density score to the `lexical_densities` list.
        - Finally, it returns the list of lexical density scores, sorted in ascending order.
        """
        _, prompts_filtered, _, _ = self.prompt_processing()
        content_words = {'NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS'}
        lexical_densities = []
        
        for filtered_prompt in prompts_filtered:
            tagged_words = nltk.pos_tag(filtered_prompt)
            content_word_count = sum(1 for word, tag in tagged_words if tag in content_words)
            lexical_density = content_word_count / len(filtered_prompt) if len(filtered_prompt) > 0 else 0
            lexical_densities.append(lexical_density)
        
        return sorted(lexical_densities)


    def parse_tree_depth(self):
        """
        This function calculates the average parse tree depth for each prompt. The depth of a parse tree reflects the syntactic complexity of a sentence. Deeper trees indicate more complex syntactic structures, such as nested clauses and multiple phrases.
        Prompts with higher average parse tree depths are likely to have more intricate sentence structures, indicating higher complexity.

        Returns:
        list: A list of average parse tree depths for all prompts.

        Explanation:
        - The function defines a helper function `get_parse_tree(sentence)` that:
        - Tokenizes the input sentence using `nltk.word_tokenize`.
        - Tags the tokens with part of speech tags using `nltk.pos_tag`.
        - Defines a simple grammar for noun phrases (NP), prepositional phrases (PP), verb phrases (VP), and clauses (CLAUSE).
        - Uses `nltk.RegexpParser` to parse the tagged tokens into a parse tree based on the defined grammar.
        - Returns the parse tree if successful, otherwise returns None.
        - The function defines another helper function `tree_depth(tree)` that:
        - Recursively calculates the depth of the parse tree.
        - Returns 1 plus the maximum depth of its children if the input is a tree, otherwise returns 0.
        - It initializes an empty list `prompt_depths` to store the average parse tree depths for each prompt.
        - For each prompt in `self.prompts_list`:
        - It tokenizes the prompt into sentences using `nltk.sent_tokenize`.
        - For each sentence:
            - It generates a parse tree using `get_parse_tree`.
            - If a parse tree is generated, it calculates its depth using `tree_depth` and appends the depth to `depths`.
        - It calculates the average depth of all sentences in the prompt and appends this average to `prompt_depths`.
        - Finally, it returns the list of average parse tree depths for all prompts.
        """
        def get_parse_tree(sentence):
            tokens = nltk.word_tokenize(sentence)
            tagged = nltk.pos_tag(tokens)
            grammar = """
                NP: {<DT>?<JJ>*<NN>}
                PP: {<IN><NP>}
                VP: {<VB.*><NP|PP|CLAUSE>+$}
                CLAUSE: {<NP><VP>}
            """
            cp = nltk.RegexpParser(grammar)
            try:
                tree = cp.parse(tagged)
                return tree
            except:
                return None

        def tree_depth(tree):
            if isinstance(tree, Tree):
                return 1 + max(tree_depth(child) for child in tree) if tree else 0
            else:
                return 0

        prompt_depths = []
        for prompt in self.prompts_list:
            depths = []
            sentences = sent_tokenize(prompt)
            for sentence in sentences:
                parse_tree = get_parse_tree(sentence)
                if parse_tree:
                    depth = tree_depth(parse_tree)
                    depths.append(depth)
            avg_depth = sum(depths) / len(depths) if depths else 0
            prompt_depths.append(avg_depth)
        
        return sorted(prompt_depths)



    def relevance(self, reference_prompts):
        """
        This function calculates the relevance of each prompt in `self.prompts_list` compared to a set of reference prompts.

        Parameters:
        reference_prompts (list): A list of reference prompts to compare against.

        Returns:
        list: A sorted list of relevance scores for all prompts in `self.prompts_list`.
        """
        vectorizer = TfidfVectorizer()
        relevance_scores = []

        for prompt in self.prompts_list:
            tfidf_matrix = vectorizer.fit_transform([prompt] + reference_prompts)
            similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
            relevance_score = similarity_matrix.mean()
            relevance_scores.append(relevance_score)

        return sorted(relevance_scores, reverse=True)

######################################################################################################################
################################################ Readability Metrics #################################################
######################################################################################################################


    def prompt_readability(self):
        flesch_scores = []

        for prompt in self.prompts_list:
            flesch_score = self.readability(prompt)
            flesch_scores.append(flesch_score)

        return sorted(flesch_scores)

    def readability(self, prompt):
        sentences = sent_tokenize(prompt)
        words = word_tokenize(prompt)
        num_sentences = len(sentences)
        num_words = len(words)
        d = cmudict.dict()

        def count_syllables(word):
            pronunciation_list = d.get(word.lower())
            if not pronunciation_list:
                return 0
            pronunciation = pronunciation_list[0]
            return sum(1 for s in pronunciation if s[-1].isdigit())

        num_syllables = sum(count_syllables(word) for word in words)
        flesch_score = round(206.835 - 1.015 * (num_words / num_sentences) - 84.6 * (num_syllables / num_words), 2)
        return flesch_score

