from typing import Any, Dict, List, Optional, Union
import random
import numpy as np
from swift.llm import DatasetMeta, ResponsePreprocessor, register_dataset
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class CSTSPreprocessor(ResponsePreprocessor):
    SYSTEM_PROMPT = "You are a precise semantic similarity evaluator. Your task is to judge the semantic similarity between two sentences based **completely** on the given condition, following the provided instructions."   
    
    PROMPT = """Judge the semantic similarity between Sentence 1 and Sentence 2 based **completely** on the given Condition.
The final output must be exactly in this format: the similarity judgment ('yes' or 'no') followed by the score in parentheses, wrapped in <answer></answer> tags. Examples: <answer>yes(4)</answer>, <answer>no(1)</answer>. Include no other text, tags, or explanations.

To arrive at this output, follow these two steps:
- **Step 1: Binary Judgment.** Determine if the sentences are 'similar' ('yes') or 'not similar' ('no').
  - 'similar': The sentences are roughly, mostly, or completely equivalent under the condition.
  - 'not similar': The sentences are dissimilar under the condition.

- **Step 2: Fine-grained Score.** Assign an integer score based on Step 1:
  - For a 'yes' judgment:
    - 5: The two sentences are completely equivalent as they mean the same thing with respect to the condition.
    - 4: The two sentences are mostly equivalent, but some unimportant details differ with respect to the condition.
    - 3: The two sentences are roughly equivalent, but some important information differs or is missing with respect to the condition.
  - For a 'no' judgment:
    - 2: The two sentences are dissimilar, but are on a similar topic with respect to the condition or shares a close semantic relationship. This applies when items are clearly different, but not direct opposites.
    - 1: The two sentences are dissimilar with respect to the condition, representing a direct opposition or a clear, unrelated difference. (e.g., 'man' vs. 'woman').

Here are some examples:

---
**Example 1:**
<Sentence1>: A girl is cooking in a kitchen and a man is standing next to her.
<Sentence2>: A man sitting with a pizza in his hand in front of pizza on the table.
<Condition>: The number of people.
<answer>no(1)</answer>
explaination: [The first sentence mentions two people, while the second sentence mentions only one person.]
---
**Example 2:**
<Sentence1>: A wood table sitting by a wood framed bed with a lamp on it.
<Sentence2>: A microwave, refrigerator, television, and wooden drawers sit in the corner of a bedroom.
<Condition>: The room type.
<answer>yes(5)</answer>
explaination: [We can infer from the two sentences that the room type are both bedroom.]
---
**Example 3:**
<Sentence1>: A small crowd gathered around the injured person.
<Sentence2>: A crowd jumps up and down to the tunes played by an artist.
<Condition>: The number of people
<answer>yes(3)</answer>
explaination: [While both sentences mention crowds, it is important and unclear how many people there are.]

Now, apply these steps to the following sentences:

<Sentence1>: {sentence1}
<Sentence2>: {sentence2}
<Condition>: {condition}
"""

    def __init__(self, 
                 is_train_dataset: bool = True, 
                 random_seed: int = 42,
                 columns: Optional[Dict[str, str]] = None, 
                 **kwargs):
        if columns is None:
            columns = {}
        
        super().__init__(columns=columns, **kwargs)
        
        self.is_train_dataset = is_train_dataset
        self.random_seed = random_seed
        self._dataset_processed = False
        self._processed_data = []

        random.seed(random_seed)
        np.random.seed(random_seed)
        
    def prepare_dataset(self, dataset):
        if self._dataset_processed:
            return dataset
            
        logger.info(f"Preparing CSTS dataset ({'train' if self.is_train_dataset else 'val'})")
        
        if hasattr(dataset, 'to_pandas'):
            df = dataset.to_pandas()
            data_list = df.to_dict('records')
        elif hasattr(dataset, 'to_list'):
            data_list = dataset.to_list()
        else:
            data_list = list(dataset)
        
        logger.info(f"Original dataset length: {len(data_list)}")
        
        # validate dataset length
        if len(data_list) % 2 != 0:
            logger.warning(f"Dataset length {len(data_list)} is not even, removing last sample")
            data_list = data_list[:-1]
        
        # train dataset shuffle
        if self.is_train_dataset:
            logger.info("Shuffling paired samples for training dataset")
            data_list = self._shuffle_pairs(data_list)

        # validate paired samples
        self._validate_paired_samples(data_list)
        
        # split pairs into individual samples
        self._processed_data = self._split_pairs_to_samples(data_list)
        logger.info(f"Processed dataset length: {len(self._processed_data)}")
        
        from datasets import Dataset
        new_dataset = Dataset.from_list(self._processed_data)
        self._dataset_processed = True
        
        return new_dataset

    def _validate_paired_samples(self, data_list: List[Dict[str, Any]]) -> None:
        logger.info("Validating paired samples...")
        validation_errors = 0
        
        for i in range(0, len(data_list), 2):
            if i+1 >= len(data_list):
                break
                
            sample1 = data_list[i]
            sample2 = data_list[i+1]
            
            if (sample1['sentence1'] != sample2['sentence1'] or 
                sample1['sentence2'] != sample2['sentence2']):
                logger.warning(f"Paired samples at indices {i} and {i+1} have mismatched sentences")
                validation_errors += 1
        
        if validation_errors > 0:
            logger.warning(f"Found {validation_errors} validation errors in paired samples")
        else:
            logger.info("All paired samples validation passed")

    def _shuffle_pairs(self, data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        paired_indices = [(i, i+1) for i in range(0, len(data_list), 2)]
        random.shuffle(paired_indices)
        
        shuffled_data = []
        for idx1, idx2 in paired_indices:
            shuffled_data.append(data_list[idx1])
            shuffled_data.append(data_list[idx2])
        
        return shuffled_data
    
    def scaled_label(self, label: Union[int, str]) -> int:
        if isinstance(label, str) and label.isdigit():
            label = int(label)
        max_label, min_label = 5, 1
        return float((label - min_label) / (max_label - min_label))

    def _split_pairs_to_samples(self, data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        samples = []
        
        for data_sample in data_list:
            csts_sample = {
                'query': self.PROMPT.format(
                    sentence1=data_sample['sentence1'].strip(),
                    sentence2=data_sample['sentence2'].strip(),
                    condition=data_sample['condition'].strip()
                ),
                'label': float(data_sample['label']) if data_sample['label']!= -1 else -1.0,
                'condition': data_sample['condition'],
                'original_sentence1': data_sample['sentence1'],
                'original_sentence2': data_sample['sentence2']
            }
            samples.append(csts_sample)
    
        return samples

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        result = super().preprocess(row)
        if result is not None and 'messages' in result:
            system_message = {
                'role': 'system',
                'content': self.SYSTEM_PROMPT
            }
            if not (result['messages'] and result['messages'][0].get('role') == 'system'):
                result['messages'].insert(0, system_message)
        
        return result


register_dataset(
    DatasetMeta(
        dataset_name='csts_train',
        dataset_path='/path/ms-swift/csts/csts-data/csts_train.csv',
        preprocess_func=CSTSPreprocessor(is_train_dataset=True, random_seed=42),
    ))

register_dataset(
    DatasetMeta(
        dataset_name='csts_val',
        dataset_path='/path/ms-swift/csts/csts-data/csts_validation.csv',
        preprocess_func=CSTSPreprocessor(is_train_dataset=False, random_seed=42),
    ))

register_dataset(
    DatasetMeta(
        dataset_name='csts_test',
        dataset_path='/path/ms-swift/csts/csts-data/csts_test.csv',
        preprocess_func=CSTSPreprocessor(is_train_dataset=False, random_seed=42),
    ))