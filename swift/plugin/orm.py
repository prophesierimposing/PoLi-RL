import os
import re
from typing import TYPE_CHECKING, Dict, List, Union, Tuple
from scipy.stats import rankdata
import numpy as np
import json
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from swift.llm import InferRequest


class ORM:

    def __call__(self, **kwargs) -> List[float]:
        raise NotImplementedError


class ReactORM(ORM):

    @staticmethod
    def evaluate_action_reward(action_pred: list, action_ref: list, cand_list: list, ref_list: list):
        f1 = []
        for i in range(len(action_pred)):
            ref_action = action_ref[i]
            pred_action = action_pred[i]

            ref_input = ref_list[i]
            cand_input = cand_list[i]

            ref_is_json = False
            try:
                ref_input_json = json.loads(ref_input)
                ref_is_json = True
            except Exception:
                ref_input_json = ref_input

            cand_is_json = False
            try:
                cand_input_json = json.loads(cand_input)
                cand_is_json = True
            except Exception:
                cand_input_json = cand_input

            if ref_action != pred_action or (ref_is_json ^ cand_is_json):
                f1.append(0)
            elif not ref_is_json and not cand_is_json:
                rougel = ReactORM.evaluate_rougel([ref_input_json], [cand_input_json])
                if rougel is None or rougel < 10:
                    f1.append(0)
                elif 10 <= rougel < 20:
                    f1.append(0.1)
                else:
                    f1.append(1)
            else:
                if not isinstance(ref_input_json, dict) or not isinstance(cand_input_json, dict):
                    # This cannot be happen, but:
                    # line 62, in evaluate_action_reward
                    # for k, v in ref_input_json.items():
                    # AttributeError: 'str' object has no attribute 'items'
                    # print(f'>>>>>>ref_input_json: {ref_input_json}, cand_input_json: {cand_input_json}')
                    f1.append(0)
                    continue

                half_match = 0
                full_match = 0
                if ref_input_json == {}:
                    if cand_input_json == {}:
                        f1.append(1)
                    else:
                        f1.append(0)
                else:
                    for k, v in ref_input_json.items():
                        if k in cand_input_json.keys():
                            if cand_input_json[k] == v:
                                full_match += 1
                            else:
                                half_match += 1

                    recall = (0.5 * half_match + full_match) / (len(ref_input_json) + 1e-30)
                    precision = (0.5 * half_match + full_match) / (len(cand_input_json) + 1e-30)
                    try:
                        f1.append((2 * recall * precision) / (recall + precision))
                    except Exception:
                        f1.append(0.0)

        if f1[0] == 1.0:
            return True
        else:
            return False

    @staticmethod
    def parse_action(text):
        if 'Action Input:' in text:
            input_idx = text.rindex('Action Input:')
            action_input = text[input_idx + len('Action Input:'):].strip()
        else:
            action_input = '{}'

        if 'Action:' in text:
            action_idx = text.rindex('Action:')
            action = text[action_idx + len('Action:'):].strip()
            if 'Action Input:' in action:
                input_idx = action.index('Action Input:')
                action = action[:input_idx].strip()
        else:
            action = 'none'
        return action, action_input

    @staticmethod
    def parse_output(text):
        action, action_input = ReactORM.parse_action(text)
        return action, action_input

    def __call__(self, infer_requests: List[Union['InferRequest', Dict]], solution: List[str], **kwargs) -> List[float]:
        rewards = []
        if not isinstance(infer_requests[0], str):
            predictions = [request['messages'][-1]['content'] for request in infer_requests]
        else:
            predictions = infer_requests
        for prediction, ground_truth in zip(predictions, solution):
            if prediction.endswith('Observation:'):
                prediction = prediction[:prediction.index('Observation:')].strip()
            action_ref = []
            action_input_ref = []
            action_pred = []
            action_input_pred = []
            reference = ground_truth
            prediction = prediction.replace('<|endoftext|>', '').replace('<|im_end|>', '').strip()
            ref_action, ref_input = ReactORM.parse_output(reference)
            pred_action, pred_input = ReactORM.parse_output(prediction)
            action_ref.append(ref_action)
            action_input_ref.append(ref_input)
            if pred_action is None:
                action_pred.append('none')
            else:
                action_pred.append(pred_action)

            if pred_input is None:
                action_input_pred.append('{}')
            else:
                action_input_pred.append(pred_input)

            reward = ReactORM.evaluate_action_reward(action_pred, action_ref, action_input_pred, action_input_ref)
            rewards.append(float(reward))
        return rewards

    @staticmethod
    def evaluate_rougel(cand_list: list, ref_list: list):
        if len(ref_list) == 0:
            return None
        try:
            from rouge import Rouge
            rouge = Rouge()
            rouge_score = rouge.get_scores(hyps=cand_list, refs=ref_list, avg=True)
            rougel = rouge_score['rouge-l']['f']
            return rougel
        except Exception:
            return None


class MathORM(ORM):

    def __init__(self):
        from transformers.utils import strtobool
        self.use_opencompass = strtobool(os.environ.get('USE_OPENCOMPASS_EVALUATOR', 'False'))
        if self.use_opencompass:
            from opencompass.datasets.math import MATHEvaluator
            self.evaluator = MATHEvaluator()

    @staticmethod
    def check_terminate(answers: Union[str, List[str]]) -> List[bool]:
        if isinstance(answers, str):
            answers = [answers]
        results = []
        for answer in answers:
            results.append('\\boxed' in answer)
        return results

    @staticmethod
    def extract_boxed_result(text):
        pattern = r'\\boxed{([^}]*)}'
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
        else:
            return text

    @staticmethod
    def clean_latex(latex_str):
        latex_str = re.sub(r'\\\(|\\\)|\\\[|\\]', '', latex_str)
        latex_str = latex_str.replace('}}', '}').replace('{', '').replace('}', '')
        return latex_str.strip()

    @staticmethod
    def parse_expression(latex_str):
        from sympy import simplify
        from sympy.parsing.latex import parse_latex
        try:
            expr = parse_latex(latex_str)
            return simplify(expr)
        except Exception:
            return None

    @staticmethod
    def compare_consecutive(first, second):
        cleaned_list = [MathORM.clean_latex(latex) for latex in [first, second]]
        parsed_exprs = [MathORM.parse_expression(latex) for latex in cleaned_list]
        if hasattr(parsed_exprs[0], 'equals') and hasattr(parsed_exprs[1], 'equals'):
            value = parsed_exprs[0].equals(parsed_exprs[1])
        else:
            value = parsed_exprs[0] == parsed_exprs[1]
        if value is None:
            value = False
        return value

    def __call__(self, infer_requests: List[Union['InferRequest', Dict]], ground_truths: List[str],
                 **kwargs) -> List[float]:
        rewards = []
        predictions = [request.messages[-1]['content'] for request in infer_requests]
        for prediction, ground_truth in zip(predictions, ground_truths):
            if '# Answer' in prediction:
                prediction = prediction.split('# Answer')[1]
            if '# Answer' in ground_truth:
                ground_truth = ground_truth.split('# Answer')[1]
            prediction = prediction.strip()
            ground_truth = ground_truth.strip()
            prediction = MathORM.extract_boxed_result(prediction)
            ground_truth = MathORM.extract_boxed_result(ground_truth)
            if self.use_opencompass:
                reward = self.evaluator.is_equiv(prediction, ground_truth)
            else:
                reward = MathORM.compare_consecutive(prediction, ground_truth)
            rewards.append(float(reward))
        return rewards


class MathAccuracy(ORM):

    def __init__(self):
        import importlib.util
        assert importlib.util.find_spec('math_verify') is not None, (
            'The math_verify package is required but not installed. '
            "Please install it using 'pip install math_verify==0.5.2'.")

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        from latex2sympy2_extended import NormalizationConfig
        from math_verify import LatexExtractionConfig, parse, verify
        rewards = []
        for content, sol in zip(completions, solution):
            gold_parsed = parse(sol, extraction_mode='first_match')
            if len(gold_parsed) != 0:
                # We require the answer to be provided in correct latex (no malformed operators)
                answer_parsed = parse(
                    content,
                    extraction_config=[
                        LatexExtractionConfig(
                            normalization_config=NormalizationConfig(
                                nits=False,
                                malformed_operators=False,
                                basic_latex=True,
                                equations=True,
                                boxed=True,
                                units=True,
                            ),
                            # Ensures that boxed is tried first
                            boxed_match_priority=0,
                            try_extract_without_anchor=False,
                        )
                    ],
                    extraction_mode='first_match',
                )
                # edge case
                try:
                    reward = float(verify(gold_parsed, answer_parsed))
                except Exception:
                    reward = 0.0
            else:
                # If the gold solution is not parseable, we reward 0 to skip this example
                reward = 0.0
            rewards.append(reward)
        return rewards


class Format(ORM):

    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that checks if the completion has a specific format."""
        pattern = r'^<think>.*?</think>\s*<answer>.*?</answer>(?![\s\S])'
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
        return [1.0 if match else 0.0 for match in matches]


class ReActFormat(ORM):

    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that checks if the completion has a specific format."""
        pattern = r'^<think>.*?</think>\s*Action:.*?Action Input:.*?$'
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
        return [1.0 if match else 0.0 for match in matches]


class CosineReward(ORM):
    # https://arxiv.org/abs/2502.03373
    def __init__(self,
                 tokenizer=None,
                 cosine_min_len_value_wrong: float = -0.5,
                 cosine_max_len_value_wrong: float = 0.0,
                 cosine_min_len_value_correct: float = 1.0,
                 cosine_max_len_value_correct: float = 0.5,
                 cosine_max_len: int = 1000,
                 accuracy_orm=None):
        self.tokenizer = tokenizer
        self.min_len_value_wrong = cosine_min_len_value_wrong
        self.max_len_value_wrong = cosine_max_len_value_wrong
        self.min_len_value_correct = cosine_min_len_value_correct
        self.max_len_value_correct = cosine_max_len_value_correct
        self.max_len = cosine_max_len
        self.accuracy_orm = accuracy_orm or MathAccuracy()

    @staticmethod
    def cosfn(t, T, min_value, max_value):
        import math
        return max_value - (max_value - min_value) * (1 - math.cos(t * math.pi / T)) / 2

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        acc_rewards = self.accuracy_orm(completions, solution, **kwargs)
        rewards = []
        for content, acc_reward in zip(completions, acc_rewards):
            is_correct = acc_reward >= 1.
            if is_correct:
                # Swap min/max for correct answers
                min_value = self.max_len_value_correct
                max_value = self.min_len_value_correct
            else:
                min_value = self.max_len_value_wrong
                max_value = self.min_len_value_wrong
            gen_len = len(self.tokenizer.encode(content))
            reward = self.cosfn(gen_len, self.max_len, min_value, max_value)
            rewards.append(reward)
        return rewards


class RepetitionPenalty(ORM):
    # https://arxiv.org/abs/2502.03373
    def __init__(self, repetition_n_grams: int = 3, repetition_max_penalty: float = -1.0):
        self.ngram_size = repetition_n_grams
        self.max_penalty = repetition_max_penalty

    @staticmethod
    def zipngram(text: str, ngram_size: int):
        words = text.lower().split()
        return zip(*[words[i:] for i in range(ngram_size)])

    def __call__(self, completions, **kwargs) -> List[float]:
        """
        reward function the penalizes repetitions

        Args:
            completions: List of model completions
        """
        rewards = []
        for completion in completions:
            if completion == '':
                rewards.append(0.0)
                continue
            if len(completion.split()) < self.ngram_size:
                rewards.append(0.0)
                continue

            ngrams = set()
            total = 0
            for ng in self.zipngram(completion, self.ngram_size):
                ngrams.add(ng)
                total += 1

            scaling = 1 - len(ngrams) / total
            reward = scaling * self.max_penalty
            rewards.append(reward)
        return rewards


class SoftOverlong(ORM):

    def __init__(self, tokenizer, soft_max_length, soft_cache_length):
        self.tokenizer = tokenizer
        assert soft_cache_length < soft_max_length
        self.soft_max_length = soft_max_length
        self.soft_cache_length = soft_cache_length

    def __call__(self, completions, **kwargs) -> List[float]:
        rewards = []
        for completion in completions:
            completion_length = len(self.tokenizer.encode(completion))
            expected_len = self.soft_max_length - self.soft_cache_length
            exceed_len = completion_length - expected_len
            rewards.append(min(-exceed_len / self.soft_cache_length, 0))
        return rewards    

# -------------------------------------------------------------------Newly add reward for C-STS------------------------------------------------------------------------------------

def extract_yes_no_and_score(completion: str) -> Tuple[str, float]:
    """Extract yes/no judgment and score from completion (for binary prompt format).
    Returns (judgment: 'yes'/'no' or '', score: float or 0.0 if invalid).
    """
    pattern = r'<answer>(yes|no)\((\d)\)</answer>'
    match = re.search(pattern, completion, re.DOTALL)
    if match:
        judgment = match.group(1).lower()
        score = float(match.group(2))
        if judgment in ['yes', 'no'] and 1 <= score <= 5:
            return judgment, score
    return '', 0.0  # Invalid extraction

def extract_score(completion: str) -> float:
    """Extract score from completion using regex (for both yes/no(score) and direct score formats)."""
    # For yes/no(score) format
    yes_no_pattern = r'<answer>(yes|no)\((\d)\)</answer>'
    yes_no_match = re.search(yes_no_pattern, completion, re.DOTALL)
    if yes_no_match:
        return float(yes_no_match.group(2))
    
    # For direct score format
    score_pattern = r'<answer>(\d)</answer>'
    score_match = re.search(score_pattern, completion, re.DOTALL)
    if score_match:
        return float(score_match.group(1))
    
    return 0.0  # Invalid, will get low reward


class YesNoFormatReward(ORM):
    """
    Format and Logic reward for yes(score)/no(score) answers.
    Checks:
    1. If output matches the regex <answer>yes/no(1-5)</answer>.
    2. If the score is logically consistent with the yes/no judgment.
       - 'yes' must have a score of 3, 4, or 5.
       - 'no' must have a score of 1 or 2.
    Reward is +1 if both checks pass, otherwise 0.
    """
    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        rewards = []
        pattern = r'^<think>.*?</think>\s*<answer>(yes|no)\(([1-5])\)</answer>'
        for content in completions:
            match = re.match(pattern, content.strip(), re.DOTALL)
            if not match:
                rewards.append(0.0)
                continue
            judgment = match.group(1)
            score = int(match.group(2))
            
            # NEW: Add logical consistency check
            is_consistent = False
            if judgment == 'yes' and score >= 3:
                is_consistent = True
            elif judgment == 'no' and score <= 2:
                is_consistent = True
            # Final reward is 1.0 only if both format and logic are correct
            rewards.append(1.0 if is_consistent else 0.0)
            
        return rewards

class BinaryJudgmentReward(ORM):
    """
    Pointwise binary judgment reward for yes/no classification.
    Rewards correct binary classification (yes for labels 3-5, no for 1-2).
    Reward range: 0 (wrong) to +1 (correct).
    """
    def __call__(self, completions: List[str], label: List[float], **kwargs) -> List[float]:
        rewards = []
        for completion, true_label in zip(completions, label):
            pred_judgment, _ = extract_yes_no_and_score(completion)
            if not pred_judgment:  # Invalid format
                rewards.append(0.0)
                continue
            
            # Map true label to binary ground truth
            true_binary = 'yes' if true_label >= 3 else 'no'
            
            # Reward: +1 for match, 0 for mismatch
            if pred_judgment == true_binary:
                reward = 1.0
            else:
                reward = 0.0
            
            rewards.append(reward)
        
        return rewards

    
class PointwiseMAEReward(ORM):
    """
    Pointwise MAE (Mean Absolute Error) reward, normalized to [0, 1].
    Reward = 1 - (normalized_mae). A perfect match gets +1, max error gets 0.
    """
    def __call__(self, completions: List[str], label: List[float], **kwargs) -> List[float]:
        rewards = []
        # Max possible MAE for scores in [1, 5] is (5-1) = 4.0
        max_mae = 4.0
        
        for completion, true_label in zip(completions, label):
            pred_score = extract_score(completion)
            
            if pred_score == 0.0:
                rewards.append(0.0)
                continue
            
            mae = abs(pred_score - true_label)
            reward = 1.0 - (mae / max_mae)
            rewards.append(np.clip(reward, 0.0, 1.0))
        
        return rewards


class ParallelBatchReward(ORM):
    """
    Simplified parallel batch reward base class (single-process version).
    Provides common functionality for batch reorganization and reward redistribution.
    Subclasses only need to implement specific reward calculation logic.
    """
    def __init__(self, num_generations):
        """
        Args:
            num_generations: Number of completions generated per prompt (K)
        """
        self.num_generations = num_generations
        logger.info(f"Initialized ParallelBatchReward with num_generations={num_generations}")

    def _extract_scores(self, completions: List[str]) -> List[float]:
        """Extract all scores from completions"""
        scores = []
        for comp in completions:
            score = extract_score(comp)
            # If format is incorrect or score not found, use a neutral/default value (e.g. 0)
            # This prevents subsequent calculation errors, but format errors should be penalized by FormatReward
            scores.append(float(score) if score is not None else 0.0)
        return scores

    def reorganize_to_parallel_batches(self, completions: List[str], labels: List[float]) -> Tuple[List[List[float]], List[List[float]], int]:
        """
        Reorganize N×K samples into K parallel batches of size N.
        """
        total_samples = len(completions)
        if total_samples == 0:
            return [], [], 0
            
        batch_size = total_samples // self.num_generations  # N
        
        if total_samples % self.num_generations != 0:
            logger.warning(f"Total samples ({total_samples}) not divisible by num_generations ({self.num_generations}). Skipping reward.")
            return [], [], 0
        
        # Extract predicted scores from completions
        predicted_scores = self._extract_scores(completions)
        
        # 1. Convert 1D list to (N, G) 2D array
        preds_array = np.array(predicted_scores).reshape(batch_size, self.num_generations)
        # Note: labels also need same processing, assuming input labels already have length N*K
        labels_array = np.array(labels).reshape(batch_size, self.num_generations)

        # 2. Transpose array to shape (G, N)
        parallel_preds_array = preds_array.transpose()
        parallel_labels_array = labels_array.transpose()
        
        # 3. Convert numpy arrays back to list of lists
        parallel_batches_pred = parallel_preds_array.tolist()
        parallel_batches_label = parallel_labels_array.tolist()
        
        return parallel_batches_pred, parallel_batches_label, batch_size

    def compute_batch_rewards(self, parallel_batches_pred: List[List[float]], 
                             parallel_batches_label: List[List[float]]) -> List[Union[float, List[float]]]:
        """
        Compute rewards for each parallel batch (abstract method).
        """
        raise NotImplementedError("Subclasses must implement compute_batch_rewards")

    def redistribute_rewards(self, batch_rewards: List[Union[float, List[float]]], batch_size: int) -> List[float]:
        """
        Redistribute batch rewards back to samples in original input order.
        (This logic is identical to your previous code)
        """
        if not batch_rewards:
            return []

        is_shared_reward = not isinstance(batch_rewards[0], list)

        if is_shared_reward:
            rewards_array = np.array(batch_rewards)[:, np.newaxis] * np.ones(batch_size)
        else:
            rewards_array = np.array(batch_rewards)

        redistributed_array = rewards_array.transpose()
        rewards = redistributed_array.flatten().tolist()
        
        return rewards

    def __call__(self, completions: List[str], label: List[float], **kwargs) -> List[float]:
        """
        Main call interface (simplified version, no distributed logic).
        """
        parallel_batches_pred, parallel_batches_label, batch_size = self.reorganize_to_parallel_batches(
            completions, label
        )
        
        if not parallel_batches_pred:
            return [0.0] * len(completions)

        batch_rewards = self.compute_batch_rewards(parallel_batches_pred, parallel_batches_label)
        
        final_rewards = self.redistribute_rewards(batch_rewards, batch_size)
        
        return final_rewards


class RankDifferenceReward(ParallelBatchReward):
    """
    Fine-grained ranking difference reward computed within parallel batch.
    """
    
    def __init__(self, num_generations: int):
        super().__init__(num_generations)
        logger.info(f"RankDifferenceReward initialized with num_generations={num_generations}")

    def compute_batch_rewards(self, parallel_batches_pred: List[List[float]], 
                              parallel_batches_label: List[List[float]]) -> List[List[float]]:
        """
        Compute ranking difference rewards for each parallel batch.
        Returns independent rewards for each sample.
        """
        all_rewards = []
        
        for batch_pred, batch_label in zip(parallel_batches_pred, parallel_batches_label):
            try:
                batch_size = len(batch_pred)
                if batch_size < 2:
                    all_rewards.append([0.0] * batch_size)
                    continue

                # Pass negative values to achieve descending ranking
                pred_ranks = rankdata([-s for s in batch_pred], method='average')
                true_ranks = rankdata([-s for s in batch_label], method='average')
                
                rank_diffs = np.abs(pred_ranks - true_ranks)
                max_possible_diff = batch_size - 1
                
                batch_rewards = 1.0 - (rank_diffs / max_possible_diff)
                
                clipped_rewards = np.clip(batch_rewards, 0.0, 1.0).tolist()
                all_rewards.append(clipped_rewards)
                
            except Exception as e:
                logger.warning(f"Error computing ranking differences: {e}")
                all_rewards.append([0.0] * len(batch_pred))
        
        return all_rewards


class PairwisePreferenceReward(ParallelBatchReward):
    """
    Pair-wise preference reward computed within within parallel batch.
    """
    
    def __init__(self, num_generations: int):
        super().__init__(num_generations)
        logger.info(f"PairwisePreferenceReward initialized with num_generations={num_generations}")

    def compute_batch_rewards(self, parallel_batches_pred: List[List[float]], 
                          parallel_batches_label: List[List[float]]) -> List[List[float]]:

        all_rewards = [] 
        
        # Iterate parallel batches
        for k in range(self.num_generations):
            batch_pred_scores = parallel_batches_pred[k]
            batch_true_labels = parallel_batches_label[k]
            
            num_prompts = len(batch_pred_scores) # N
            rewards_for_this_batch = [0.0] * num_prompts
            
            for i in range(0, num_prompts - 1, 2):
                # Extract predictions and ground truth for adjacent sample pair A and B
                pred_score_A = batch_pred_scores[i]
                true_label_A = batch_true_labels[i]
                
                pred_score_B = batch_pred_scores[i+1]
                true_label_B = batch_true_labels[i+1]

                # Compute true difference and predicted difference
                true_diff = true_label_A - true_label_B
                pred_diff = pred_score_A - pred_score_B

                # 1. First determine if basic ranking preference is correct
                is_preference_correct = False
                if (true_diff > 0 and pred_diff > 0) or \
                (true_diff == 0 and pred_diff == 0):
                    is_preference_correct = True
                
                # ranking preference is wrong, reward is directly 0
                if not is_preference_correct:
                    reward_for_pair = 0.0
                else:
                    # 2. If ranking is correct, give base reward and compute additional reward
                    base_reward = 0.5 
                    
                    # Compute difference error to determine additional reward amount
                    diff_error = abs(pred_diff - true_diff)
                    # Maximum possible difference error, e.g. true 5/1(diff 4) vs predicted 5/4(diff 1), error is 3
                    max_diff_error = 3.0
                    
                    # Additional reward is inversely proportional to difference error, range [0, 0.5]
                    # (1.0 - base_reward) is the maximum value of additional reward
                    bonus_reward = (1.0 - base_reward) * (1.0 - (diff_error / max_diff_error))
                    bonus_reward = np.clip(bonus_reward, 0.0, 1.0 - base_reward)

                    # Final reward = base score + additional score
                    reward_for_pair = base_reward + bonus_reward
                
                # Assign computed reward to this pair of samples
                rewards_for_this_batch[i] = reward_for_pair
                rewards_for_this_batch[i+1] = reward_for_pair
                
            all_rewards.append(rewards_for_this_batch)
            
        return all_rewards


orms = {
    'toolbench': ReactORM,
    'math': MathORM,
    'accuracy': MathAccuracy,
    'format': Format,
    'react_format': ReActFormat,
    'cosine': CosineReward,
    'repetition': RepetitionPenalty,
    'soft_overlong': SoftOverlong,
    'yes_no_format': YesNoFormatReward,
    'binary_judgment': BinaryJudgmentReward,
    'pointwise_mae': PointwiseMAEReward,
    'rank_difference': RankDifferenceReward,
    'pairwise_preference': PairwisePreferenceReward,
}
