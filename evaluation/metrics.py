import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Set
from collections import defaultdict, Counter
import re
import string
from dataclasses import dataclass
import logging
from pathlib import Path
import json
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import sacrebleu
from sentence_transformers import SentenceTransformer
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import spacy
from transformers import AutoTokenizer, AutoModel


try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


@dataclass
class MetricConfig:
    bleu_smoothing: str = "method1"
    bleu_max_order: int = 4
    
    rouge_types: List[str] = None
    rouge_use_stemmer: bool = True
    
    radgraph_reward_level: str = "partial"
    
    clinical_keywords_file: Optional[str] = None
    anatomy_keywords: List[str] = None
    pathology_keywords: List[str] = None
    
    sentence_transformer_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    clinical_bert_model: str = "emilyalsentzer/Bio_ClinicalBERT"
    
    radcliq_weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.rouge_types is None:
            self.rouge_types = ["rouge1", "rouge2", "rougeL"]
        
        if self.anatomy_keywords is None:
            self.anatomy_keywords = [
                "lung", "lungs", "heart", "cardiac", "chest", "thorax", "mediastinum",
                "pleura", "diaphragm", "ribs", "spine", "vertebra", "clavicle",
                "sternum", "trachea", "bronchi", "vessels", "aorta", "pulmonary"
            ]
        
        if self.pathology_keywords is None:
            self.pathology_keywords = [
                "pneumonia", "consolidation", "opacity", "infiltrate", "nodule",
                "mass", "tumor", "cardiomegaly", "edema", "effusion", "pneumothorax",
                "atelectasis", "emphysema", "fibrosis", "fracture", "infection",
                "malignancy", "abnormal", "abnormality", "lesion", "disease"
            ]
        
        if self.radcliq_weights is None:
            self.radcliq_weights = {
                "bleu": 0.25,
                "bertscore": 0.25,
                "sembscore": 0.25,
                "radgraph_f1": 0.25
            }


class BLEUMetric:
    
    def __init__(self, config: MetricConfig):
        self.config = config
        self.smoothing_fn = getattr(SmoothingFunction(), config.bleu_smoothing)
    
    def compute_bleu(
        self,
        predictions: List[str],
        references: List[List[str]],
        max_order: int = None
    ) -> Dict[str, float]:
        if max_order is None:
            max_order = self.config.bleu_max_order
        
        bleu_scores = {}
        
        tokenized_preds = [self._tokenize(pred) for pred in predictions]
        tokenized_refs = [[self._tokenize(ref) for ref in ref_list] for ref_list in references]
        
        for n in range(1, max_order + 1):
            weights = [1.0/n] * n + [0.0] * (4 - n)
            
            scores = []
            for pred, ref_list in zip(tokenized_preds, tokenized_refs):
                score = sentence_bleu(
                    ref_list, pred, 
                    weights=weights,
                    smoothing_function=self.smoothing_fn
                )
                scores.append(score)
            
            bleu_scores[f'bleu_{n}'] = np.mean(scores)
        
        try:
            corpus_bleu = sacrebleu.corpus_bleu(predictions, list(zip(*references)))
            bleu_scores['corpus_bleu'] = corpus_bleu.score / 100.0
        except:
            bleu_scores['corpus_bleu'] = bleu_scores.get('bleu_4', 0.0)
        
        return bleu_scores
    
    def _tokenize(self, text: str) -> List[str]:
        text = text.lower().translate(str.maketrans('', '', string.punctuation))
        return text.split()


class ROUGEMetric:
    
    def __init__(self, config: MetricConfig):
        self.config = config
        self.scorer = rouge_scorer.RougeScorer(
            config.rouge_types, 
            use_stemmer=config.rouge_use_stemmer
        )
    
    def compute_rouge(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        rouge_scores = defaultdict(list)
        
        for pred, ref in zip(predictions, references):
            scores = self.scorer.score(ref, pred)
            
            for rouge_type, score in scores.items():
                rouge_scores[f'{rouge_type}_precision'].append(score.precision)
                rouge_scores[f'{rouge_type}_recall'].append(score.recall)
                rouge_scores[f'{rouge_type}_f1'].append(score.fmeasure)
        
        final_scores = {}
        for key, values in rouge_scores.items():
            final_scores[key] = np.mean(values)
        
        return final_scores


class CIDErMetric:
    
    def __init__(self):
        pass
    
    def compute_cider(
        self,
        predictions: List[str],
        references: List[List[str]]
    ) -> Dict[str, float]:
        tokenized_preds = [self._tokenize(pred) for pred in predictions]
        tokenized_refs = [[self._tokenize(ref) for ref in ref_list] for ref_list in references]
        
        pred_ngrams = [self._get_ngrams(pred, 4) for pred in tokenized_preds]
        ref_ngrams = [[self._get_ngrams(ref, 4) for ref in ref_list] for ref_list in tokenized_refs]
        
        doc_freqs = self._compute_doc_frequencies(ref_ngrams)
        
        cider_scores = []
        for pred_ng, ref_ng_list in zip(pred_ngrams, ref_ngrams):
            score = self._compute_cider_score(pred_ng, ref_ng_list, doc_freqs)
            cider_scores.append(score)
        
        return {'cider': np.mean(cider_scores)}
    
    def _tokenize(self, text: str) -> List[str]:
        return text.lower().split()
    
    def _get_ngrams(self, tokens: List[str], n: int) -> Dict[Tuple[str, ...], int]:
        ngrams = defaultdict(int)
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            ngrams[ngram] += 1
        return dict(ngrams)
    
    def _compute_doc_frequencies(self, ref_ngrams: List[List[Dict]]) -> Dict[Tuple[str, ...], int]:
        doc_freqs = defaultdict(int)
        
        for ref_list in ref_ngrams:
            seen_ngrams = set()
            for ref_ng in ref_list:
                for ngram in ref_ng:
                    if ngram not in seen_ngrams:
                        doc_freqs[ngram] += 1
                        seen_ngrams.add(ngram)
        
        return dict(doc_freqs)
    
    def _compute_cider_score(
        self,
        pred_ngrams: Dict[Tuple[str, ...], int],
        ref_ngrams_list: List[Dict[Tuple[str, ...], int]],
        doc_freqs: Dict[Tuple[str, ...], int]
    ) -> float:
        score = 0.0
        
        for n in range(1, 5):
            pred_n = {k: v for k, v in pred_ngrams.items() if len(k) == n}
            ref_n_list = [{k: v for k, v in ref_ng.items() if len(k) == n} for ref_ng in ref_ngrams_list]
            
            if not pred_n:
                continue
            
            similarities = []
            for ref_n in ref_n_list:
                sim = self._cosine_similarity(pred_n, ref_n, doc_freqs)
                similarities.append(sim)
            
            score += np.mean(similarities)
        
        return score / 4.0
    
    def _cosine_similarity(
        self,
        pred_ngrams: Dict[Tuple[str, ...], int],
        ref_ngrams: Dict[Tuple[str, ...], int],
        doc_freqs: Dict[Tuple[str, ...], int]
    ) -> float:
        pred_tfidf = {}
        ref_tfidf = {}
        
        total_docs = len(doc_freqs)
        
        for ngram, count in pred_ngrams.items():
            tf = count / sum(pred_ngrams.values())
            idf = np.log(total_docs / (doc_freqs.get(ngram, 1) + 1))
            pred_tfidf[ngram] = tf * idf
        
        for ngram, count in ref_ngrams.items():
            tf = count / sum(ref_ngrams.values())
            idf = np.log(total_docs / (doc_freqs.get(ngram, 1) + 1))
            ref_tfidf[ngram] = tf * idf
        
        all_ngrams = set(pred_tfidf.keys()) | set(ref_tfidf.keys())
        
        if not all_ngrams:
            return 0.0
        
        dot_product = sum(pred_tfidf.get(ng, 0) * ref_tfidf.get(ng, 0) for ng in all_ngrams)
        pred_norm = np.sqrt(sum(v**2 for v in pred_tfidf.values()))
        ref_norm = np.sqrt(sum(v**2 for v in ref_tfidf.values()))
        
        if pred_norm == 0 or ref_norm == 0:
            return 0.0
        
        return dot_product / (pred_norm * ref_norm)


class METEORMetric:
    
    def __init__(self):
        pass
    
    def compute_meteor(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        meteor_scores = []
        
        for pred, ref in zip(predictions, references):
            try:
                score = meteor_score([ref.split()], pred.split())
                meteor_scores.append(score)
            except:
                meteor_scores.append(0.0)
        
        return {'meteor': np.mean(meteor_scores)}


class BERTScoreMetric:
    
    def __init__(self, config: MetricConfig):
        self.config = config
        self.model = SentenceTransformer(config.sentence_transformer_model)
    
    def compute_bertscore(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        pred_embeddings = self.model.encode(predictions)
        ref_embeddings = self.model.encode(references)
        
        similarities = []
        for pred_emb, ref_emb in zip(pred_embeddings, ref_embeddings):
            sim = np.dot(pred_emb, ref_emb) / (np.linalg.norm(pred_emb) * np.linalg.norm(ref_emb))
            similarities.append(sim)
        
        bert_score = np.mean(similarities)
        
        return {
            'bertscore_f1': bert_score,
            'bertscore_precision': bert_score,
            'bertscore_recall': bert_score
        }


class RadGraphF1Metric:
    
    def __init__(self, config: MetricConfig):
        self.config = config
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logging.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def compute_radgraph_f1(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        if self.nlp is None:
            return self._compute_simple_entity_f1(predictions, references)
        
        pred_entities = [self._extract_clinical_entities(pred) for pred in predictions]
        ref_entities = [self._extract_clinical_entities(ref) for ref in references]
        
        entity_f1_scores = []
        for pred_ents, ref_ents in zip(pred_entities, ref_entities):
            f1 = self._compute_entity_f1(pred_ents, ref_ents)
            entity_f1_scores.append(f1)
        
        return {
            'radgraph_f1': np.mean(entity_f1_scores),
            'radgraph_precision': np.mean(entity_f1_scores),
            'radgraph_recall': np.mean(entity_f1_scores)
        }
    
    def _extract_clinical_entities(self, text: str) -> Set[str]:
        doc = self.nlp(text.lower())
        entities = set()
        
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORG", "GPE"]:
                continue
            entities.add(ent.text.lower())
        
        words = text.lower().split()
        for word in words:
            if word in self.config.anatomy_keywords or word in self.config.pathology_keywords:
                entities.add(word)
        
        return entities
    
    def _compute_simple_entity_f1(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        all_keywords = set(self.config.anatomy_keywords + self.config.pathology_keywords)
        
        f1_scores = []
        for pred, ref in zip(predictions, references):
            pred_words = set(pred.lower().split())
            ref_words = set(ref.lower().split())
            
            pred_entities = pred_words & all_keywords
            ref_entities = ref_words & all_keywords
            
            f1 = self._compute_entity_f1(pred_entities, ref_entities)
            f1_scores.append(f1)
        
        return {
            'radgraph_f1': np.mean(f1_scores),
            'radgraph_precision': np.mean(f1_scores),
            'radgraph_recall': np.mean(f1_scores)
        }
    
    def _compute_entity_f1(self, pred_entities: Set[str], ref_entities: Set[str]) -> float:
        if not ref_entities:
            return 1.0 if not pred_entities else 0.0
        
        if not pred_entities:
            return 0.0
        
        intersection = pred_entities & ref_entities
        precision = len(intersection) / len(pred_entities)
        recall = len(intersection) / len(ref_entities)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * precision * recall / (precision + recall)


class ClinicalAccuracyMetric:
    
    def __init__(self, config: MetricConfig):
        self.config = config
    
    def compute_clinical_accuracy(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        keyword_scores = self._compute_keyword_accuracy(predictions, references)
        
        anatomy_scores = self._compute_anatomy_accuracy(predictions, references)
        
        pathology_scores = self._compute_pathology_accuracy(predictions, references)
        
        normality_scores = self._compute_normality_accuracy(predictions, references)
        
        clinical_accuracy = np.mean([
            keyword_scores['accuracy'],
            anatomy_scores['accuracy'],
            pathology_scores['accuracy'],
            normality_scores['accuracy']
        ])
        
        return {
            'clinical_accuracy': clinical_accuracy,
            'keyword_accuracy': keyword_scores['accuracy'],
            'anatomy_accuracy': anatomy_scores['accuracy'],
            'pathology_accuracy': pathology_scores['accuracy'],
            'normality_accuracy': normality_scores['accuracy']
        }
    
    def _compute_keyword_accuracy(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        all_keywords = set(self.config.anatomy_keywords + self.config.pathology_keywords)
        
        matches = 0
        total = 0
        
        for pred, ref in zip(predictions, references):
            pred_words = set(pred.lower().split())
            ref_words = set(ref.lower().split())
            
            pred_keywords = pred_words & all_keywords
            ref_keywords = ref_words & all_keywords
            
            matches += len(pred_keywords & ref_keywords)
            total += len(ref_keywords | pred_keywords)
        
        accuracy = matches / total if total > 0 else 1.0
        return {'accuracy': accuracy}
    
    def _compute_anatomy_accuracy(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        anatomy_keywords = set(self.config.anatomy_keywords)
        
        correct = 0
        total = len(predictions)
        
        for pred, ref in zip(predictions, references):
            pred_anatomy = set(pred.lower().split()) & anatomy_keywords
            ref_anatomy = set(ref.lower().split()) & anatomy_keywords
            
            if pred_anatomy == ref_anatomy:
                correct += 1
        
        return {'accuracy': correct / total if total > 0 else 1.0}
    
    def _compute_pathology_accuracy(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        pathology_keywords = set(self.config.pathology_keywords)
        
        correct = 0
        total = len(predictions)
        
        for pred, ref in zip(predictions, references):
            pred_pathology = bool(set(pred.lower().split()) & pathology_keywords)
            ref_pathology = bool(set(ref.lower().split()) & pathology_keywords)
            
            if pred_pathology == ref_pathology:
                correct += 1
        
        return {'accuracy': correct / total if total > 0 else 1.0}
    
    def _compute_normality_accuracy(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        normal_indicators = {"normal", "unremarkable", "clear", "negative"}
        abnormal_indicators = {"abnormal", "positive", "findings", "opacity", "consolidation"}
        
        correct = 0
        total = len(predictions)
        
        for pred, ref in zip(predictions, references):
            pred_words = set(pred.lower().split())
            ref_words = set(ref.lower().split())
            
            pred_normal = bool(pred_words & normal_indicators)
            pred_abnormal = bool(pred_words & abnormal_indicators)
            ref_normal = bool(ref_words & normal_indicators)
            ref_abnormal = bool(ref_words & abnormal_indicators)
            
            if pred_normal and not pred_abnormal:
                pred_class = "normal"
            elif pred_abnormal and not pred_normal:
                pred_class = "abnormal"
            else:
                pred_class = "unclear"
            
            if ref_normal and not ref_abnormal:
                ref_class = "normal"
            elif ref_abnormal and not ref_normal:
                ref_class = "abnormal"
            else:
                ref_class = "unclear"
            
            if pred_class == ref_class:
                correct += 1
        
        return {'accuracy': correct / total if total > 0 else 1.0}


class RadCliQMetric:
    
    def __init__(self, config: MetricConfig):
        self.config = config
        self.bleu_metric = BLEUMetric(config)
        self.bertscore_metric = BERTScoreMetric(config)
        self.radgraph_metric = RadGraphF1Metric(config)
        
        self.semb_model = SentenceTransformer(config.clinical_bert_model)
    
    def compute_radcliq(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        bleu_references = [[ref] for ref in references]
        
        bleu_scores = self.bleu_metric.compute_bleu(predictions, bleu_references)
        bertscore_scores = self.bertscore_metric.compute_bertscore(predictions, references)
        radgraph_scores = self.radgraph_metric.compute_radgraph_f1(predictions, references)
        semb_scores = self._compute_sembscore(predictions, references)
        
        bleu_score = bleu_scores.get('bleu_4', 0.0)
        bertscore = bertscore_scores.get('bertscore_f1', 0.0)
        radgraph_f1 = radgraph_scores.get('radgraph_f1', 0.0)
        sembscore = semb_scores.get('sembscore', 0.0)
        
        radcliq_score = (
            self.config.radcliq_weights['bleu'] * bleu_score +
            self.config.radcliq_weights['bertscore'] * bertscore +
            self.config.radcliq_weights['sembscore'] * sembscore +
            self.config.radcliq_weights['radgraph_f1'] * radgraph_f1
        )
        
        radcliq_v1 = 1.0 / (1.0 + radcliq_score)
        
        return {
            'radcliq': radcliq_score,
            'radcliq_v1': radcliq_v1,
            'radcliq_bleu': bleu_score,
            'radcliq_bertscore': bertscore,
            'radcliq_sembscore': sembscore,
            'radcliq_radgraph_f1': radgraph_f1
        }
    
    def _compute_sembscore(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        pred_embeddings = self.semb_model.encode(predictions)
        ref_embeddings = self.semb_model.encode(references)
        
        similarities = []
        for pred_emb, ref_emb in zip(pred_embeddings, ref_embeddings):
            sim = np.dot(pred_emb, ref_emb) / (np.linalg.norm(pred_emb) * np.linalg.norm(ref_emb))
            similarities.append(sim)
        
        return {'sembscore': np.mean(similarities)}


class RadixpertEvaluator:
    
    def __init__(self, config: MetricConfig = None):
        if config is None:
            config = MetricConfig()
        self.config = config
        
        self.bleu_metric = BLEUMetric(config)
        self.rouge_metric = ROUGEMetric(config)
        self.cider_metric = CIDErMetric()
        self.meteor_metric = METEORMetric()
        self.bertscore_metric = BERTScoreMetric(config)
        self.radgraph_metric = RadGraphF1Metric(config)
        self.clinical_metric = ClinicalAccuracyMetric(config)
        self.radcliq_metric = RadCliQMetric(config)
        
        self.logger = logging.getLogger("RadixpertEvaluator")
    
    def evaluate(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device = None,
        max_samples: Optional[int] = None
    ) -> Dict[str, float]:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model.eval()
        
        predictions = []
        references = []
        sample_count = 0
        
        self.logger.info("Starting model evaluation...")
        
        with torch.no_grad():
            for batch in dataloader:
                images = batch['images'].to(device)
                
                generated_reports = model.generate_report(
                    images=images,
                    max_length=512,
                    temperature=0.7,
                    num_beams=4
                )
                
                predictions.extend(generated_reports)
                references.extend(batch['texts'])
                
                sample_count += len(generated_reports)
                
                if max_samples and sample_count >= max_samples:
                    break
        
        self.logger.info(f"Evaluating {len(predictions)} generated reports...")
        
        return self.compute_metrics(predictions, references)
    
    def compute_metrics(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        metrics = {}
        
        bleu_references = [[ref] for ref in references]
        
        bleu_scores = self.bleu_metric.compute_bleu(predictions, bleu_references)
        metrics.update(bleu_scores)
        
        rouge_scores = self.rouge_metric.compute_rouge(predictions, references)
        metrics.update(rouge_scores)
        
        cider_scores = self.cider_metric.compute_cider(predictions, bleu_references)
        metrics.update(cider_scores)
        
        meteor_scores = self.meteor_metric.compute_meteor(predictions, references)
        metrics.update(meteor_scores)
        
        bertscore_scores = self.bertscore_metric.compute_bertscore(predictions, references)
        metrics.update(bertscore_scores)
        
        radgraph_scores = self.radgraph_metric.compute_radgraph_f1(predictions, references)
        metrics.update(radgraph_scores)
        
        clinical_scores = self.clinical_metric.compute_clinical_accuracy(predictions, references)
        metrics.update(clinical_scores)
        
        radcliq_scores = self.radcliq_metric.compute_radcliq(predictions, references)
        metrics.update(radcliq_scores)
        
        self.logger.info("Evaluation completed successfully")
        self.logger.info(f"Key metrics - BLEU-4: {metrics.get('bleu_4', 0):.3f}, "
                        f"CIDEr: {metrics.get('cider', 0):.3f}, "
                        f"RadCliQ-v1: {metrics.get('radcliq_v1', 0):.3f}")
        
        return metrics
    
    def evaluate_single_sample(
        self,
        prediction: str,
        reference: str
    ) -> Dict[str, float]:
        return self.compute_metrics([prediction], [reference])
    
    def print_evaluation_report(self, metrics: Dict[str, float]):
        print("\n" + "="*60)
        print("RADIXPERT EVALUATION REPORT")
        print("="*60)
        
        print("\n📊 PRIMARY METRICS:")
        print(f"  BLEU-4:       {metrics.get('bleu_4', 0):.4f}")
        print(f"  CIDEr:        {metrics.get('cider', 0):.4f}")
        print(f"  RadCliQ-v1:   {metrics.get('radcliq_v1', 0):.4f}")
        
        print("\n📝 LANGUAGE GENERATION METRICS:")
        for i in range(1, 5):
            bleu_key = f'bleu_{i}'
            if bleu_key in metrics:
                print(f"  BLEU-{i}:       {metrics[bleu_key]:.4f}")
        
        print(f"  METEOR:       {metrics.get('meteor', 0):.4f}")
        print(f"  ROUGE-L F1:   {metrics.get('rougeL_f1', 0):.4f}")
        
        print("\n🏥 CLINICAL METRICS:")
        print(f"  RadGraph F1:      {metrics.get('radgraph_f1', 0):.4f}")
        print(f"  Clinical Accuracy: {metrics.get('clinical_accuracy', 0):.4f}")
        print(f"  Anatomy Accuracy:  {metrics.get('anatomy_accuracy', 0):.4f}")
        print(f"  Pathology Accuracy: {metrics.get('pathology_accuracy', 0):.4f}")
        
        print("\n🧠 SEMANTIC METRICS:")
        print(f"  BERTScore F1:   {metrics.get('bertscore_f1', 0):.4f}")
        print(f"  SembScore:      {metrics.get('radcliq_sembscore', 0):.4f}")
        
        print("\n" + "="*60)


def create_evaluator(
    clinical_bert_model: str = "emilyalsentzer/Bio_ClinicalBERT",
    **kwargs
) -> RadixpertEvaluator:
    config = MetricConfig(
        clinical_bert_model=clinical_bert_model,
        **kwargs
    )
    return RadixpertEvaluator(config)


def evaluate_predictions(
    predictions: List[str],
    references: List[str],
    config: Optional[MetricConfig] = None
) -> Dict[str, float]:
    evaluator = RadixpertEvaluator(config)
    return evaluator.compute_metrics(predictions, references)


if __name__ == "__main__":
    print("Testing Radixpert Evaluation Metrics...")
    
    predictions = [
        "No acute cardiopulmonary abnormality. Heart size and mediastinal contours are normal.",
        "Bilateral lower lobe consolidation consistent with pneumonia. Heart size is enlarged.",
        "Clear lungs. No focal consolidation or pleural effusion. Normal cardiac silhouette."
    ]
    
    references = [
        "No acute findings. Normal heart size and mediastinal structures.",
        "Bilateral pneumonia with cardiomegaly present in both lower lobes.",
        "Lungs are clear bilaterally. No consolidation or effusion. Heart size normal."
    ]
    
    evaluator = create_evaluator()
    
    metrics = evaluator.compute_metrics(predictions, references)
    
    evaluator.print_evaluation_report(metrics)
    
    print("\n🔍 INDIVIDUAL METRIC TESTING:")
    
    bleu_scores = evaluator.bleu_metric.compute_bleu(predictions, [[ref] for ref in references])
    print(f"BLEU-4: {bleu_scores['bleu_4']:.4f}")
    
    radgraph_scores = evaluator.radgraph_metric.compute_radgraph_f1(predictions, references)
    print(f"RadGraph F1: {radgraph_scores['radgraph_f1']:.4f}")
    
    clinical_scores = evaluator.clinical_metric.compute_clinical_accuracy(predictions, references)
    print(f"Clinical Accuracy: {clinical_scores['clinical_accuracy']:.4f}")
    
    radcliq_scores = evaluator.radcliq_metric.compute_radcliq(predictions, references)
    print(f"RadCliQ-v1: {radcliq_scores['radcliq_v1']:.4f}")
    
    print("\nEvaluation metrics testing completed successfully!")
