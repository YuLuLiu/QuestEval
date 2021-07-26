from typing import List, Tuple, Dict, Callable
import os
import json
import numpy as np
import logging
from datasets import load_metric
import spacy
import torch
from questeval import DIR, __version__
from questeval.questeval_metric import QuestEval
from questeval.utils import (
    API_T2T,
    sentencize,
    calculate_f1_squad,
    calculate_BERTScore,
    extract_table_answers,
    text2hash
)

HF_ORGANIZATION = "ThomasNLG"

class Mask_QuestEval(QuestEval):
    def __init__(
            self,
            task: str = "text2text",
            language: str = "en",
            answer_types: Tuple = ('NER', 'NOUN'),
            list_scores: Tuple = ('answerability', 'bertscore', 'f1'),
            src_preproc_pipe=None,
            do_weighter: bool = False,
            do_consistency: bool = False,
            qg_batch_size: int = 36,
            clf_batch_size: int = 48,
            limit_sent: int = 5,
            reduction_multi_refs: Callable = max,
            no_cuda: bool = False,
            use_cache: bool = True
    ) -> None:
        super().__init__(
            task,
            language,
            answer_types,
            list_scores,
            src_preproc_pipe,
            do_weighter,
            do_consistency,
            qg_batch_size,
            clf_batch_size,
            limit_sent,
            reduction_multi_refs,
            no_cuda,
            use_cache)
        self.sep = "</s>"

    def _load_all_models(self) -> Dict:
        # Textual hypothesis
        models = {"hyp": {}}
        if self.language == 'en':
            models['hyp']['QA'] = f'questeval/t5_checkpoint-70000'
            models['hyp']['QG'] = f'{HF_ORGANIZATION}/t5-qg_squad1-en'
        else:
            raise("Multilingual evaluation not handled yet.")

        # (if) multimodal sources
        if self.task == "data2text":
            models['src'] = dict()
            models['src']['QA'] = f'{HF_ORGANIZATION}/t5-qa_webnlg_synth-en'
            models['src']['QG'] = f'{HF_ORGANIZATION}/t5-qg_webnlg_synth-en'

        # Loading all the different models
        for modality in models.keys():
            for task in models[modality].keys():
                if not type(models[modality][task]) == str:
                    continue
                models[modality][task]= self.get_model(model_name=models[modality][task])

        # Loading the weighter
        models['Weighter'] = None
        if self.do_weighter:
            models['Weighter'] = self.get_model(model_name=f'{HF_ORGANIZATION}/t5-weighter_cnndm-en')

        # Linking already loaded models for the other keys
        for k in ["src", "ref"]:
            if models.get(k) == None:
                models[k] = dict()
                models[k]['QA'] = models['hyp']['QA']
                models[k]['QG'] = models['hyp']['QG']

        return models

    def _generate_masked_question(self, source: str, chunk: str):
        sentences = self.spacy_pipeline(source).sents
        for sent in sentences:
            sent_text = sent.text
            if chunk in sent_text:
                question = sent_text.replace(chunk, '<mask>')
                break
        return question

    def _predict_questions(
        self,
        to_do_exs: List[tuple],
        type_logs: str
    ) -> List[str]:

        question_texts = []
        for asw, context in to_do_exs:
            question_texts.append(self._generate_masked_question(source=context, chunk=asw))
        return question_texts

    def get_model(self, model_name: str,):
        keep_score_idx = None

        if 't5' in model_name.lower():
            if "checkpoint" in model_name.lower():
                keep_score_idx = 32102 #check t5 special token
            if "qa" in model_name.lower():
                # 73 is the index for the token unanswerable in T5 vocabulary
                keep_score_idx = 73
            if 'weighter' in model_name.lower():
                # 1176 is the index for the token true in T5 vocabulary
                keep_score_idx = 1176
            if model_name == f"{HF_ORGANIZATION}/t5-qg_squad1-en":
                # the default models were trained with this prefix 'sv1' and 'nqa' prefix on the two datasets
                self.qg_prefix = 'sv1'

            # batch size
            model_batch_size = self.qg_batch_size if "qg" in model_name.lower() else self.clf_batch_size

            model = API_T2T(
                pretrained_model_name_or_path=model_name,
                keep_score_idx=keep_score_idx,
                max_source_length=512,
                model_batch_size=model_batch_size,
                device=self.device
            )

        else:
            raise NotImplementedError(f'Model Name Not Handled: the model name should contain t5 ({model_name}).')

        return model
