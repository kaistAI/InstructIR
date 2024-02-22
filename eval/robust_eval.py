from collections import defaultdict 
import numpy as np 
from typing import List, Dict, Tuple
import logging
from beir.retrieval.evaluation import EvaluateRetrieval
import pytrec_eval

logger = logging.getLogger(__name__)

class CustomEvaluateRetrieval(EvaluateRetrieval):
    @staticmethod
    def robustness_evaluate(
                queries: Dict[str, Dict[str, int]],
                qrels: Dict[str, Dict[str, int]], 
                results: Dict[str, Dict[str, float]], 
                k_values: List[int],
                type='query',
                ignore_identical_ids: bool=True) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:
        
        if ignore_identical_ids:
            logging.info('For evaluation, we ignore identical query and document ids (default), please explicitly set ``ignore_identical_ids=False`` to ignore this.')
            popped = []
            for qid, rels in results.items():
                for pid in list(rels):
                    if qid == pid:
                        results[qid].pop(pid)
                        popped.append(pid)

        metrics=defaultdict(list)
        group_metric = {}
        
        for k in k_values:
            group_metric[f"NDCG@{k}"] = []
            group_metric[f"MAP@{k}"] = []
            group_metric[f"Recall@{k}"] = []
            group_metric[f"P@{k}"] = []
            
            group_metric[f"NDCG@{k}_min"] = []
            group_metric[f"MAP@{k}_min"] = []
            group_metric[f"Recall@{k}_min"] = []
            group_metric[f"P@{k}_min"] = []
        
        map_string = "map_cut." + ",".join([str(k) for k in k_values])
        ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
        recall_string = "recall." + ",".join([str(k) for k in k_values])
        precision_string = "P." + ",".join([str(k) for k in k_values])
            
        # Grouping instances based on [instruction / query]
        group_per_instruction={}
        group_per_query={}
        for qid, rels in results.items():
            query_info = queries[qid]
            instruction, query = query_info.split('[SEP]')
            instruction = instruction.strip()
            query = query.strip()

            if not group_per_instruction.get(instruction):
                group_per_instruction[instruction]={}

            if not group_per_query.get(query):
                group_per_query[query]={}
                    
            group_per_instruction[instruction][qid]=rels
            group_per_query[query][qid]=rels


        if type=='instruction':
            logger.info("Group per instruction is selected")
            target_group = group_per_instruction
        elif type =='query':
            logger.info("Group per query is selected")
            target_group = group_per_query
        else:
            raise NotImplementedError
                
        logger.info(f"Total {len(target_group)} number of group is selected out of {len(results)}")

        for group_id, group_results in target_group.items():
            ndcg = {}
            _map = {}
            recall = {}
            precision = {}
            
            for k in k_values:
                ndcg[f"NDCG@{k}"] = []
                _map[f"MAP@{k}"] = []
                recall[f"Recall@{k}"] = []
                precision[f"P@{k}"] = []
            
            evaluator = pytrec_eval.RelevanceEvaluator(qrels, {map_string, ndcg_string, recall_string, precision_string})
            group_scores = evaluator.evaluate(group_results)
            for query_id in group_scores.keys():
                for k in k_values:
                    ndcg[f"NDCG@{k}"].append(group_scores[query_id]["ndcg_cut_" + str(k)])
                    _map[f"MAP@{k}"].append(group_scores[query_id]["map_cut_" + str(k)])
                    recall[f"Recall@{k}"].append(group_scores[query_id]["recall_" + str(k)])
                    precision[f"P@{k}"].append(group_scores[query_id]["P_"+ str(k)])

            # AVG per group
            for k in k_values:
                ## Macro average for min score (Robustness score) => avg for avg
                ## Micro average for normal score
                group_metric[f"NDCG@{k}_min"].append(min(ndcg[f"NDCG@{k}"]))
                group_metric[f"NDCG@{k}"].extend(ndcg[f"NDCG@{k}"])
                group_metric[f"MAP@{k}_min"].append(min(_map[f"MAP@{k}"]))
                group_metric[f"MAP@{k}"].extend(_map[f"MAP@{k}"])
                group_metric[f"Recall@{k}_min"].append(min(recall[f"Recall@{k}"]))
                group_metric[f"Recall@{k}"].extend(recall[f"Recall@{k}"])
                group_metric[f"P@{k}_min"].append(min(precision[f"P@{k}"]))
                group_metric[f"P@{k}"].extend(precision[f"P@{k}"])
        
        
        '''
        calculate per group
        '''
        for name, v in group_metric.items():
            metrics['cluster_{}_avg'.format(name)]= sum(v) / len(v)

        metrics = dict(sorted(metrics.items()))

        logging.info("\n")
        logging.info("group_per {}".format(type))
        # logging.info(metrics)
        for k in metrics.keys():
            logging.info("{}: {:.4f}".format(k, metrics[k]*100))

        return metrics
    
    