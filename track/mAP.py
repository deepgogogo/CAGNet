from mmaction.core.evaluation.ava_evaluation import object_detection_evaluation as det_eval
from mmaction.core.evaluation.ava_evaluation import standard_fields
from mmaction.core.evaluation.ava_evaluation.metrics import compute_average_precision
import json
import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d
from itertools import permutations, combinations

from mmaction.core.evaluation.ava_evaluation import metrics, per_image_evaluation, standard_fields
from mmaction.core.evaluation.ava_evaluation.object_detection_evaluation import ObjectDetectionEvaluation,DetectionEvaluator,ObjectDetectionEvaluator

import matplotlib.pyplot as plt
import numpy as np
import argparse


class PREvaluator(DetectionEvaluator):
    """A class to evaluate detections."""

    def __init__(self,
                    categories,
                    matching_iou_threshold=0.5,
                    evaluate_corlocs=False,
                    metric_prefix=None,
                    use_weighted_mean_ap=False,
                    evaluate_masks=False,
                    save_PRcurve=True):
        """Constructor.

        Args:
            categories: A list of dicts, each of which has the following keys -
                'id': (required) an integer id uniquely identifying this
                    category.
                'name': (required) string representing category name e.g.,
                    'cat', 'dog'.
            matching_iou_threshold: IOU threshold to use for matching
                groundtruth boxes to detection boxes.
            evaluate_corlocs: (optional) boolean which determines if corloc
                scores are to be returned or not.
            metric_prefix: (optional) string prefix for metric name; if None,
                no prefix is used.
            use_weighted_mean_ap: (optional) boolean which determines if the
                mean average precision is computed directly from the scores and
                tp_fp_labels of all classes.
            evaluate_masks: If False, evaluation will be performed based on
                boxes. If True, mask evaluation will be performed instead.

        Raises:
            ValueError: If the category ids are not 1-indexed.
        """
        super(PREvaluator, self).__init__(categories)
        self._num_classes = max([cat['id'] for cat in categories])
        if min(cat['id'] for cat in categories) < 1:
            raise ValueError('Classes should be 1-indexed.')
        self._matching_iou_threshold = matching_iou_threshold
        self._use_weighted_mean_ap = use_weighted_mean_ap
        self._label_id_offset = 1
        self._evaluate_masks = evaluate_masks
        self._evaluation = ObjectDetectionEvaluation(
            num_groundtruth_classes=self._num_classes,
            matching_iou_threshold=self._matching_iou_threshold,
            use_weighted_mean_ap=self._use_weighted_mean_ap,
            label_id_offset=self._label_id_offset,
        )
        self._image_ids = set([])
        self._evaluate_corlocs = evaluate_corlocs
        self._metric_prefix = (metric_prefix + '_') if metric_prefix else ''
        self._save_PRcurve=save_PRcurve

    def add_single_ground_truth_image_info(self, image_id, groundtruth_dict):
        """Adds groundtruth for a single image to be used for evaluation.

        Args:
            image_id: A unique string/integer identifier for the image.
            groundtruth_dict: A dictionary containing -
                standard_fields.InputDataFields.groundtruth_boxes: float32
                    numpy array of shape [num_boxes, 4] containing `num_boxes`
                    groundtruth boxes of the format [ymin, xmin, ymax, xmax] in
                    absolute image coordinates.
                standard_fields.InputDataFields.groundtruth_classes: integer
                    numpy array of shape [num_boxes] containing 1-indexed
                    groundtruth classes for the boxes.
                standard_fields.InputDataFields.groundtruth_instance_masks:
                    Optional numpy array of shape [num_boxes, height, width]
                    with values in {0, 1}.

        Raises:
            ValueError: On adding groundtruth for an image more than once. Will
                also raise error if instance masks are not in groundtruth
                dictionary.
        """
        # if image_id in self._image_ids:
        #     raise ValueError(
        #         'Image with id {} already added.'.format(image_id))
        # distributed sampler may add extra samples causing multi image id issue
        if image_id in self._image_ids:
            return

        groundtruth_classes = (
            groundtruth_dict[
                standard_fields.InputDataFields.groundtruth_classes] -
            self._label_id_offset)

        groundtruth_masks = None
        if self._evaluate_masks:
            if (standard_fields.InputDataFields.groundtruth_instance_masks
                    not in groundtruth_dict):
                raise ValueError(
                    'Instance masks not in groundtruth dictionary.')
            groundtruth_masks = groundtruth_dict[
                standard_fields.InputDataFields.groundtruth_instance_masks]
        self._evaluation.add_single_ground_truth_image_info(
            image_key=image_id,
            groundtruth_boxes=groundtruth_dict[
                standard_fields.InputDataFields.groundtruth_boxes],
            groundtruth_class_labels=groundtruth_classes,
            groundtruth_masks=groundtruth_masks,
        )
        self._image_ids.update([image_id])

    def add_single_detected_image_info(self, image_id, detections_dict):
        """Adds detections for a single image to be used for evaluation.

        Args:
            image_id: A unique string/integer identifier for the image.
            detections_dict: A dictionary containing -
                standard_fields.DetectionResultFields.detection_boxes: float32
                    numpy array of shape [num_boxes, 4] containing `num_boxes`
                    detection boxes of the format [ymin, xmin, ymax, xmax] in
                    absolute image coordinates.
                standard_fields.DetectionResultFields.detection_scores: float32
                    numpy array of shape [num_boxes] containing detection
                    scores for the boxes.
                standard_fields.DetectionResultFields.detection_classes:
                    integer numpy array of shape [num_boxes] containing
                    1-indexed detection classes for the boxes.
                standard_fields.DetectionResultFields.detection_masks: uint8
                    numpy array of shape [num_boxes, height, width] containing
                    `num_boxes` masks of values ranging between 0 and 1.

        Raises:
            ValueError: If detection masks are not in detections dictionary.
        """
        detection_classes = (
            detections_dict[
                standard_fields.DetectionResultFields.detection_classes] -
            self._label_id_offset)
        detection_masks = None
        if self._evaluate_masks:
            if (standard_fields.DetectionResultFields.detection_masks
                    not in detections_dict):
                raise ValueError(
                    'Detection masks not in detections dictionary.')
            detection_masks = detections_dict[
                standard_fields.DetectionResultFields.detection_masks]
        self._evaluation.add_single_detected_image_info(
            image_key=image_id,
            detected_boxes=detections_dict[
                standard_fields.DetectionResultFields.detection_boxes],
            detected_scores=detections_dict[
                standard_fields.DetectionResultFields.detection_scores],
            detected_class_labels=detection_classes,
            detected_masks=detection_masks,
        )

    @staticmethod
    def create_category_index(categories):
        """Creates dictionary of COCO compatible categories keyed by category
        id.

        Args:
            categories: a list of dicts, each of which has the following keys:
                'id': (required) an integer id uniquely identifying this
                    category.
                'name': (required) string representing category name
                    e.g., 'cat', 'dog', 'pizza'.

        Returns:
            category_index: a dict containing the same entries as categories,
                but keyed by the 'id' field of each category.
        """
        category_index = {}
        for cat in categories:
            category_index[cat['id']] = cat
        return category_index

    def evaluate(self):
        """Compute evaluation result.

        Returns:
            A dictionary of metrics with the following fields -

            1. summary_metrics:
                'Precision/mAP@<matching_iou_threshold>IOU': mean average
                precision at the specified IOU threshold

            2. per_category_ap: category specific results with keys of the form
                'PerformanceByCategory/mAP@<matching_iou_threshold>IOU/category'
        """
        (per_class_ap, mean_ap, precisions, recalls, per_class_corloc,
            mean_corloc) = self._evaluation.evaluate()

        # precision_list=[p.tolist() for p in precisions]
        # recall_list=[r.tolist() for r in recalls]

        precision_list=precisions
        recall_list=recalls

        # print(precisions)
        # print(recalls)
        # print(len(precisions))
        # print(len(recalls))
        # for i in range(len(precisions)):
        #     print(precisions[i].shape)
        #     for d in range(precisions[i].shape[0]):
        #         print(precisions[i][d])
        #         print(recalls[i][d])
        #         # input('pause')

        # for i in range(len(recalls)):
        #     print(recalls[i].shape)


        # save PR curve
        if self._save_PRcurve:
            fig, ax = plt.subplots(1, 1, figsize=(5, 5)) 
            for i in range(len(self._categories)):
                p=precisions[i].tolist()
                r=recalls[i].tolist()
                ax.plot(r, p, linewidth=0.5, color='grey')  # plot(recall, precision) 
                # ax.scatter(r, p, color='grey',linewidths=0.1)  # plot(recall, precision) 
                # ax.plot(px, py.mean(1), linewidth=2, color='blue', label='all classes %.3f mAP@0.5' % ap[:, 0].mean()) 
            ax.set_xlabel('Recall') 
            ax.set_ylabel('Precision') 
            ax.set_xlim(0, 1) 
            ax.set_ylim(0, 1) 
            plt.legend() 
            fig.tight_layout() 
            fig.savefig('prcurve', dpi=200) 

        

        metric = f'mAP@{self._matching_iou_threshold}IOU'
        pascal_metrics = {self._metric_prefix + metric: mean_ap}
        if self._evaluate_corlocs:
            pascal_metrics[self._metric_prefix +
                            'Precision/meanCorLoc@{}IOU'.format(
                                self._matching_iou_threshold)] = mean_corloc
        category_index = self.create_category_index(self._categories)
        for idx in range(per_class_ap.size):
            if idx + self._label_id_offset in category_index:
                display_name = (
                    self._metric_prefix +
                    'PerformanceByCategory/AP@{}IOU/{}'.format(
                        self._matching_iou_threshold,
                        category_index[idx + self._label_id_offset]['name'],
                    ))
                pascal_metrics[display_name] = per_class_ap[idx]

                # Optionally add CorLoc metrics.classes
                if self._evaluate_corlocs:
                    display_name = (
                        self._metric_prefix +
                        'PerformanceByCategory/CorLoc@{}IOU/{}'.format(
                            self._matching_iou_threshold,
                            category_index[idx +
                                            self._label_id_offset]['name'],
                        ))
                    pascal_metrics[display_name] = per_class_corloc[idx]

        return pascal_metrics,precision_list,recall_list

    def clear(self):
        """Clears the state to prepare for a fresh evaluation."""
        self._evaluation = ObjectDetectionEvaluation(
            num_groundtruth_classes=self._num_classes,
            matching_iou_threshold=self._matching_iou_threshold,
            use_weighted_mean_ap=self._use_weighted_mean_ap,
            label_id_offset=self._label_id_offset,
        )
        self._image_ids.clear()


def xyxy2yxyx(bbox):
    assert len(bbox.shape)==2 and isinstance(bbox,np.ndarray)
    nbbox=np.zeros_like(bbox,dtype=float)
    nbbox[:,0]=bbox[:,1]
    nbbox[:,1]=bbox[:,0]
    nbbox[:,2]=bbox[:,3]
    nbbox[:,3]=bbox[:,2]
    return nbbox

def retrieve_grd_inter(inter_grd,inter_bboxes):
    """ retrieve inter up triangle N*(N-1)->N*(N-1)/2,"""
    assert len(inter_grd)==len(inter_bboxes)
    inter_grd=np.array(inter_grd,dtype=int)
    inter_bboxes=np.array(inter_bboxes,dtype=float)

    N=int(np.math.sqrt(inter_grd.shape[0])+1)
    triul_idx= np.array(list(permutations(range(N),2)))
    triu_idx=np.array(list(combinations(range(N),2)))

    n_inter_grd=np.zeros((N,N),dtype=np.int)
    n_inter_grd[triul_idx[:,0],triul_idx[:,1]]=inter_grd
    n_inter_grd=n_inter_grd[triu_idx[:,0],triu_idx[:,1]]
    
    n_inter_bboxes=np.zeros((N,N,4))
    n_inter_bboxes[triul_idx[:,0],triul_idx[:,1],:]=inter_bboxes
    n_inter_bboxes=n_inter_bboxes[triu_idx[:,0],triu_idx[:,1],:]

    return n_inter_grd[n_inter_grd==1],n_inter_bboxes[n_inter_grd==1,:]

def retrieve_prd_inter(inter_prd,inter_bboxes,inter_confidence):
    """ retrieve inter up triangle N*(N-1)->N*(N-1)/2"""
    assert len(inter_prd)==len(inter_bboxes)
    inter_prd=np.array(inter_prd,dtype=int)
    inter_confidence=np.array(inter_confidence,dtype=float)
    inter_bboxes=np.array(inter_bboxes,dtype=float)[:,0:4]

    N=int(np.math.sqrt(inter_prd.shape[0]))+1
    triul_idx=np.array(list(permutations(range(N),2)))
    triu_idx=np.array(list(combinations(range(N),2)))

    n_inter_prd=np.zeros((N,N))
    n_inter_prd[triul_idx[:,0],triul_idx[:,1]]=inter_prd
    n_inter_prd=n_inter_prd[triu_idx[:,0],triu_idx[:,1]]

    n_inter_confidence=np.zeros((N,N))
    n_inter_confidence[triul_idx[:,0],triul_idx[:,1]]=inter_confidence
    n_inter_confidence=n_inter_confidence[triu_idx[:,0],triu_idx[:,1]]

    n_inter_bboxes=np.zeros((N,N,4))
    n_inter_bboxes[triul_idx[:,0],triul_idx[:,1],:]=inter_bboxes
    n_inter_bboxes=n_inter_bboxes[triu_idx[:,0],triu_idx[:,1]]
    
    return n_inter_prd[n_inter_prd==1],n_inter_bboxes[n_inter_prd==1,:],n_inter_confidence[n_inter_prd==1]

def interpolate_precision_recall(precision,recall):
    """precision,recall are list of ndarray,length is number of class"""
    max_len=max(precision,key=lambda d:d.shape[0]).shape[0]
    new_precision,new_recall=[],[]
    for p,r in zip(precision,recall):
        f=interp1d(r,p)
        n_r=np.linspace(r.min(),r.max(),max_len)
        n_p=f(n_r)
        new_precision.append(n_p)
        new_recall.append(n_r)
    return new_precision,new_recall


def precision_recall(y_truth, y_pred, show=False):
    
    bit_classes=['bend','box','handshake','hifive','hug','kick','pat','push','others']
    bit_categories = []
    for i in range(len(bit_classes)):
        bit_categories.append({'id': i+1, 'name': bit_classes[i]})
        
    act_evaluator=PREvaluator(bit_categories,save_PRcurve=False)
    inter_evaluator=PREvaluator([{'id':1,'name':'inter'}],save_PRcurve=False)
    
    for grd in y_truth:
        img_key=grd['image']['file_name']
        act_evaluator.add_single_ground_truth_image_info(
            img_key,{
                standard_fields.InputDataFields.groundtruth_boxes:
                xyxy2yxyx(np.array(grd['bboxes'],dtype=float)),
                standard_fields.InputDataFields.groundtruth_classes:
                np.array(grd['action_grd'], dtype=int)+1                
            }
        )
    
        inter_grd,inter_bboxes=retrieve_grd_inter(grd['inter_grd'],grd['inter_bboxes'])
        inter_evaluator.add_single_ground_truth_image_info(
            img_key,{
                standard_fields.InputDataFields.groundtruth_boxes:
                xyxy2yxyx(inter_bboxes),
                standard_fields.InputDataFields.groundtruth_classes:
                np.array(inter_grd)                
            }
        )

    for det in y_pred:
        img_key=det['image']['file_name']
        if len(det['bboxes'])==1 or len(det['bboxes'])==0:
            continue
        act_evaluator.add_single_detected_image_info(
            img_key,{
                standard_fields.DetectionResultFields.detection_boxes:
                    xyxy2yxyx(np.array(det['bboxes'], dtype=float)[:,0:4]),
                standard_fields.DetectionResultFields.detection_classes:
                    np.array(det['action_pred'], dtype=int)+1,
                standard_fields.DetectionResultFields.detection_scores:
                    np.array(det['action_confidence'], dtype=float)
            }
        )
        inter_prd,inter_bboxes,inter_confidence=retrieve_prd_inter(det['inter_pred'],det['inter_bboxes'],det['inter_confidence'])
        inter_evaluator.add_single_detected_image_info(
            img_key,{
                standard_fields.DetectionResultFields.detection_boxes:
                    xyxy2yxyx(inter_bboxes),
                standard_fields.DetectionResultFields.detection_classes:
                    inter_prd,
                standard_fields.DetectionResultFields.detection_scores:
                    inter_confidence
            }
        )
    
    act_metrics,act_pricision,act_recall=act_evaluator.evaluate()
    inter_metrics,inter_pricision,inter_recall=inter_evaluator.evaluate()
    act_pricision.extend(inter_pricision)
    act_recall.extend(inter_recall)
    p_arr,r_arr=interpolate_precision_recall(act_pricision,act_recall)# interpolate the target 
    p_arr=np.array(p_arr).mean(axis=0)
    r_arr=np.array(r_arr).mean(axis=0) 
    
    mAP=compute_average_precision(p_arr,r_arr)
    if show:
        print(f'total mAP={mAP}')
        for display_name in act_metrics:
            print(f'{display_name}=\t{act_metrics[display_name]}')
        for display_name in inter_metrics:
            print(f'{display_name}=\t{inter_metrics[display_name]}')
    
    return {'mAP':mAP}


if __name__=='__main__':
    
    parser=argparse.ArgumentParser()
    parser.add_argument('--truth_file',
                        type=str)
    parser.add_argument('--pred_file',
                        type=str)
    args=parser.parse_args()
    
    truth_file='/home/whz/CAGNet/scratch/mAp/BIT_grdbox_inter.json'
    # pred_file='/home/whz/CAGNet/scratch/mAp/bit_groundtruth_inter.json'
    pred_file='/home/whz/CAGNet/scratch/mAp/bit_fcos_trained_pred_inter.json'
    with open(truth_file) as f:
        y_truth=json.load(f)
    with open(pred_file) as f:
        y_pred=json.load(f)
    precision_recall(y_truth,y_pred)
    
    pass