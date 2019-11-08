from fastestimator.trace import Trace
import numpy as np
import tensorflow as tf
from collections import defaultdict
import pdb

#'selected_indices_padded','valid_outputs', 'abs_loc','image_id','pred_cls',"cls_gt", "x1_gt", "y1_gt", "x2_gt", "y2_gt"

class Mean_avg_precision(Trace):
    def __init__(self, categories, maxDets, selected_indices_key, valid_output_key, image_id_key, pred_cls_key, abs_loc_key, cls_gt_key, x1_gt_key, y1_gt_key, x2_gt_key, y2_gt_key, mode="eval", output_name="map"):
        #############

        super().__init__(outputs=output_name, mode=mode)
        self.selected_indices_key = selected_indices_key
        self.valid_output_key = valid_output_key
        self.image_id_key = image_id_key
        self.pred_cls_key = pred_cls_key
        self.abs_loc_key = abs_loc_key
        self.cls_gt_key = cls_gt_key
        self.x1_gt_key = x1_gt_key
        self.y1_gt_key = y1_gt_key
        self.x2_gt_key = x2_gt_key
        self.y2_gt_key = y2_gt_key

        ##############
        self.iouThrs = np.linspace(.5, 0.95, np.round((0.95 - .5) / .05) + 1, endpoint=True)
        self.recThrs = np.linspace(.0, 1.00, np.round((1.00 - .0) / .01) + 1, endpoint=True)
        self.categories = np.array(categories)
        self.categories  = [n for n,cat in enumerate(self.categories)]
        self.maxDets = maxDets
        self.image_ids = []

    def on_epoch_begin(self, state):
        self.evalImgs = {}
        self.eval = {}


    def on_batch_begin(self, state):
        self.gt= defaultdict(list)       # gt for evaluation
        self.dt = defaultdict(list)       # dt for evaluation
        self.batch_image_ids = []        # img_ids per batch
        self.ious = defaultdict(list)

    def on_batch_end(self, state):

#'selected_indices_padded','valid_outputs', 'abs_loc','image_id','pred_cls',"cls_gt", "x1_gt", "y1_gt", "x2_gt", "y2_gt"

        selected_indices = state["batch"][self.selected_indices_key]
        selected_indices_padded_np = selected_indices.numpy()
        valid_outputs = state["batch"][self.valid_output_key]
        valid_outputs_np = valid_outputs.numpy()
        abs_loc = state["batch"][self.abs_loc_key]
        loc_pred_abs_np = abs_loc.numpy()
        self.batch_image_ids = state["batch"][self.image_id_key]
        self.batch_image_ids = self.batch_image_ids.numpy()
        self.image_ids.append(self.batch_image_ids)

        pred_cls = state["batch"][self.pred_cls_key]

        cls_best_score = tf.reduce_max(pred_cls, axis=-1)
        cls_best_class = tf.argmax(pred_cls, axis=-1)
        cls_best_score_np = cls_best_score.numpy()
        cls_best_class_np = cls_best_class.numpy()

        cls_gt = state["batch"][self.cls_gt_key]
        cls_gt_np = cls_gt.numpy()
        x1_gt = state["batch"][self.x1_gt_key]
        x1_gt_np = x1_gt.numpy()
        x2_gt = state["batch"][self.x2_gt_key]
        x2_gt_np = x2_gt.numpy()
        y1_gt = state["batch"][self.y1_gt_key]
        y1_gt_np = y1_gt.numpy()
        y2_gt = state["batch"][self.y2_gt_key]
        y2_gt_np = y2_gt.numpy()


        predicted_bb = []
        for idx, (sip_elem , vo_elem, loc_pred_elem, score_pred_elem, cls_pred_elem, image_id_elem) in enumerate(
            zip(selected_indices_padded_np, valid_outputs_np, loc_pred_abs_np,
                cls_best_score_np, cls_best_class_np, self.batch_image_ids)):
            sel_indices = sip_elem[:vo_elem]
            for ind in sel_indices:
#                 pdb.set_trace()
                x1,y1,x2,y2 = loc_pred_elem[ind]
                cls = cls_pred_elem[ind]
                score = score_pred_elem[ind]
                tmp_dict = {'idx':image_id_elem, 'x1':x1,'y1':y1,'x2':x2, 'y2':y2,'cls': cls, 'score':score}
                predicted_bb.append(tmp_dict)

        ground_truth_bb  = []
        for idx, (cls_gt_elem, x1_gt_elem, y1_gt_elem, x2_gt_elem, y2_gt_elem, image_id_elem) in enumerate(
            zip(cls_gt_np, x1_gt_np, y1_gt_np, x2_gt_np, y2_gt_np, self.batch_image_ids)):
            for cls_gt_inst, x1_gt_inst, y1_gt_inst, x2_gt_inst, y2_gt_inst in zip(cls_gt_elem,x1_gt_elem, 
                                                                                   y1_gt_elem, x2_gt_elem, y2_gt_elem):
                temp_dict = {'idx': image_id_elem, 'x1':x1_gt_inst,'y1':y1_gt_inst,'x2':x2_gt_inst,
                             'y2':y2_gt_inst,'cls':cls_gt_inst }
                ground_truth_bb.append(temp_dict)

        pdb.set_trace()
        ground_truth_bb = ground_truth_bb[:10]


        for dict_elem in ground_truth_bb:
            self.gt[dict_elem['idx'], dict_elem['cls']].append(dict_elem)
        for dict_elem in predicted_bb:
            self.dt[dict_elem['idx'], dict_elem['cls']].append(dict_elem)

        num_category = len(self.categories)
        self.ious = {(img_id,cat_id): compute_iou(self.dt[image_id, cat_id], self.gt[image_id,cat_id]) for img_id in
                self.batch_image_ids for cat_id in self.categories }
        for catid in self.categories:
            for img_id in self.batch_image_ids:
                self.evalImgs[(cat_id,img_d)] = self.evaluate_img(img_id,cat_id)


    def on_epoch_end(self, state):

        self.accumulate()
        self.summarize()

    def accumulate(self):

        key_list = self.evalImgs
        key_list = sorted(key_list)
        eval_list = [self.evalImgs[key] for key in key_list]
        pdb.set_trace()

        T = len(self.iouThrs)
        R  = len(self.recThrs)
        K = len(self.categories)
        cat_list = self.categories

        I = len(self.image_ids)

        precison = -np.ones((T,R,K))
        recall = -np.ones((T,K))
        scores = -np.ones((T,R,K))


        for k in cat_list:
            Nk = k*I
            E = [eval_list[Nk+img_idx]  for img_idx in range(I)]
            E = [e for e in E if not e is None]
            if len(E) == 0:
                continue
            dtScores = np.concatenate([e['dtScores'] for e in E])

            inds = np.argsort(-dtScores, kind='mergesort')
            dtScoresSorted = dtScores[inds]

            dtm  = np.concatenate([e['dtMatches'] for e in E], axis=1)[:,inds]
            dtIg = np.concatenate([e['dtIgnore']  for e in E], axis=1)[:,inds]

            gtIg = np.concatenate([e['gtIgnore'] for e in E])
            npig = np.count_nonzero(gtIg==0 )
            if npig == 0:
                continue

            tps = np.logical_and(               dtm,  np.logical_not(dtIg) )
            fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg) )

            tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
            fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)

            for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                tp = np.array(tp)
                fp = np.array(fp)
                nd = len(tp)
                rc = tp / npig
                pr = tp / (fp+tp+np.spacing(1))
                q  = np.zeros((R,))
                ss = np.zeros((R,))

                if nd:
                    recall[t,k] = rc[-1]
                else:
                    recall[t,k] = 0
                pr = pr.tolist(); q = q.tolist()

                for i in range(nd-1, 0, -1):
                    if pr[i] > pr[i-1]:
                        pr[i-1] = pr[i]
                inds = np.searchsorted(rc, self.recThrs, side='left')
                try:
                    for ri, pi in enumerate(inds):
                        q[ri] = pr[pi]
                        ss[ri] = dtScoresSorted[pi]
                except:
                    pass
                precision[t,:,k] = np.array(q)
                scores[t,:,k] = np.array(ss)
        self.eval = {
            'counts': [T, R, K],
            'precision': precision,
            'recall':   recall,
            'scores': scores,
        }


    def summarize(self, iou=0.5):
        s = self.eval['precision']
        if iouThr is not None:
            t = np.where(iou == self.iouThrs)[0]
            s = s[t]
        if len(s[s>-1])==0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s>-1])
        print('Mean average precision', mean_s)




    def evaluate_img(self, img_id, cat_id):
            T = self.iouThrs 
            maxDet = self.maxDets
            dt = self.dt[img_id,cat_id]
            gt = self.gt[img_id, cat_id]
            num_dt = len(dt)
            num_gt = len(gt)
            if num_gt==0 or num_dt==0:
                return None

            dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
            dt = [dt[i] for i in dtind]

            iou_mat = self.ious[img_id, cat_id]

            T = len(self.iouThrs)

            dtm = np.zeros((T, num_gt))
            gtm = np.zeros((T, num_dt))
            gtIg =[ 0 for g in gt]
            dtIg = np.zeros((T,D))

            if len(iou_mat)!=0:
                for thres_idx, thres_elem in enumerate(T):
                    for dt_idx, dt_elem in enumerate(dt):
                        m = -1
                        iou =  min([thres_elem,1-1e-10])
                        for gt_idx, gt_elem in enumerate(gt):
                            if gtm[thres_idx, gt_idx]>0:
                                continue
                            if iou_mat[dt_idx, gt_idx] > iou:
                                iou = iou_mat[dt_idx, gt_idx]
                                m = gt_idx

                        if m != -1:
                            dtm[thres_idx, dt_idx] = gt[m]['idx']
                            gtm[thres_idx, gt_idx] = 1

            dtIg = (dtm == 0)

            return {
                    'image_id':     imgId,
                    'category_id':  catId,
                    'gtIds':        [g['id'] for g in gt],
                    'dtMatches':    dtm,
                    'gtMatches':    gtm,
                    'dtScores':     [d['score'] for d in dt],
                    'gtIgnore':     gtIg,
                    'dtIgnore':     dtIg,
                }


    def compute_iou(dt, gt):
        num_dt = len(dt)
        num_gt = len(gt)

        if num_gt==0 or num_dt==0:   #  or?  and?
            return []

        boxes_a = np.zeros(shape=(0,4), dtype=float)
        boxes_b = np.zeros(shape(0,4), dtype=float)

        inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in inds]
        if len(dt) > self.maxDets:
            dt=dt[0:self.maxDets]

        for dt_elem in dt:
            x1,y1,x2,y2= dt['x1'], dt['y1'], dt['x2'], dt['y2']
            boxes_a = np.append(boxes_a, np.array([[x1,y1,x2,y2]]))

        for gt_elem in gt:
            x1,y1,x2,y2 =gt['x1'], gt['y1'], gt['x1'], gt['y2']
            boxes_b = np.append(boxes_b, np.array([[x1,y1,x2,y2]]))

        iou_dt_gt  = get_iou(boxes_a, boxes_b)
        return iou_dt_gt

    def get_iou(self, boxes1, boxes2):
        """Computes the value of intersection over union (IoU) of two array of boxes.

        Args:
            box1 (array): first boxes in N x 4
            box2 (array): second box in M x 4

        Returns:
            float: IoU value in N x M
        """
        x11, y11, x12, y12 = np.split(boxes1, 4, axis=1)
        x21, y21, x22, y22 = np.split(boxes2, 4, axis=1)
        w1 = x12 - x11
        h1 = y12 - y11
        w2 = x22 - x21
        h2 = y22 - y21
        xmin = np.maximum(x11, np.transpose(x21))
        ymin = np.maximum(y11, np.transpose(y21))
        xmax = np.minimum(x12, np.transpose(x22))
        ymax = np.minimum(y12, np.transpose(y22))
        inter_area = np.maximum((xmax - xmin + 1), 0) * np.maximum((ymax - ymin + 1), 0)
        area1 = (w1 + 1) * (h1 + 1)
        area2 = (w2 + 1) * (h2 + 1)
        iou = inter_area / (area1 + area2.T - inter_area)
        return iou
