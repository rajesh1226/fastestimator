from fastestimator.trace import Trace
import numpy as np
import tensorflow as tf
from collections import defaultdict
from fastestimator.architecture.retinanet import get_iou
from fastestimator.architecture.retinanet import get_fpn_anchor_box
from pycocotools import mask as maskUtils
import pdb

#'selected_indices_padded','valid_outputs', 'abs_loc','image_id','pred_cls',"cls_gt", "x1_gt", "y1_gt", "w_gt", "h_gt"

class Mean_avg_precision(Trace):
    def __init__(self, categories, maxDets, selected_indices_key, valid_output_key, image_id_key, pred_cls_key, abs_loc_key, 
    cls_gt_key, x1_gt_key, y1_gt_key, w_gt_key, h_gt_key,
    num_gt_bb_key, cls_gt_bb_key, x1_gt_bb_key, y1_gt_bb_key, w_gt_bb_key,h_gt_bb_key,
     mode="eval", output_name="map"):

        super().__init__(outputs=output_name, mode=mode)
        self.selected_indices_key = selected_indices_key
        self.valid_output_key = valid_output_key
        self.image_id_key = image_id_key
        self.pred_cls_key = pred_cls_key
        self.abs_loc_key = abs_loc_key
        self.cls_gt_key = cls_gt_key
        self.x1_gt_key = x1_gt_key
        self.y1_gt_key = y1_gt_key
        self.w_gt_key = w_gt_key
        self.h_gt_key = h_gt_key

        self.num_gt_bb_key = num_gt_bb_key
        self.cls_gt_bb_key = cls_gt_bb_key
        self.x1_gt_bb_key = x1_gt_bb_key
        self.y1_gt_bb_key = y1_gt_bb_key
        self.w_gt_bb_key = w_gt_bb_key
        self.h_gt_bb_key = h_gt_bb_key

        self.iouThrs = np.linspace(.5, 0.95, np.round((0.95 - .5) / .05) + 1, endpoint=True)
        self.recThrs = np.linspace(.0, 1.00, np.round((1.00 - .0) / .01) + 1, endpoint=True)
        self.categories = np.array(categories)
        self.categories  = [n for n,cat in enumerate(self.categories)]
        self.maxDets = maxDets
        self.image_ids = []
        self.anch_box = get_fpn_anchor_box(input_shape=(512, 512, 3))[0]

    def on_epoch_begin(self, state):
        self.image_ids = []   # append all the image ids coming from each iteration
        self.evalImgs = {}
        self.eval = {}


    def on_batch_begin(self, state):
        self.gt= defaultdict(list)       # gt for evaluation
        self.dt = defaultdict(list)       # dt for evaluation
        self.batch_image_ids = []        # img_ids per batch
        self.ious = defaultdict(list)

    def on_batch_end(self, state):

        selected_indices = state["batch"][self.selected_indices_key]
        selected_indices_padded_np = selected_indices.numpy()
        valid_outputs = state["batch"][self.valid_output_key]
        valid_outputs_np = valid_outputs.numpy()
        abs_loc = state["batch"][self.abs_loc_key]
        loc_pred_abs_np = abs_loc.numpy()
        self.batch_image_ids = state["batch"][self.image_id_key]
        self.batch_image_ids = self.batch_image_ids.numpy()

        pred_cls = state["batch"][self.pred_cls_key]

        cls_best_score = tf.reduce_max(pred_cls, axis=-1)
        cls_best_class = tf.argmax(pred_cls, axis=-1)
        cls_best_score_np = cls_best_score.numpy()
        cls_best_class_np = cls_best_class.numpy()

        
        cls_gt = state["batch"][self.cls_gt_key]
        cls_gt_np = cls_gt.numpy()
        x1_gt = state["batch"][self.x1_gt_key]
        x1_gt_np = x1_gt.numpy()
        w_gt = state["batch"][self.w_gt_key]
        w_gt_np = w_gt.numpy()
        y1_gt = state["batch"][self.y1_gt_key]
        y1_gt_np = y1_gt.numpy()
        h_gt = state["batch"][self.h_gt_key]
        h_gt_np = h_gt.numpy()


        num_gt_bb =  state["batch"][self.num_gt_bb_key]
        num_gt_bb_np = num_gt_bb.numpy()
        cls_gt_bb = state["batch"][self.cls_gt_bb_key]
        cls_gt_bb_np = cls_gt_bb.numpy()
        x1_gt_bb = state["batch"][self.x1_gt_bb_key]
        x1_gt_bb_np = x1_gt_bb.numpy()
        w_gt_bb = state["batch"][self.w_gt_bb_key]
        w_gt_bb_np = w_gt_bb.numpy()
        y1_gt_bb = state["batch"][self.y1_gt_bb_key]
        y1_gt_bb_np = y1_gt_bb.numpy()
        h_gt_bb = state["batch"][self.h_gt_bb_key]
        h_gt_bb_np = h_gt_bb.numpy()



        predicted_bb = []
        for idx, (sip_elem , vo_elem, loc_pred_elem, score_pred_elem, cls_pred_elem, image_id_elem) in enumerate( zip(selected_indices_padded_np, 
        valid_outputs_np, loc_pred_abs_np,cls_best_score_np, cls_best_class_np, self.batch_image_ids)):
            self.image_ids.append(image_id_elem)
            sel_indices = sip_elem[:vo_elem]
            for ind in sel_indices:
                x1,y1,w,h = loc_pred_elem[ind]
                category = cls_pred_elem[ind]
                score = score_pred_elem[ind]
                tmp_dict = {'idx':image_id_elem, 'x1':x1,'y1':y1,'w':w, 'h':h,'cls': category, 'score':score}
                predicted_bb.append(tmp_dict)

        ground_truth_bb  = []
        for idx, (num_gt_bb_elem, cls_gt_bb_elem, x1_gt_bb_elem, y1_gt_bb_elem, w_gt_bb_elem, h_gt_bb_elem, image_id_elem) in enumerate(
            zip(num_gt_bb_np, cls_gt_bb_np, x1_gt_bb_np, y1_gt_bb_np, w_gt_bb_np, h_gt_bb_np, self.batch_image_ids)):

            
            for bb_idx in range(num_gt_bb_elem):
                x1_gt, y1_gt , w_gt, h_gt = x1_gt_bb_elem[bb_idx], y1_gt_bb_elem[bb_idx], w_gt_bb_elem[bb_idx], h_gt_bb_elem[bb_idx]
                cls_gt = cls_gt_bb_elem[bb_idx]
                temp_dict = {'idx': image_id_elem, 'x1':x1_gt,'y1':y1_gt,'w':w_gt,'h':h_gt,'cls':cls_gt }
                ground_truth_bb.append(temp_dict)
            #if (image_id_elem == 483375):
            #    pdb.set_trace()


        for dict_elem in ground_truth_bb:
            self.gt[dict_elem['idx'], dict_elem['cls']].append(dict_elem)
        for dict_elem in predicted_bb:
            self.dt[dict_elem['idx'], dict_elem['cls']].append(dict_elem)

        num_category = len(self.categories)
        self.ious = {(img_id,cat_id): self.compute_iou(self.dt[img_id, cat_id], self.gt[img_id,cat_id]) for img_id in
                self.batch_image_ids for cat_id in self.categories }
        if (483375,6) in self.ious.keys():
            print('print the value of 483375,6',self.ious[483375, 6])
        for cat_id in self.categories:
            for img_id in self.batch_image_ids:
                self.evalImgs[(cat_id,img_id)] = self.evaluate_img(cat_id, img_id)


    def on_epoch_end(self, state):
        self.accumulate()
        self.summarize()
        self.summarize(iou=0.75)

    def accumulate(self):
        key_list = self.evalImgs
        key_list = sorted(key_list)
        eval_list = [self.evalImgs[key] for key in key_list]
        
        T = len(self.iouThrs)
        R  = len(self.recThrs)
        K = len(self.categories)
        cat_list = self.categories

        I = len(self.image_ids)
        maxDet = self.maxDets

        print('length of eval_list',len(eval_list))
        print('length of image_ids', len(self.image_ids))
        print('length of category',len(self.categories))


        precision = -np.ones((T,R,K))
        recall = -np.ones((T,K))
        scores = -np.ones((T,R,K))


        for k in cat_list:
            Nk = k*I
            E = [eval_list[Nk+img_idx]  for img_idx in range(I)]
            E = [e for e in E if not e is None]
            if len(E) == 0:
                continue
            dtScores = np.concatenate([e['dtScores'][0:maxDet] for e in E])

            inds = np.argsort(-dtScores, kind='mergesort')
            dtScoresSorted = dtScores[inds]


            dtm  = np.concatenate([e['dtMatches'][:,0:maxDet] for e in E], axis=1)[:,inds]
            dtIg = np.concatenate([e['dtIgnore'][:,0:maxDet]  for e in E], axis=1)[:,inds]

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
        if iou is not None:
            t = np.where(iou == self.iouThrs)[0]
            s = s[t]
        if len(s[s>-1])==0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s>-1])
            print("Mean average preci","{:.3f}".format(mean_s))

        print("Mean average preci","{:.3f}".format(mean_s))


    def evaluate_img(self, cat_id, img_id):
        T = self.iouThrs 
        dt = self.dt[ img_id, cat_id]
        gt = self.gt[ img_id, cat_id]
        num_dt = len(dt)
        num_gt = len(gt)
        if num_gt==0 and num_dt==0:
            return None

        dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in dtind[0:100]]

        iou_mat = self.ious[img_id, cat_id]

        # print("num_dt:",num_dt,"num_gt:",num_gt,"iou_mat:", iou_mat.shape)

        T = len(self.iouThrs)

        dtm = np.zeros((T, num_dt))
        gtm = np.zeros((T, num_gt))
        gtIg =[ 0 for g in gt]
        dtIg = np.zeros((T,num_dt))

        if len(iou_mat)!=0:
            for thres_idx, thres_elem in enumerate(self.iouThrs):
                for dt_idx, dt_elem in enumerate(dt):
                    m = -1
                    iou =  min([thres_elem,1-1e-10])
                    for gt_idx, gt_elem in enumerate(gt):
                        if gtm[thres_idx, gt_idx]>0:
                            continue
                        if iou_mat[dt_idx, gt_idx] >= iou:
                            iou = iou_mat[dt_idx, gt_idx]
                            m = gt_idx

                    if m != -1:
                        dtm[thres_idx, dt_idx] = gt[m]['idx']
                        gtm[thres_idx, m] = 1
                        dtIg[thres_idx, dt_idx] = gtIg[m]
        
        # dtIg = (dtm == 0)  
        if img_id ==483375  and cat_id==6 :
            print(dtm)

        return {
                'image_id':     img_id,
                'category_id':  cat_id,
                'gtIds':        [g['idx'] for g in gt],
                'dtMatches':    dtm,
                'gtMatches':    gtm,
                'dtScores':     [d['score'] for d in dt],
                'gtIgnore':     gtIg,
                'dtIgnore':     dtIg,
            }


    def compute_iou(self, dt, gt):
        num_dt = len(dt)
        num_gt = len(gt)

        if num_gt==0 and num_dt==0:   #  or?  and?
            return []

        boxes_a = np.zeros(shape=(0,4), dtype=float)
        boxes_b = np.zeros(shape=(0,4), dtype=float)

        inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in inds]
        if len(dt) > 100:
            dt=dt[0:100]

        #for dt_elem in dt:
        #    x1,y1,w,h= dt_elem['x1'], dt_elem['y1'], dt_elem['w'], dt_elem['h']
        #    boxes_a = np.append(boxes_a, np.array([[x1,y1,w,h]]), axis=0)

        #for gt_elem in gt:
        #    x1,y1,w,h =gt_elem['x1'], gt_elem['y1'], gt_elem['w'], gt_elem['h']
        #    boxes_b = np.append(boxes_b, np.array([[x1,y1,w,h]]), axis=0)

        #iou_dt_gt  = get_iou(boxes_a, boxes_b)

        boxes_a = [ [  dt_elem['x1'], dt_elem['y1'], dt_elem['w'], dt_elem['h']  ] for dt_elem in dt]
        boxes_b = [ [  gt_elem['x1'], gt_elem['y1'], gt_elem['w'], gt_elem['h']  ] for gt_elem in gt]

        iscrowd = [0 for o in gt]
        iou_dt_gt  = maskUtils.iou(boxes_a, boxes_b, iscrowd)
        return iou_dt_gt
