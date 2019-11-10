from fastestimator.trace import Trace
import numpy as np
import tensorflow as tf
from collections import defaultdict
import pdb

#'selected_indices_padded','valid_outputs', 'abs_loc','image_id','pred_cls',"cls_gt", "x1_gt", "y1_gt", "x2_gt", "y2_gt"

class Mean_avg_precision(Trace):
    def __init__(self, categories, maxDets, selected_indices_key, valid_output_key, image_id_key, pred_cls_key, abs_loc_key, 
    cls_gt_key, x1_gt_key, y1_gt_key, x2_gt_key, y2_gt_key, mode="eval", output_name="map"):

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

        self.iouThrs = np.linspace(.5, 0.95, np.round((0.95 - .5) / .05) + 1, endpoint=True)
        self.recThrs = np.linspace(.0, 1.00, np.round((1.00 - .0) / .01) + 1, endpoint=True)
        self.categories = np.array(categories)
        self.categories  = [n for n,cat in enumerate(self.categories)]
        self.maxDets = maxDets
        self.image_ids = []
        self.anch_box = get_fpn_anchor_box(input_shape=(512, 512, 3))

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
        x2_gt = state["batch"][self.x2_gt_key]
        x2_gt_np = x2_gt.numpy()
        y1_gt = state["batch"][self.y1_gt_key]
        y1_gt_np = y1_gt.numpy()
        y2_gt = state["batch"][self.y2_gt_key]
        y2_gt_np = y2_gt.numpy()


        predicted_bb = []
        for idx, (sip_elem , vo_elem, loc_pred_elem, score_pred_elem, cls_pred_elem, image_id_elem) in enumerate( zip(selected_indices_padded_np, 
        valid_outputs_np, loc_pred_abs_np,cls_best_score_np, cls_best_class_np, self.batch_image_ids)):
            self.image_ids.append(image_id_elem)
            sel_indices = sip_elem[:vo_elem]
            for ind in sel_indices:
                x1,y1,x2,y2 = loc_pred_elem[ind]
                category = cls_pred_elem[ind]
                score = score_pred_elem[ind]
                tmp_dict = {'idx':image_id_elem, 'x1':x1,'y1':y1,'x2':x2, 'y2':y2,'cls': category, 'score':score}
                predicted_bb.append(tmp_dict)

        ground_truth_bb  = []
        for idx, (cls_gt_elem, x1_gt_elem, y1_gt_elem, x2_gt_elem, y2_gt_elem, image_id_elem) in enumerate(zip(cls_gt_np, x1_gt_np, y1_gt_np, x2_gt_np, y2_gt_np, self.batch_image_ids)):
            gt_valid_idx = np.where(cls_gt_elem>=0)
            gt_valid_bbs = len(gt_valid_idx[0])
            cls_gt_inst = cls_gt_elem[gt_valid_idx[0]]
            anch_box = self.anch_box[gt_valid_idx[0]]
            
            for bb_idx in range(gt_valid_bbs):
                x1_ac ,y1_ac, x2_ac, y2_ac = anch_box[bb_idx] 
                w_ac = x2_ac - x1_ac
                h_ac = y2_ac -y1_ac
                x1_gt, y1_gt , x2_gt, y2_gt = x1_gt_elem[bb_idx], y1_gt_elem[bb_idx], x2_gt_elem[bb_idx], y2_gt_elem[bb_idx]
                x1 = x1_gt * (w_ac) + x1_ac
                y1 = y1_gt * (h_ac) + y1_ac
                x2 = x2_gt * (w_ac) + x2_ac
                y2 = y2_gt * (h_ac) + y2_ac
                temp_dict = {'idx': image_id_elem, 'x1':x1,'y1':y1,'x2':x2,'y2':y2,'cls':cls_gt_inst[bb_idx] }
                ground_truth_bb.append(temp_dict)

        ground_truth_bb =  [ dict(t) for t in set(tuple(sorted(d.items())) for d in ground_truth_bb)]
        for dict_elem in ground_truth_bb:
            self.gt[dict_elem['idx'], dict_elem['cls']].append(dict_elem)
        for dict_elem in predicted_bb:
            self.dt[dict_elem['idx'], dict_elem['cls']].append(dict_elem)

        num_category = len(self.categories)
        self.ious = {(img_id,cat_id): self.compute_iou(self.dt[img_id, cat_id], self.gt[img_id,cat_id]) for img_id in
                self.batch_image_ids for cat_id in self.categories }
        for cat_id in self.categories:
            for img_id in self.batch_image_ids:
                self.evalImgs[(cat_id,img_id)] = self.evaluate_img(cat_id, img_id)


    def on_epoch_end(self, state):

        self.accumulate()
        self.summarize()

    def accumulate(self):

        key_list = self.evalImgs
        key_list = sorted(key_list)
        eval_list = [self.evalImgs[key] for key in key_list]
        
        T = len(self.iouThrs)
        R  = len(self.recThrs)
        K = len(self.categories)
        cat_list = self.categories

        I = len(self.image_ids)

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
        maxDet = self.maxDets
        dt = self.dt[ img_id, cat_id]
        gt = self.gt[ img_id, cat_id]
        num_dt = len(dt)
        num_gt = len(gt)
        if num_gt==0 and num_dt==0:
            return None

        dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in dtind[0:self.maxDets]]

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
                        if iou_mat[dt_idx, gt_idx] > iou:
                            iou = iou_mat[dt_idx, gt_idx]
                            m = gt_idx

                    if m != -1:
                        # pdb.set_trace()
                        dtm[thres_idx, dt_idx] = gt[m]['idx']
                        gtm[thres_idx, gt_idx] = 1
        
        # dtIg = (dtm == 0)  
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
        if len(dt) > self.maxDets:
            dt=dt[0:self.maxDets]

        for dt_elem in dt:
            x1,y1,x2,y2= dt_elem['x1'], dt_elem['y1'], dt_elem['x2'], dt_elem['y2']
            boxes_a = np.append(boxes_a, np.array([[x1,y1,x2,y2]]), axis=0)

        for gt_elem in gt:
            x1,y1,x2,y2 =gt_elem['x1'], gt_elem['y1'], gt_elem['x2'], gt_elem['y2']
            boxes_b = np.append(boxes_b, np.array([[x1,y1,x2,y2]]), axis=0)

        iou_dt_gt  = get_iou(boxes_a, boxes_b)
        return iou_dt_gt

def get_iou(boxes1, boxes2):
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

def get_fpn_anchor_box(input_shape):
    """Returns the anchor boxes of the Feature Pyramid Net.

    Args:
        input_shape (tuple): shape of input image.

    Returns:
        array: numpy array with all anchor boxes.
    """
    assert len(input_shape) == 3
    h, w, _ = input_shape
    assert h % 32 == 0 and w % 32 == 0
    sizes = [32, 64, 128, 256, 512]
    shapes = [(int(h / 8), int(w / 8))]  # P3
    num_pixel = np.prod(shapes)
    for _ in range(4):  # P4 through P7
        shapes.append((int(np.ceil(shapes[-1][0] / 2)), int(np.ceil(shapes[-1][1] / 2))))
        num_pixel += np.prod(shapes[-1])
    anchorbox = np.zeros((9 * num_pixel, 4))
    base_multipliers = [2**(0.0), 2**(1 / 3), 2**(2 / 3)]
    aspect_ratio_multiplier = [(1.0, 1.0), (2.0, 1.0), (1.0, 2.0)]
    anchor_idx = 0
    for shape,size in zip(shapes,sizes):
        p_h, p_w = shape
        base_y = h / p_h
        base_x = w / p_w
        for i in range(p_h):
            center_y = (i + 1 / 2) * base_y
            for j in range(p_w):
                center_x = (j + 1 / 2) * base_x
                for base_multiplier in base_multipliers:
                    for aspect_x, aspect_y in aspect_ratio_multiplier:
                        ratio = aspect_x/aspect_y
                        width_anch = base_multiplier * size
                        height_anch = base_multiplier * size
                        area = width_anch * height_anch
                        width_anch = np.sqrt(area*ratio)
                        height_anch = np.sqrt(area/ratio)
                        x1 = max(center_x - (width_anch) / 2, 0.0)  # x1
                        y1 = max(center_y - (height_anch) / 2, 0.0)  # y1
                        x2 = min(center_x + (width_anch) / 2, float(w))  # x2
                        y2 = min(center_y + (height_anch) / 2, float(h))  # y2
                        anchorbox[anchor_idx, 0] = x1
                        anchorbox[anchor_idx, 1] = y1
                        anchorbox[anchor_idx, 2] = x2 
                        anchorbox[anchor_idx, 3] = y2 
                        anchor_idx += 1
        if p_h == 1 and p_w == 1:  # the next level of 1x1 feature map is still 1x1, therefore ignore
            break
    return np.float32(anchorbox)
