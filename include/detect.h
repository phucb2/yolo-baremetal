#ifndef DETECT_H
#define DETECT_H

#include "model.h"

/* Ultralytics Detect (reg_max=1, end2end): one2one heads on P3,P4,P5 -> postprocess [1, max_det, 6]. */
status_t detect_forward_one2one(model_t* model, int detect_module_idx,
                                const tensor_t* p3, const tensor_t* p4, const tensor_t* p5,
                                tensor_t* out_postprocess);

/* Test hook: preds row-major (N, 4+nc), xyxy pixels + per-class sigmoid scores; writes top-k into out (1,K,6,1). */
status_t detect_postprocess_from_pred(const float* pred_n_4plusnc, int N, int nc, int max_det,
                                      tensor_t* out_postprocess);

#endif
