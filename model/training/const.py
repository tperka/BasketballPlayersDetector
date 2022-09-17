COCO_STATS_CSV_HEADER = ["AP @ IoU=0.50:0.95",
              "AP @ IoU=0.50",
              "AP @ IoU=0.75",
              "AR @ IoU=0.5:0.95"]

# indexes selected because we don't need max dets or area distinction which is available in default COCO evaluator
COCO_STATS_INDEXES = [0, 1, 2, 8]
