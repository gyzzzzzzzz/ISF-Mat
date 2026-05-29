from mmsegext.evaluation.metrics.region_metrics_utils import RegionMetricsParallel
rmp=RegionMetricsParallel(num_classes=2,thresholds=0.5,skip_class_num=[0,255],area_filter=[1,10**9])
