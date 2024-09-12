from topcow24_eval.constants import TASK, TRACK

# TODO: set the track and task
track = TRACK.CT
task = TASK.MULTICLASS_SEGMENTATION


# TODO: whether to crop the images for evaluations
# this is relevant to the multiclass segmentation task
# If `need_crop` is False, then `roi_path` will be ignored,
# and no cropping will be done.
# If `need_crop` is True and `roi_path` has roi_txt files,
# then the evaluations will be performed on the cropped gt, pred.
# it has no effect on Task 2 or Task 3
need_crop = False


# TODO: predictions, ground-truth (and roi-txt if need_crop)
# ALL NEED TO HAVE SAME NUMBER OF FILES
# set the number of cases manually here for sanity check
expected_num_cases = 2
