def get_center_box(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1 + x2)/2), int((y2 + y2)/2)


def get_width_box(bbox):
    return bbox[2]-bbox[0]