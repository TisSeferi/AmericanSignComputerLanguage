
class JkBlades:
    def __init__(self):
        self.resample_cnt = 8
        self.radius = 8
        self.euclidean_distance = True
        self.z_normalize = True
        self.inner_product = False
        self.lower_bound = True
        self.cf_abs_distance = True
        self.cf_bb_widths = True

    def set_ip_defaults(self):
        self.resample_cnt = 16
        self.radius = 2
        self.euclidean_distance = False
        self.z_normalize = False
        self.inner_product = True
        self.lower_bound = True
        self.cf_abs_distance = True
        self.cf_bb_widths = True

    def set_ed_defaults(self):
        self.resample_cnt = 16
        self.radius = 2
        self.euclidean_distance = True
        self.z_normalize = True
        self.inner_product = False
        self.lower_bound = True
        self.cf_abs_distance = True
        self.cf_bb_widths = True
