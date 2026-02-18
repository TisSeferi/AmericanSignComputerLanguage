from dataclasses import dataclass


@dataclass
class JkBlades:
    resample_cnt: int = 8
    radius: int = 8
    euclidean_distance: bool = True
    z_normalize: bool = True
    inner_product: bool = False
    lower_bound: bool = True
    cf_abs_distance: bool = True
    cf_bb_widths: bool = True

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
