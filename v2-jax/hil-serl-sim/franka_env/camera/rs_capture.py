import numpy as np

try:
    import pyrealsense2 as rs  # Intel RealSense cross-platform open-source API
except Exception as _rs_import_err:  # noqa: BLE001
    rs = None
    _RS_IMPORT_ERR = _rs_import_err
else:
    _RS_IMPORT_ERR = None


class RSCapture:
    """RealSense 采集；无 pyrealsense2 时仅占位，仿真 / learner 不应实例化本类。"""

    def get_device_serial_numbers(self):
        if rs is None:
            raise RuntimeError(
                "pyrealsense2 is not available on this machine "
                f"({_RS_IMPORT_ERR}). RSCapture is only for real hardware."
            )
        devices = rs.context().devices
        return [d.get_info(rs.camera_info.serial_number) for d in devices]

    def __init__(self, name, serial_number, dim=(640, 480), fps=15, depth=False, exposure=40000):
        if rs is None:
            raise RuntimeError(
                "pyrealsense2 is not available on this machine "
                f"({_RS_IMPORT_ERR}). RSCapture is only for real hardware."
            )
        self.name = name
        assert serial_number in self.get_device_serial_numbers()
        self.serial_number = serial_number
        self.depth = depth
        self.pipe = rs.pipeline()
        self.cfg = rs.config()
        self.cfg.enable_device(self.serial_number)
        self.cfg.enable_stream(rs.stream.color, dim[0], dim[1], rs.format.bgr8, fps)
        if self.depth:
            self.cfg.enable_stream(rs.stream.depth, dim[0], dim[1], rs.format.z16, fps)
        self.profile = self.pipe.start(self.cfg)
        self.s = self.profile.get_device().query_sensors()[0]
        self.s.set_option(rs.option.exposure, exposure)

        align_to = rs.stream.color
        self.align = rs.align(align_to)

    def read(self):
        frames = self.pipe.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        if self.depth:
            depth_frame = aligned_frames.get_depth_frame()

        if color_frame.is_video_frame():
            image = np.asarray(color_frame.get_data())
            if self.depth and depth_frame.is_depth_frame():
                depth = np.expand_dims(np.asarray(depth_frame.get_data()), axis=2)
                return True, np.concatenate((image, depth), axis=-1)
            else:
                return True, image
        else:
            return False, None

    def close(self):
        self.pipe.stop()
        self.cfg.disable_all_streams()
