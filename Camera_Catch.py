import cv2
import numpy as np
import os
import time
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
libs_path = os.path.join(current_dir, 'libs')

from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat, OBAlignMode, VideoStreamProfile


class Gemini335Camera:
    def __init__(self):
        self.pipeline = Pipeline()
        self.config = Config()

        try:
            # 1. 配置彩色流
            color_profiles = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            color_profile = color_profiles.get_default_video_stream_profile()
            self.config.enable_stream(color_profile)

            # 2. 配置深度流
            depth_profiles = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            depth_profile = depth_profiles.get_default_video_stream_profile()
            self.config.enable_stream(depth_profile)

            # 3. 开启硬件对齐 (Gemini 335 特色：深度图完美对齐彩色图)
            self.config.set_align_mode(OBAlignMode.ALIGN_D2C_HW_MODE)

            # 4. 启动相机
            self.pipeline.start(self.config)
            print("Gemini 335 启动成功！")
        except Exception as e:
            print(f"初始化失败: {e}")

    def capture_image(self):
        """
        功能：抓取当前一帧图像
        返回：color_image, depth_image
        """
        frames = self.pipeline.wait_for_frames(100)
        if frames is None:
            return None, None

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if color_frame is None or depth_frame is None:
            return None, None

        # 转换为 numpy 格式
        color_data = np.asanyarray(color_frame.get_data()).reshape((color_frame.get_height(), color_frame.get_width(), 3))
        color_img = cv2.cvtColor(color_data, cv2.COLOR_RGB2BGR)

        depth_data = np.asanyarray(depth_frame.get_data()).view(np.uint16).reshape((depth_frame.get_height(), depth_frame.get_width()))

        return color_img, depth_data

    def show_realtime(self):
        """
        功能：实时图像预览
        操作：按 's' 键抓取单帧，按 'q' 键退出
        """
        print("进入预览模式。按 's' 抓取并保存图像，按 'q' 退出。")
        while True:
            color, depth = self.capture_image()
            if color is None: continue

            # 为了预览，对深度图进行伪彩色处理
            depth_view = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            depth_view = cv2.applyColorMap(depth_view, cv2.COLORMAP_JET)

            # 左右拼接显示
            combined = np.hstack((color, depth_view))
            cv2.imshow("Gemini 335 Preview (RGB | Depth)", combined)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            elif key & 0xFF == ord('s'):
                timestamp = int(time.time())
                cv2.imwrite(f"capture_{timestamp}_color.jpg", color)
                np.save(f"capture_{timestamp}_depth.npy", depth)
                print(f"图像已保存：capture_{timestamp}")

    def stop(self):
        self.pipeline.stop()
        cv2.destroyAllWindows()

# --- 调用示例 ---
if __name__ == "__main__":
    cam = Gemini335Camera()
    
    # 场景 1：实时看图像（适合调试机械臂位置）
    cam.show_realtime()

    # 场景 2：在你的代码里直接抓取一张图给算法用
    # color, depth = cam.capture_image()
    # if color is not None:
    #     print("单帧抓取成功")

    cam.stop()