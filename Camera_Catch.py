import cv2
import numpy as np
import os
import time
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
libs_path = os.path.join(current_dir, 'libs')
if os.path.exists(libs_path):
    os.add_dll_directory(libs_path) # 告诉程序去 libs 找驱动

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

            # 3. 开启软件对齐 (Gemini 335 特色：深度图完美对齐彩色图)
            self.config.set_align_mode(OBAlignMode.SW_MODE)

            # 4. 启动相机
            self.pipeline.start(self.config)
            print("Gemini 335 启动成功！")
        except Exception as e:
            print(f"初始化失败: {e}")

    def capture_image(self):
        frames = self.pipeline.wait_for_frames(100)
        if frames is None:
            return None, None

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if color_frame is None or depth_frame is None:
            return None, None

        # 1. 处理彩色图
        raw_data = color_frame.get_data()
        fmt = color_frame.get_format()

        #使用 MJPG
        if fmt == OBFormat.MJPG:
            # 解码压缩格式
            color_img = cv2.imdecode(np.frombuffer(raw_data, dtype=np.uint8), cv2.IMREAD_COLOR)
        elif fmt == OBFormat.RGB888:
            # 原始 RGB 格式
            color_img = np.asanyarray(raw_data).reshape((color_frame.get_height(), color_frame.get_width(), 3))
            color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
        else:
            # 兼容其他可能出现的 YUV 格式（尝试直接 reshape，如果报错再处理）
            try:
                color_img = np.asanyarray(raw_data).reshape((color_frame.get_height(), color_frame.get_width(), 3))
            except:
                print(f"无法处理的图像格式: {fmt}")
                return None, None

        # 2. 处理深度图
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

            # 确保深度图和彩色图高度一致
            if color.shape[0] != depth_view.shape[0]:
                # 将 depth_view 缩放到和 color 一样的高度
                # 同时也按比例缩放宽度，或者直接缩放到彩色图的大小
                depth_view = cv2.resize(depth_view, (color.shape[1], color.shape[0]))

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