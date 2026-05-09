import cv2
import numpy as np
import os
import time
import sys
from pupil_apriltags import Detector

current_dir = os.path.dirname(os.path.abspath(__file__))
libs_path = os.path.join(current_dir, 'libs')
if os.path.exists(libs_path):
    os.add_dll_directory(libs_path) # 告诉程序去 libs 找驱动

from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat, OBAlignMode, VideoStreamProfile


class Gemini335Camera:
    def __init__(self):
        self.pipeline = Pipeline()
        self.config = Config()

        # 初始化 AprilTag 检测器
        self.at_detector = Detector(families='tag36h11',nthreads=2)

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

    def detect_apriltags(self, color_img,depth_img,max_dist=2000):
        #识别AprilTag并通过深度传感器过渡距离
        gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
        # 仅识别 ID，不进行复杂的 3D 姿态推算
        tags = self.at_detector.detect(gray, estimate_tag_pose=False)
        
        valid_tags = []
        for tag in tags:
            # 获取中心点像素坐标
            ix, iy = int(tag.center[0]), int(tag.center[1])
            
            # 边界检查
            if iy >= depth_img.shape[0] or ix >= depth_img.shape[1]: continue
            
            # 直接从深度传感器获取该点的距离 (mm)
            dist = depth_img[iy, ix]
            
            # 限制在 2 米 (2000mm) 以内
            if 0 < dist <= max_dist:
                valid_tags.append({
                    'id': tag.tag_id,
                    'center': (ix, iy),
                    'distance': dist,
                    'corners': tag.corners.astype(int)
                })
        return valid_tags

    def show_realtime(self):
        """
        功能：实时图像预览
        操作：按 's' 键抓取单帧，按 'q' 键退出
        """
        print("进入预览模式。按 's' 抓取并保存图像，按 'q' 退出。")
        print("多模式预览启动：")
        print("- 窗口1: 原始RGB (带Tag识别)")
        print("- 窗口2: RGB+深度图拼接")
        print("- 窗口3: 深度融合图 (Fusion)")
        print("按 'q' 退出。")

        while True:
            color, depth = self.capture_image()
            if color is None: continue

            # 运行识别逻辑
            tags = self.detect_apriltags(color, depth, max_dist=3500)

            # 在彩色图上画出识别结果,拷贝出一个副本在副本上画图，保证原始color数据纯净
            canvas_rgb = color.copy()
            for tag in tags:
                # 画框
                cv2.polylines(canvas_rgb, [tag['corners']], True, (0, 255, 0), 2)
                # 画中心点
                cv2.circle(canvas_rgb, tag['center'], 5, (0, 0, 255), -1)
                # 显示 ID 和 深度传感器测得的距离
                info_text = f"ID:{tag['id']} Dist:{tag['distance']}mm"
                cv2.putText(canvas_rgb, info_text, (tag['center'][0]-50, tag['center'][1]-20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


            # 为了预览，对深度图进行伪彩色处理
            depth_view = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            depth_view = cv2.applyColorMap(depth_view, cv2.COLORMAP_JET)

            # 确保深度图和彩色图高度一致
            if color.shape[0] != depth_view.shape[0]:
                # 将 depth_view 缩放到和 color 一样的高度
                # 同时也按比例缩放宽度，或者直接缩放到彩色图的大小
                depth_view = cv2.resize(depth_view, (color.shape[1], color.shape[0]))

            #生成3种显示模式

            #模式A，只有RGB

            #模式B，RGB+深度图拼接
            combined_split = np.hstack((canvas_rgb, depth_view))
            
            # 模式 C: 融合图 (将伪彩色深度图透明叠加在 RGB 上)
            # 0.6 和 0.4 分别是 RGB 和深度的透明度权重，可以自行调整
            fused_img = cv2.addWeighted(canvas_rgb, 0.6, depth_view, 0.4, 0)

            # --- 4. 多窗口显示 ---
            # 窗口 1: 识别监控
            cv2.imshow("1. RGB Recognition", canvas_rgb)
            
            # 窗口 2: 拼接对比
            cv2.imshow("2. Split View (RGB + Depth)", combined_split)
            
            # 窗口 3: 融合视图
            cv2.imshow("3. Fusion View (Overlay)", fused_img)
            
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