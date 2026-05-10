import cv2
import numpy as np
import os
import time
import sys
from pupil_apriltags import Detector

# ================= 驱动与 DLL 加载 =================
current_dir = os.path.dirname(os.path.abspath(__file__))
libs_path = os.path.join(current_dir, 'libs')
if os.path.exists(libs_path):
    os.add_dll_directory(libs_path) 

from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat, OBAlignMode, VideoStreamProfile


class Gemini335Camera:
    def __init__(self):
        self.pipeline = Pipeline()
        self.config = Config()
        self.intrinsics = None

        # 初始化 AprilTag 检测器
        self.at_detector = Detector(families='tag36h11', nthreads=2)

        try:
            # 1. 配置彩色流
            color_profiles = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            color_profile = color_profiles.get_default_video_stream_profile()
            self.config.enable_stream(color_profile)

            # 2. 配置深度流
            depth_profiles = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            depth_profile = depth_profiles.get_default_video_stream_profile()
            self.config.enable_stream(depth_profile)

            # 3. 开启软件对齐 (必选)
            self.config.set_align_mode(OBAlignMode.SW_MODE)

            # 4. 启动相机
            self.pipeline.start(self.config)

            # 等待硬件参数加载
            time.sleep(0.5) 

            # 获取出厂标定参数
            self.camera_param = self.pipeline.get_camera_param()
            self.intrinsics = self.camera_param.rgb_intrinsic 

            print("Gemini 335 启动成功！已加载出厂内参。")
        except Exception as e:
            print(f"初始化失败: {e}")
            sys.exit(1)

    def get_3d_coordinates(self, u, v, depth_mm):
        """ 标定转换：像素转3D坐标 """
        if self.intrinsics is None or depth_mm <= 0:
            return None

        # 基于内参的针孔模型公式
        x_mm = (u - self.intrinsics.cx) * depth_mm / self.intrinsics.fx
        y_mm = (v - self.intrinsics.cy) * depth_mm / self.intrinsics.fy
        z_mm = float(depth_mm)

        return (round(x_mm, 2), round(y_mm, 2), round(z_mm, 2))

    def capture_image(self):
        frames = self.pipeline.wait_for_frames(100)
        if frames is None:
            return None, None

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if color_frame is None or depth_frame is None:
            return None, None

        raw_data = color_frame.get_data()
        fmt = color_frame.get_format()

        if fmt == OBFormat.MJPG:
            color_img = cv2.imdecode(np.frombuffer(raw_data, dtype=np.uint8), cv2.IMREAD_COLOR)
        elif fmt == OBFormat.RGB888:
            color_img = np.asanyarray(raw_data).reshape((color_frame.get_height(), color_frame.get_width(), 3))
            color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
        else:
            try:
                color_img = np.asanyarray(raw_data).reshape((color_frame.get_height(), color_frame.get_width(), 3))
            except:
                return None, None

        depth_data = np.asanyarray(depth_frame.get_data()).view(np.uint16).reshape((depth_frame.get_height(), depth_frame.get_width()))
        return color_img, depth_data

    def detect_apriltags(self, color_img, depth_img, max_dist=2000):
        gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
        tags = self.at_detector.detect(gray, estimate_tag_pose=False)
        
        valid_tags = []
        for tag in tags:
            ix, iy = int(tag.center[0]), int(tag.center[1])
            if iy >= depth_img.shape[0] or ix >= depth_img.shape[1]: continue
            
            dist = depth_img[iy, ix]
            
            if 0 < dist <= max_dist:
                pos_3d = self.get_3d_coordinates(ix, iy, dist)
                # 仅添加能算出有效 3D 坐标的 Tag
                if pos_3d is not None:
                    valid_tags.append({
                        'id': tag.tag_id,
                        'center': (ix, iy),
                        'pos_3d': pos_3d,
                        'corners': tag.corners.astype(int)
                    })
        return valid_tags

    def show_realtime(self):
        print("预览模式启动：仅显示标定后的 XYZ 坐标。按 'q' 退出。")

        while True:
            color, depth = self.capture_image()
            if color is None: continue

            # 识别 (限制 3.5 米)
            tags = self.detect_apriltags(color, depth, max_dist=2000)

            canvas_rgb = color.copy()
            for tag in tags:
                # 画框和中心点
                cv2.polylines(canvas_rgb, [tag['corners']], True, (0, 255, 0), 2)
                cv2.circle(canvas_rgb, tag['center'], 5, (0, 0, 255), -1)
                
                # --- 核心修改：安全获取并仅显示 XYZ 标定坐标 ---
                pos = tag.get('pos_3d')
                if pos is not None:
                    x, y, z = pos
                    info_text = f"ID:{tag['id']} XYZ:[{x},{y},{z}]mm"
                else:
                    info_text = f"ID:{tag['id']} XYZ:Invalid"
                
                cv2.putText(canvas_rgb, info_text, (tag['center'][0]-80, tag['center'][1]-20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # 深度图伪彩色处理
            depth_view = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            depth_view = cv2.applyColorMap(depth_view, cv2.COLORMAP_JET)

            if color.shape[0] != depth_view.shape[0]:
                depth_view = cv2.resize(depth_view, (color.shape[1], color.shape[0]))

            # 生成显示视图
            combined_split = np.hstack((canvas_rgb, depth_view))
            fused_img = cv2.addWeighted(canvas_rgb, 0.6, depth_view, 0.4, 0)

            cv2.imshow("1. Recognition (XYZ)", canvas_rgb)
            cv2.imshow("2. Split View", combined_split)
            cv2.imshow("3. Fusion View", fused_img)
            
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            elif key & 0xFF == ord('s'):
                timestamp = int(time.time())
                cv2.imwrite(f"cap_{timestamp}.jpg", color)

    def stop(self):
        try:
            self.pipeline.stop()
        except:
            pass
        cv2.destroyAllWindows()

if __name__ == "__main__":
    cam = Gemini335Camera()
    try:
        cam.show_realtime()
    except KeyboardInterrupt:
        pass
    finally:
        cam.stop()