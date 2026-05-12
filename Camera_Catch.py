import cv2
import numpy as np
import os
import time
import sys
import platform # 新增：用于判断操作系统
from pupil_apriltags import Detector

# ================= 1. 驱动与 DLL/SO 加载 (通用适配) =================
current_dir = os.path.dirname(os.path.abspath(__file__))
libs_path = os.path.join(current_dir, 'libs')

if platform.system() == 'Windows':
    # Windows 环境：必须手动添加 DLL 目录
    if os.path.exists(libs_path):
        os.add_dll_directory(libs_path) 
else:
    # Linux 环境：将库路径加入系统搜索路径
    # 提醒：请确保 libs 目录下放置的是 .so 文件
    sys.path.append(libs_path)
    sys.path.append(current_dir)

# 导入相机 SDK
from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat, OBAlignMode, VideoStreamProfile

class Gemini335Camera:
    def __init__(self):
        self.pipeline = Pipeline()
        self.config = Config()
        self.intrinsics = None

        # --- 调整检测器参数 ---
        self.at_detector = Detector(families='tag36h11', nthreads=4, quad_decimate=1.0, decode_sharpening=0.5)
        self.tag_history = {} # 格式: {id: {'dists': [], 'state': 1}}

        try:
            # --- 强制请求 1280x720 高分辨率 ---
            color_profiles = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            try:
                color_profile = color_profiles.get_video_stream_profile(1280, 720, OBFormat.MJPG, 30)
            except:
                color_profile = color_profiles.get_default_video_stream_profile()
            self.config.enable_stream(color_profile)

            depth_profiles = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            depth_profile = depth_profiles.get_default_video_stream_profile()
            self.config.enable_stream(depth_profile)

            self.config.set_align_mode(OBAlignMode.SW_MODE)
            self.pipeline.start(self.config)

            time.sleep(1.0) 
            self.camera_param = self.pipeline.get_camera_param()
            # 兼容新旧版 SDK 属性名
            self.intrinsics = getattr(self.camera_param, 'rgb_intrinsic', getattr(self.camera_param, 'color_intrinsic', None))

            print(f"Gemini 335 启动。当前系统: {platform.system()}, 分辨率: {color_profile.get_width()}x{color_profile.get_height()}")
        except Exception as e:
            print(f"初始化失败: {e}")
            sys.exit(1)

    def get_real_3d_pose(self, u, v, depth_z):
        """ 标定转换：像素+深度 -> 3D坐标(XYZ) + 直线距离(R) """
        if self.intrinsics is None or depth_z <= 0:
            return None
        x = (u - self.intrinsics.cx) * depth_z / self.intrinsics.fx
        y = (v - self.intrinsics.cy) * depth_z / self.intrinsics.fy
        z = float(depth_z)
        r = np.sqrt(x**2 + y**2 + z**2)
        return {'point_3d': (round(x, 2), round(y, 2), round(z, 2)), 'real_dist': round(r, 2)}

    def capture_image(self):
        """ 采集并解码图像 """
        frames = self.pipeline.wait_for_frames(100)
        if not frames: return None, None
        cf = frames.get_color_frame()
        df = frames.get_depth_frame()
        if not cf or not df: return None, None

        raw_data = cf.get_data()
        if cf.get_format() == OBFormat.MJPG:
            color_img = cv2.imdecode(np.frombuffer(raw_data, dtype=np.uint8), cv2.IMREAD_COLOR)
        else:
            color_img = np.asanyarray(raw_data).reshape((cf.get_height(), cf.get_width(), 3))
            if cf.get_format() == OBFormat.RGB888:
                color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)

        depth_data = np.asanyarray(df.get_data()).view(np.uint16).reshape((df.get_height(), df.get_width()))
        return color_img, depth_data

    def detect_two_level_tags(self, color_img, depth_img, coarse_limit=3500, precise_limit=1000):
        """ 二级识别逻辑 (包含平滑滤波和滞后缓冲区) """
        gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
        tags = self.at_detector.detect(gray)
        
        valid_results = []
        buffer = 50  # 滞后缓冲区 50mm

        for tag in tags:
            tid = tag.tag_id
            ix, iy = int(tag.center[0]), int(tag.center[1])
            
            # --- 多点平均深度采样 ---
            depth_samples = []
            corners = tag.corners.astype(int)
            for (sx, sy) in list(corners) + [(ix, iy)]:
                if 0 <= sy < depth_img.shape[0] and 0 <= sx < depth_img.shape[1]:
                    d = depth_img[sy, sx]
                    if d > 0: depth_samples.append(d)
            if len(depth_samples) < 3: continue
            raw_z = sum(depth_samples) / len(depth_samples)

            # --- 时域平滑滤波 (5帧平均) ---
            if tid not in self.tag_history:
                self.tag_history[tid] = {'dists': [raw_z], 'state': 1}
            else:
                self.tag_history[tid]['dists'].append(raw_z)
                if len(self.tag_history[tid]['dists']) > 5:
                    self.tag_history[tid]['dists'].pop(0)
            
            smooth_z = sum(self.tag_history[tid]['dists']) / len(self.tag_history[tid]['dists'])
            
            pose = self.get_real_3d_pose(ix, iy, smooth_z)
            if not pose: continue
            
            r_dist = pose['real_dist']
            current_state = self.tag_history[tid]['state']

            # --- 滞后判别逻辑 (防闪烁) ---
            if current_state == 1: # 当前是粗略
                new_state = 2 if r_dist < (precise_limit - buffer) else 1
            else: # 当前是精细
                new_state = 1 if r_dist > (precise_limit + buffer) else 2
            
            self.tag_history[tid]['state'] = new_state

            if r_dist <= coarse_limit:
                valid_results.append({
                    'id': tid,
                    'level': new_state,
                    'xyz': pose['point_3d'],
                    'r': r_dist,
                    'center': (ix, iy),
                    'corners': corners
                })
        return valid_results

    def show_realtime(self):
        print("多窗口监控模式启动...")
        while True:
            color, depth = self.capture_image()
            if color is None: continue

            tags = self.detect_two_level_tags(color, depth)

            canvas_rgb = color.copy()
            for tag in tags:
                bgr = (0, 255, 0) if tag['level'] == 2 else (0, 255, 255)
                mode_str = "PRECISE" if tag['level'] == 2 else "COARSE"

                cv2.polylines(canvas_rgb, [tag['corners']], True, bgr, 2)
                x, y, z = tag['xyz']
                info = f"[{mode_str}] ID:{tag['id']} R:{int(tag['r'])}mm"
                pos_info = f"X:{x} Y:{y} Z:{z}"
                cv2.putText(canvas_rgb, info, (tag['center'][0]-80, tag['center'][1]-35), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 2)
                cv2.putText(canvas_rgb, pos_info, (tag['center'][0]-80, tag['center'][1]-15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 2)

            # 深度图伪彩色
            depth_view = cv2.applyColorMap(cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U), cv2.COLORMAP_JET)
            if canvas_rgb.shape[0] != depth_view.shape[0]:
                depth_view = cv2.resize(depth_view, (canvas_rgb.shape[1], canvas_rgb.shape[0]))
            
            combined_split = np.hstack((canvas_rgb, depth_view))
            fused_img = cv2.addWeighted(canvas_rgb, 0.6, depth_view, 0.4, 0)

            cv2.imshow("1. Recognition (XYZ+R Mode)", canvas_rgb)
            cv2.imshow("2. Split View (RGB + Depth)", combined_split)
            cv2.imshow("3. Fusion View (Calibration Check)", fused_img)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'): break
            elif key & 0xFF == ord('s'):
                cv2.imwrite("saved_color.jpg", color)

    def stop(self):
        try: self.pipeline.stop()
        except: pass
        cv2.destroyAllWindows()

if __name__ == "__main__":
    cam = Gemini335Camera()
    try:
        cam.show_realtime()
    finally:
        cam.stop()