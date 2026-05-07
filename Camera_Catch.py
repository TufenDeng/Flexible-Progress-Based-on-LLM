import numpy as np
import cv2
from openni import openni2
from openni import _openni2 as utils

class OrbbecCamera:
    def __init__(self, library_path):
        """
        :param library_path: OpenNI2 SDK 的 Redist 目录路径
        """
        # 1. 初始化 OpenNI2
        openni2.initialize(library_path)
        self.dev = openni2.Device.open_any()
        
        # 2. 创建深度流
        self.depth_stream = self.dev.create_depth_stream()
        self.depth_stream.start()
        self.depth_stream.set_video_mode(utils.OniVideoMode(
            pixelFormat=utils.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM, 
            resolutionX=640, resolutionY=480, fps=30))
        
        # 3. 创建彩色流
        # 注意：部分奥比中光型号（如Astra Pro）彩色图走的是普通UVC协议，
        # 如果下面代码报错，则需要用 cv2.VideoCapture(0) 获取彩色图
        try:
            self.color_stream = self.dev.create_color_stream()
            self.color_stream.start()
            self.color_stream.set_video_mode(utils.OniVideoMode(
                pixelFormat=utils.OniPixelFormat.ONI_PIXEL_FORMAT_RGB888, 
                resolutionX=640, resolutionY=480, fps=30))
            self.use_openni_color = True
        except Exception as e:
            print("彩色流无法通过OpenNI开启，切换至UVC模式 (OpenCV VideoCapture)")
            self.cap = cv2.VideoCapture(0) # 这里的索引可能需要根据实际调整
            self.use_openni_color = False

        # 4. 深度图与彩色图对齐 (关键)
        self.dev.set_image_registration_mode(openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR)

    def get_frames(self):
        """ 获取图像帧 """
        # 获取深度图
        d_frame = self.depth_stream.read_frame()
        d_data = d_frame.get_buffer_as_uint16()
        depth_img = np.frombuffer(d_data, dtype=np.uint16).reshape(480, 640)

        # 获取彩色图
        if self.use_openni_color:
            c_frame = self.color_stream.read_frame()
            c_data = c_frame.get_buffer_as_uint8()
            color_img = np.frombuffer(c_data, dtype=np.uint8).reshape(480, 640, 3)
            color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
        else:
            ret, color_img = self.cap.read()
            if not ret: return None, None
        
        return color_img, depth_img

    def show_realtime(self):
        """ 实时显示函数 """
        print("按 'q' 键退出预览，按 's' 键抓取保存图像")
        while True:
            color, depth = self.get_frames()
            if color is None: break

            # 深度图伪彩色化方便观察
            depth_show = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            depth_show = cv2.applyColorMap(depth_show, cv2.COLORMAP_JET)

            # 拼接显示
            combined = np.hstack((color, depth_show))
            cv2.imshow("Orbbec RGBD (Left: Color | Right: Depth)", combined)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            elif key & 0xFF == ord('s'):
                cv2.imwrite("captured_color.png", color)
                np.save("captured_depth.npy", depth) # 保存原始深度数据
                print("图像已抓取并保存！")

    def stop(self):
        self.depth_stream.stop()
        if self.use_openni_color:
            self.color_stream.stop()
        else:
            self.cap.release()
        openni2.unload()
        cv2.destroyAllWindows()

# --- 调用示例 ---
if __name__ == "__main__":
    # 请将此路径修改为你电脑上 OpenNI2 Redist 文件夹的绝对路径
    # 例如：r'C:\Program Files\OpenNI2\Redist'
    SDK_PATH = "./OpenNI2/Redist" 
    
    cam = OrbbecCamera(SDK_PATH)
    
    # 执行实时图像预览
    cam.show_realtime()
    
    cam.stop()