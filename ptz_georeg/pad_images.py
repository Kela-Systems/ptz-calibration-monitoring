from .utils import pad_images_in_directory

if __name__ == '__main__':
    input_dir = 'C:/Users/vasil/Documents/data/PTZ_Calibration_and_georegistration/gan_shomron/20251106_175440_scan_equirect_-1.0_0.0_0.0/20251106_175440_scan_equirect_-1.0_0.0_0.0/frames'
    output_dir = 'C:/Users/vasil/Documents/data/PTZ_Calibration_and_georegistration/gan_shomron/20251106_175440_scan_equirect_-1.0_0.0_0.0/20251106_175440_scan_equirect_-1.0_0.0_0.0/frames_padded'
    pad_images_in_directory(input_dir, output_dir, 1520)