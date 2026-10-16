import os
from PIL import Image
import glob

def check_and_fix_mujoco_textures(folder_path, gridsize=2):
    """
    专门为MuJoCo检查并修复纹理尺寸
    gridsize: MuJoCo的网格大小，默认为2
    """
    png_files = glob.glob(os.path.join(folder_path, '**/*.png'), recursive=True)
    
    fixed_count = 0
    already_ok_count = 0
    
    for png_file in png_files:
        try:
            with Image.open(png_file) as img:
                width, height = img.size
                
                # 检查是否符合MuJoCo的gridsize要求
                width_ok = width % gridsize == 0
                height_ok = height % gridsize == 0
                
                if width_ok and height_ok:
                    print(f"✓ {os.path.basename(png_file)}: 尺寸正常 ({width}x{height} 是 {gridsize} 的倍数)")
                    already_ok_count += 1
                    continue
                
                # 计算新的符合要求的尺寸
                new_width = 512 #(width // gridsize) * gridsize  # 向下取整到最近的倍数
                new_height = 512 #(height // gridsize) * gridsize
                
                # 确保至少为gridsize
                new_width = max(new_width, gridsize)
                new_height = max(new_height, gridsize)
                
                print(f"✗ {os.path.basename(png_file)}: {width}x{height} -> {new_width}x{new_height}")
                
                # 创建备份
                backup_path = png_file.replace('.png', '_backup.png')
                img.save(backup_path)
                print(f"  备份保存为: {os.path.basename(backup_path)}")
                
                # 调整尺寸并保存
                resized_img = img.resize((new_width, new_height), Image.LANCZOS)
                resized_img.save(png_file, 'PNG')
                
                fixed_count += 1
                
        except Exception as e:
            print(f"错误处理 {png_file}: {e}")
    
    print(f"\n处理完成！")
    print(f"已修复: {fixed_count} 个文件")
    print(f"正常: {already_ok_count} 个文件")

def check_textures_only(folder_path, gridsize=2):
    """
    只检查不修改，用于预览
    """
    png_files = glob.glob(os.path.join(folder_path, '**/*.png'), recursive=True)
    
    problematic_files = []
    
    for png_file in png_files:
        try:
            with Image.open(png_file) as img:
                width, height = img.size
                
                width_ok = width % gridsize == 0
                height_ok = height % gridsize == 0
                
                status = "✓" if width_ok and height_ok else "✗"
                print(f"{status} {os.path.basename(png_file)}: {width}x{height}")
                
                if not (width_ok and height_ok):
                    problematic_files.append(png_file)
                    
        except Exception as e:
            print(f"错误读取 {png_file}: {e}")
    
    if problematic_files:
        print(f"\n需要修复的文件 ({len(problematic_files)} 个):")
        for file in problematic_files:
            print(f"  {os.path.basename(file)}")

if __name__ == "__main__":
    folder_path = input("请输入包含纹理的文件夹路径: ").strip()
    
    if not os.path.exists(folder_path):
        print("文件夹不存在！")
        exit()
    
    # 首先检查所有文件
    print("=== 检查纹理尺寸 ===")
    check_textures_only(folder_path)
    
    # 询问是否修复
    choice = input("\n是否修复有问题的文件？(y/n): ").strip().lower()
    if choice == 'y':
        print("\n=== 开始修复 ===")
        check_and_fix_mujoco_textures(folder_path)
    else:
        print("已取消修复")