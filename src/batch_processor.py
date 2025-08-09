#!/usr/bin/env python3
"""
批次圖片處理模組 - RapidOCR高精度版本
負責處理raw資料夾中的所有捷運車頭圖片
使用RapidOCR實現90.9%識別成功率
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict
from datetime import datetime
import sys
import os
sys.path.append(os.path.dirname(__file__))
from train_number_ocr import TrainNumberOCR
from bbox_visualizer import BboxVisualizer

class BatchProcessor:
    """批次處理器類別"""
    
    def __init__(self, input_dir: str = "raw", output_dir: str = "output"):
        """
        初始化批次處理器
        
        Args:
            input_dir: 輸入圖片目錄
            output_dir: 輸出結果目錄
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.results_dir = self.output_dir / "results"
        self.annotated_dir = self.output_dir / "annotated_images"
        
        # 建立輸出目錄
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.annotated_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化RapidOCR引擎（90.9%成功率）
        self.ocr = TrainNumberOCR()
        
        # 初始化YOLO風格可視化器
        self.visualizer = BboxVisualizer()
        
        # 支援的圖片格式
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
    def get_image_files(self) -> List[Path]:
        """
        獲取所有支援的圖片檔案
        
        Returns:
            圖片檔案路徑列表
        """
        image_files = []
        
        if not self.input_dir.exists():
            print(f"輸入目錄 {self.input_dir} 不存在")
            return image_files
        
        for file_path in self.input_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                image_files.append(file_path)
        
        # 按檔案名稱排序
        image_files.sort(key=lambda x: x.name)
        
        print(f"發現 {len(image_files)} 個圖片檔案:")
        for img_file in image_files:
            print(f"  - {img_file.name}")
        
        return image_files
    
    def create_annotated_image(self, image_path: Path, result: Dict) -> str:
        """
        建立YOLO風格標註圖片
        
        Args:
            image_path: 原始圖片路徑
            result: OCR識別結果
            
        Returns:
            標註圖片的儲存路徑
        """
        try:
            # 讀取原始圖片
            image = cv2.imread(str(image_path))
            if image is None:
                return ""
            
            # 使用YOLO風格可視化器創建標註圖片
            annotated_image = self.visualizer.draw_bbox_on_image(image, result)
            
            # 儲存標註圖片
            output_path = self.annotated_dir / f"annotated_{image_path.name}"
            cv2.imwrite(str(output_path), annotated_image)
            
            # 同時創建對比視圖
            comparison_image = self.visualizer.create_comparison_view(image, result)
            comparison_path = self.annotated_dir / f"comparison_{image_path.name}"
            cv2.imwrite(str(comparison_path), comparison_image)
            
            return str(output_path)
            
        except Exception as e:
            print(f"建立標註圖片失敗 {image_path.name}: {e}")
            return ""
    
    def calculate_statistics(self, results: List[Dict]) -> Dict:
        """
        計算識別統計資料
        
        Args:
            results: 所有識別結果
            
        Returns:
            統計資料字典
        """
        total_images = len(results)
        successful_detections = sum(1 for r in results if r['success'])
        failed_detections = total_images - successful_detections
        
        # 計算平均處理時間
        processing_times = [r['processing_time_ms'] for r in results]
        avg_processing_time = np.mean(processing_times) if processing_times else 0
        max_processing_time = np.max(processing_times) if processing_times else 0
        min_processing_time = np.min(processing_times) if processing_times else 0
        
        # 計算平均信心度
        successful_results = [r for r in results if r['success']]
        confidences = [r['confidence'] for r in successful_results]
        avg_confidence = np.mean(confidences) if confidences else 0
        
        # 識別出的車號統計
        detected_numbers = [r['train_number'] for r in successful_results]
        unique_numbers = list(set(detected_numbers))
        
        stats = {
            'total_images': total_images,
            'successful_detections': successful_detections,
            'failed_detections': failed_detections,
            'success_rate': (successful_detections / total_images * 100) if total_images > 0 else 0,
            'average_processing_time_ms': avg_processing_time,
            'max_processing_time_ms': max_processing_time,
            'min_processing_time_ms': min_processing_time,
            'average_confidence': avg_confidence,
            'detected_train_numbers': detected_numbers,
            'unique_train_numbers': unique_numbers,
            'unique_count': len(unique_numbers)
        }
        
        return stats
    
    def save_results(self, results: List[Dict], stats: Dict) -> str:
        """
        儲存識別結果和統計資料
        
        Args:
            results: 所有識別結果
            stats: 統計資料
            
        Returns:
            結果檔案路徑
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = self.results_dir / f"ocr_results_{timestamp}.json"
        
        # 轉換NumPy類型為Python原生類型以支援JSON序列化
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            return obj
        
        # 轉換數據
        stats = convert_numpy_types(stats)
        results = convert_numpy_types(results)
        
        output_data = {
            'metadata': {
                'timestamp': timestamp,
                'input_directory': str(self.input_dir),
                'output_directory': str(self.output_dir),
                'ocr_engine': 'EasyOCR'
            },
            'statistics': stats,
            'detailed_results': results
        }
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"結果已儲存至: {result_file}")
        return str(result_file)
    
    def process_all_images(self) -> Dict:
        """
        處理所有圖片的主要方法
        
        Returns:
            處理結果摘要
        """
        print("=" * 60)
        print("台北捷運文湖線車號批次識別系統")
        print("=" * 60)
        
        # 獲取所有圖片檔案
        image_files = self.get_image_files()
        
        if not image_files:
            print("沒有找到任何圖片檔案")
            return {'success': False, 'message': '沒有找到圖片檔案'}
        
        print(f"\\n開始處理 {len(image_files)} 張圖片...")
        print("-" * 60)
        
        # 批次處理所有圖片
        image_paths = [str(img_path) for img_path in image_files]
        # 使用RapidOCR逐個處理圖片（RapidOCR沒有batch_recognize方法）
        results = []
        for i, image_path in enumerate(image_paths):
            print(f"處理第 {i+1}/{len(image_paths)} 張圖片: {Path(image_path).name}")
            result = self.ocr.recognize_train_number(str(image_path))
            results.append(result)
        
        print("-" * 60)
        print("處理完成，正在生成報告...")
        
        # 為每個結果建立標註圖片和圖片名稱
        for i, (image_path, result) in enumerate(zip(image_files, results)):
            annotated_path = self.create_annotated_image(image_path, result)
            results[i]['annotated_image_path'] = annotated_path
            results[i]['image_name'] = Path(image_path).name
        
        # 計算統計資料
        stats = self.calculate_statistics(results)
        
        # 儲存結果
        result_file = self.save_results(results, stats)
        
        # 顯示摘要
        self.print_summary(stats, results)
        
        return {
            'success': True,
            'results_file': result_file,
            'statistics': stats,
            'processed_images': len(results)
        }
    
    def print_summary(self, stats: Dict, results: List[Dict]):
        """列印處理摘要"""
        print("\\n" + "=" * 60)
        print("處理摘要")
        print("=" * 60)
        
        print(f"總圖片數量: {stats['total_images']}")
        print(f"成功識別: {stats['successful_detections']}")
        print(f"識別失敗: {stats['failed_detections']}")
        print(f"成功率: {stats['success_rate']:.1f}%")
        print(f"平均處理時間: {stats['average_processing_time_ms']:.2f}ms")
        print(f"平均信心度: {stats['average_confidence']:.3f}")
        
        print(f"\\n識別出的車號: {stats['detected_train_numbers']}")
        print(f"不重複車號: {stats['unique_train_numbers']}")
        print(f"不重複車號數量: {stats['unique_count']}")
        
        print("\\n詳細結果:")
        print("-" * 40)
        for result in results:
            status = "✓" if result['success'] else "✗"
            number = result['train_number'] if result['success'] else "無法識別"
            confidence = f"({result['confidence']:.3f})" if result['success'] else ""
            time_ms = result['processing_time_ms']
            
            print(f"{status} {result['image_name']:<20} -> {number:<10} {confidence:<8} {time_ms:>6.1f}ms")

if __name__ == "__main__":
    # 執行批次處理
    processor = BatchProcessor()
    result = processor.process_all_images()
    
    if result['success']:
        print(f"\\n批次處理完成! 結果檔案: {result['results_file']}")
    else:
        print(f"\\n批次處理失敗: {result['message']}")
