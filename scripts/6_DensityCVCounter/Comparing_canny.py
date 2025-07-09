import cv2
import numpy as np
from scipy.signal import find_peaks
import os
import glob
import pandas as pd
from datetime import datetime

class EnhancedSheetStackCounter:
    def __init__(self, cluster_max_distance=30):
        self.cluster_max_distance = cluster_max_distance
        self.batch_results = []
        self.all_methods = ['canny_percentile', 'canny_otsu', 'canny_adaptive', 'canny_gradient_magnitude',
                           'canny_morphological', 'canny_iterative', 'canny_weighted', 'canny_bilateral_enhanced',
                           'canny_pyramid', 'canny_local_adaptive']
        
    def preprocess_image(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not load image from {img_path}")
        
        original = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply noise reduction and enhancement
        for _ in range(3):
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        
        return original, gray
    
    def detect_edges(self, gray, method='canny_percentile'):
        """Unified edge detection with multiple Canny methods"""
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        
        if method == 'canny_percentile':
            low_threshold = np.percentile(gray, 33)
            high_threshold = np.percentile(gray, 66)
            edges = cv2.Canny(gray, low_threshold, high_threshold, apertureSize=3, L2gradient=True)
            
        elif method == 'canny_otsu':
            otsu_threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            edges = cv2.Canny(gray, 0.5 * otsu_threshold, otsu_threshold, apertureSize=3, L2gradient=True)
            
        elif method == 'canny_adaptive':
            sigma = 0.33
            median = np.median(gray)
            lower = int(max(0, (1.0 - sigma) * median))
            upper = int(min(255, (1.0 + sigma) * median))
            edges = cv2.Canny(gray, lower, upper, apertureSize=3, L2gradient=True)
            
        elif method == 'canny_gradient_magnitude':
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(grad_x**2 + grad_y**2)
            mean_mag = np.mean(magnitude)
            std_mag = np.std(magnitude)
            lower = max(0, mean_mag - 0.5 * std_mag)
            upper = min(255, mean_mag + 1.5 * std_mag)
            edges = cv2.Canny(gray, lower, upper, apertureSize=3, L2gradient=True)
            
        elif method == 'canny_morphological':
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            morph_grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
            edges = cv2.Canny(morph_grad, 30, 90, apertureSize=3, L2gradient=True)
            
        elif method == 'canny_iterative':
            edges_combined = np.zeros_like(gray)
            threshold_pairs = [(30, 100), (50, 150), (20, 80)]
            for i, (low, high) in enumerate(threshold_pairs):
                if i == 0:
                    processed_gray = gray
                elif i == 1:
                    processed_gray = cv2.GaussianBlur(gray, (3, 3), 0)
                else:
                    processed_gray = cv2.bilateralFilter(gray, 9, 75, 75)
                edges = cv2.Canny(processed_gray, low, high, apertureSize=3, L2gradient=True)
                edges_combined = cv2.bitwise_or(edges_combined, edges)
            edges = edges_combined
            
        elif method == 'canny_weighted':
            edges_3 = cv2.Canny(gray, 50, 150, apertureSize=3, L2gradient=True)
            edges_5 = cv2.Canny(gray, 50, 150, apertureSize=5, L2gradient=True)
            edges_7 = cv2.Canny(gray, 50, 150, apertureSize=7, L2gradient=True)
            edges_combined = cv2.addWeighted(edges_3, 0.5, edges_5, 0.3, 0)
            edges_combined = cv2.addWeighted(edges_combined, 1.0, edges_7, 0.2, 0)
            _, edges = cv2.threshold(edges_combined, 127, 255, cv2.THRESH_BINARY)
            
        elif method == 'canny_bilateral_enhanced':
            bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
            gaussian = cv2.GaussianBlur(bilateral, (0, 0), 2.0)
            unsharp_mask = cv2.addWeighted(bilateral, 1.5, gaussian, -0.5, 0)
            edges = cv2.Canny(unsharp_mask, 30, 90, apertureSize=3, L2gradient=True)
            
        elif method == 'canny_pyramid':
            pyramid = [gray]
            current = gray
            for i in range(2):
                current = cv2.pyrDown(current)
                pyramid.append(current)
            
            edges_pyramid = []
            for level, img in enumerate(pyramid):
                low_thresh = 30 + level * 10
                high_thresh = 90 + level * 20
                edges_level = cv2.Canny(img, low_thresh, high_thresh, apertureSize=3, L2gradient=True)
                edges_pyramid.append(edges_level)
            
            edges = edges_pyramid[-1]
            for i in range(len(edges_pyramid) - 2, -1, -1):
                edges = cv2.pyrUp(edges)
                h, w = edges_pyramid[i].shape
                edges = cv2.resize(edges, (w, h))
                edges = cv2.bitwise_or(edges, edges_pyramid[i])
                
        elif method == 'canny_local_adaptive':
            h, w = gray.shape
            edges = np.zeros_like(gray)
            block_size = 64
            for y in range(0, h, block_size):
                for x in range(0, w, block_size):
                    y_end = min(y + block_size, h)
                    x_end = min(x + block_size, w)
                    block = gray[y:y_end, x:x_end]
                    if block.size == 0:
                        continue
                    local_mean = np.mean(block)
                    local_std = np.std(block)
                    low_thresh = max(10, local_mean - 0.5 * local_std)
                    high_thresh = min(255, local_mean + 1.5 * local_std)
                    block_edges = cv2.Canny(block, low_thresh, high_thresh, apertureSize=3, L2gradient=True)
                    edges[y:y_end, x:x_end] = block_edges
                    
        else:  # default canny
            edges = cv2.Canny(gray, 50, 150, apertureSize=3, L2gradient=True)
        
        return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, horizontal_kernel)
    
    def analyze_edge_density(self, edges, window_size=10):
        height, width = edges.shape
        edge_density = []
        for y in range(height):
            window_start = max(0, y - window_size // 2)
            window_end = min(height, y + window_size // 2 + 1)
            window_edges = edges[window_start:window_end, :]
            density = np.sum(window_edges) / (window_edges.shape[0] * window_edges.shape[1])
            edge_density.append(density)
        return np.array(edge_density)
    
    def find_peaks_in_density(self, edge_density, min_prominence=0.01, min_distance=5):
        peaks, properties = find_peaks(
            edge_density, 
            prominence=min_prominence,
            distance=min_distance,
            # height=np.mean(edge_density)
            height=int(((np.max(edge_density) + np.min(edge_density))/2)*1/6),
        )
        return peaks
    
    def cluster_peaks(self, peaks):
        if len(peaks) == 0:
            return [], []
        
        clusters = []
        current_cluster = [peaks[0]]
        
        for i in range(1, len(peaks)):
            distance = peaks[i] - current_cluster[-1]
            if distance <= self.cluster_max_distance:
                current_cluster.append(peaks[i])
            else:
                clusters.append(current_cluster)
                current_cluster = [peaks[i]]
        clusters.append(current_cluster)
        
        cluster_info = []
        for i, cluster in enumerate(clusters):
            cluster_dict = {
                'cluster_id': i + 1,
                'peaks': cluster,
                'count': len(cluster),
                'span': max(cluster) - min(cluster),
                'center_y': np.mean(cluster)
            }
            cluster_info.append(cluster_dict)
        
        return clusters, cluster_info
    
    def get_largest_cluster(self, cluster_info):
        if not cluster_info:
            return None
        return max(cluster_info, key=lambda x: x['count'])
    
    def process_single_image(self, img_path, method='canny_percentile'):
        try:
            original, gray = self.preprocess_image(img_path)
            edges = self.detect_edges(gray, method)
            edge_density = self.analyze_edge_density(edges, window_size=10)
            density_peaks = self.find_peaks_in_density(edge_density)
            clusters, cluster_info = self.cluster_peaks(density_peaks)
            largest_cluster = self.get_largest_cluster(cluster_info)
            
            result = {
                'image_path': img_path,
                'image_name': os.path.basename(img_path),
                'method': method,
                'total_peaks': len(density_peaks),
                'num_clusters': len(clusters),
                'final_count': largest_cluster['count'] if largest_cluster else 0,
                'largest_cluster_info': largest_cluster,
                'processing_status': 'success'
            }
            
            return result
            
        except Exception as e:
            return {
                'image_path': img_path,
                'image_name': os.path.basename(img_path),
                'method': method,
                'total_peaks': 0,
                'num_clusters': 0,
                'final_count': 0,
                'largest_cluster_info': None,
                'processing_status': f'error: {str(e)}'
            }
    
    def analyze_all_methods(self, folder_path, file_extensions=('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff')):
        """Analyze each image with all methods and print comprehensive table"""
        print(f"=== ANALYZING ALL METHODS FOR FOLDER: {folder_path} ===")
        
        # Find all image files
        image_files = []
        for extension in file_extensions:
            image_files.extend(glob.glob(os.path.join(folder_path, extension)))
            image_files.extend(glob.glob(os.path.join(folder_path, extension.upper())))
        
        # Remove duplicates by converting to set and back to list
        image_files = list(set(image_files))
        
        if not image_files:
            print("No image files found!")
            return None, None
        
        print(f"Found {len(image_files)} unique images. Processing with {len(self.all_methods)} methods...")
        
        # Process each image with all methods
        detailed_results = []
        summary_data = []
        
        for i, img_path in enumerate(sorted(image_files), 1):
            img_name = os.path.basename(img_path)
            print(f"[{i}/{len(image_files)}] Processing {img_name}...")
            
            method_counts = {}
            for method in self.all_methods:
                result = self.process_single_image(img_path, method)
                detailed_results.append(result)
                method_counts[method] = result['final_count']
            
            # Create summary row for this image
            summary_row = {'Image': img_name}
            summary_row.update(method_counts)
            summary_row['Best_Method'] = max(method_counts, key=method_counts.get)
            summary_row['Max_Count'] = max(method_counts.values())
            summary_row['Min_Count'] = min(method_counts.values())
            summary_row['Range'] = summary_row['Max_Count'] - summary_row['Min_Count']
            summary_data.append(summary_row)
        
        # Create DataFrames
        detailed_df = pd.DataFrame(detailed_results)
        summary_df = pd.DataFrame(summary_data)
        
        # Print comprehensive table
        self.print_comprehensive_table(summary_df)
        self.print_method_statistics(detailed_df)
        
        return detailed_df, summary_df
    
    def print_comprehensive_table(self, summary_df):
        """Print a comprehensive table with all methods and images"""
        print("\n" + "="*150)
        print("COMPREHENSIVE RESULTS TABLE - ALL METHODS vs ALL IMAGES")
        print("="*150)
        
        # Prepare method columns for display (shortened names for better formatting)
        method_display_names = {
            'canny_percentile': 'Percentile',
            'canny_otsu': 'Otsu',
            'canny_adaptive': 'Adaptive',
            'canny_gradient_magnitude': 'GradMag',
            'canny_morphological': 'Morph',
            'canny_iterative': 'Iterative',
            'canny_weighted': 'Weighted',
            'canny_bilateral_enhanced': 'Bilateral',
            'canny_pyramid': 'Pyramid',
            'canny_local_adaptive': 'LocalAdap'
        }
        
        # Create header
        header = f"{'Image':<20}"
        for method in self.all_methods:
            short_name = method_display_names.get(method, method[:9])
            header += f"{short_name:>10}"
        header += f"{'Best':>12}{'Max':>6}{'Min':>6}{'Range':>7}"
        
        print(header)
        print("-" * len(header))
        
        # Print each row
        for _, row in summary_df.iterrows():
            # Truncate image name if too long
            img_name = row['Image']
            if len(img_name) > 19:
                img_name = img_name[:16] + "..."
            
            line = f"{img_name:<20}"
            
            # Add method counts
            total = 0
            for method in self.all_methods:
                count = row[method]
                total += count
                line += f"{count:>10}"
            
            # Add summary statistics
            best_method = method_display_names.get(row['Best_Method'], row['Best_Method'][:9])
            # line += f"{best_method:>12}{row['Max_Count']:>6}{row['Min_Count']:>6}{row['Range']:>7}"
            mean_val = np.mean([row['Max_Count'], row['Min_Count']])
            line += f"{best_method:>12}{row['Max_Count']:>6}{row['Min_Count']:>6}{row['Range']:>7}{total/10:>8.2f}"

            
            print(line)
        
        print("-" * len(header))
        print(f"Total Images: {len(summary_df)}")
        
    def print_method_statistics(self, detailed_df):
        """Print method performance statistics"""
        print("\n" + "="*80)
        print("METHOD PERFORMANCE STATISTICS")
        print("="*80)
        
        method_stats = []
        for method in self.all_methods:
            method_data = detailed_df[detailed_df['method'] == method]
            stats = {
                'Method': method,
                'Mean': method_data['final_count'].mean(),
                'Std': method_data['final_count'].std(),
                'Max': method_data['final_count'].max(),
                'Min': method_data['final_count'].min(),
                'Success_Rate': (method_data['processing_status'] == 'success').mean() * 100
            }
            method_stats.append(stats)
        
        # Print header
        print(f"{'Method':<22}{'Mean':>8}{'Std':>8}{'Max':>6}{'Min':>6}{'Success%':>10}")
        print("-" * 80)
        
        # Sort by mean count (descending)
        method_stats.sort(key=lambda x: x['Mean'], reverse=True)
        
        for stat in method_stats:
            print(f"{stat['Method']:<22}{stat['Mean']:>8.1f}{stat['Std']:>8.1f}"
                  f"{stat['Max']:>6}{stat['Min']:>6}{stat['Success_Rate']:>9.1f}%")
        
        print("\n" + "="*80)
        print("BEST PERFORMING METHODS (by number of images where method gave highest count)")
        print("="*80)
        
        # Count wins for each method using groupby approach to handle duplicates
        method_wins = {method: 0 for method in self.all_methods}
        
        # Group by image name and get best method for each unique image
        unique_images = detailed_df.groupby('image_name')
        
        for image_name, group in unique_images:
            # Get the maximum count for this image
            max_count = group['final_count'].max()
            # Get all methods that achieved this maximum count
            best_methods = group[group['final_count'] == max_count]['method'].tolist()
            
            # Award points (split if tie)
            points_per_method = 1.0 / len(best_methods)
            for method in best_methods:
                method_wins[method] += points_per_method
        
        # Sort and print
        sorted_wins = sorted(method_wins.items(), key=lambda x: x[1], reverse=True)
        
        print(f"{'Rank':<6}{'Method':<22}{'Wins':>8}{'Win %':>8}")
        print("-" * 50)
        
        total_unique_images = len(unique_images)
        for i, (method, wins) in enumerate(sorted_wins, 1):
            win_pct = (wins / total_unique_images) * 100
            print(f"{i:<6}{method:<22}{wins:>8.1f}{win_pct:>7.1f}%")

# Usage
def main():
    counter = EnhancedSheetStackCounter(cluster_max_distance=30)
    path = "data/nowrap_images_center"
    # path =  "data/Ramco_edits/3_nowrap_images_bound_sharp"
    path = "data/nowrap_images_center/sharpened"
    
    
    
    # Analyze all methods for all images and print comprehensive table
    detailed_df, summary_df = counter.analyze_all_methods(path)

if __name__ == "__main__":
    main()
    
# python scripts/6_DensityCVCounter/new.py   
