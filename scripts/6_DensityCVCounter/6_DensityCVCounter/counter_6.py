import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
import random
visualize_results = 0


class SheetStackCounter:
    def __init__(self):
        self.debug_mode = True
        
    def preprocess_image(self, img_path):
        """
        Preprocess the image to enhance edge detection for stacked sheets
        """
        # Read image
        img = cv2.imread(img_path)
        
        if img is None:
            raise ValueError(f"Could not load image from {img_path}")
        kernel = np.array([
            [ 0, -1,  0],
            [-1,  5, -1],
            [ 0, -1,  0]
        ])
        # kernel = np.array([
        #     [ 0, -1,  0],
        #     [0,  3, 0],
        #     [ 0, -1,  0]
        # ])
        sharpened = img
        sharpened = cv2.filter2D(sharpened, -1, kernel)
        original = img.copy()
        
        # Convert to grayscale
        gray = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)
        
        # Apply multiple Gaussian blur iterations to reduce noise
        for _ in range(2):
            gray = cv2.GaussianBlur(gray, (11, 3), 0)
        
        # Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(15, 8))
        gray = clahe.apply(gray)
        # gray = cv2.GaussianBlur(gray, (3, 1), 0)
        
        # Apply morphological operations to enhance horizontal lines
        # Create horizontal kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
        
        # Morphological opening to enhance horizontal lines
        gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        
        return original, gray
    
    def detect_edges(self, gray):
        """
        Detect edges and enhance horizontal features
        """
        # Canny edge detection with adjusted parameters
        edges_processed = cv2.Canny(gray, 50, 150, apertureSize=3, L2gradient=True)
        # 2. Compute vertical Sobel gradient (Y-direction)
        sob_y = cv2.Sobel(gray, cv2.CV_64F, dx=0, dy=1, ksize=3)


        # 3. Keep only NEGATIVE gradients (light → dark from top to bottom)
        # light_to_dark = np.where(sob_y < -150, -sob_y, 0)  # make negative values positive, set rest to 0
        light_to_dark = np.where(sob_y > 0, sob_y, 0)  # make negative values positive, set rest to 0
        

        # 4. Normalize to 0–255 for visualization
        norm_y = np.uint8((light_to_dark / (light_to_dark.max() + 1e-6)) * 255)

        # Keep only strong white gradients above a threshold
        threshold_value = 100  # Adjust this based on how strong the lines are
        _, edges_processed = cv2.threshold(norm_y, threshold_value, 255, cv2.THRESH_BINARY)
        
            # Step 6: Flip the edge image horizontally
        flipped_edges = cv2.flip(edges_processed, 1)  # flipCode=1 means horizontal flip

        # Step 7: Superimpose (bitwise OR)
        edges_processed = cv2.bitwise_or(edges_processed, flipped_edges)
        
        edges_processed = cv2.Canny(edges_processed, 50, 100, apertureSize=3, L2gradient=True)
        
        
        
        # h, w = edges_processed.shape[:2]
        # edges_processed = edges_processed[:, :w // 2]
        
        
        
        
        # Create horizontal kernel for morphological operations
        # horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        # edges_processed = cv2.morphologyEx(edges_processed, cv2.MORPH_CLOSE, horizontal_kernel)
        
        return edges_processed
    
    def analyze_edge_density(self, edges, window_size=5):
        """
        Analyze horizontal edge density across the image height
        """
        height, width = edges.shape
        edge_density = []
        
        # Calculate edge density for each row using a sliding window
        for y in range(height):
            window_start = max(0, y - window_size // 2)
            window_end = min(height, y + window_size // 2 + 1)
            
            window_edges = edges[window_start:window_end, :]
            density = np.sum(window_edges) / (window_edges.shape[0] * window_edges.shape[1])
            edge_density.append(density)
        
        return np.array(edge_density)
    
    def find_peaks_in_density(self, edge_density, min_prominence=0.01, min_distance=5, min_valley_drop_a=10, abs_min_distance=4):
        """
        Find peaks in edge density that correspond to sheet edges.
        Allows close peaks if there's a significant valley between them.
        Further filters out isolated peaks that don't belong to a cluster of ≥4 within ±25 pixels.
        """
        # Step 1: Get initial candidates (no distance filter)
        all_peaks, properties = find_peaks(
            edge_density,
            prominence=min_prominence,
            height=int(((np.max(edge_density) + np.min(edge_density)) / 2) * 3 / 6)
        )

        filtered_peaks = []

        for i, p in enumerate(all_peaks):
            if not filtered_peaks:
                filtered_peaks.append(p)
            else:
                prev_p = filtered_peaks[-1]
                if abs(p - prev_p) <= min_distance:
                    # Check valley depth between peaks
                    valley_start = min(prev_p, p)
                    valley_end = max(prev_p, p)
                    valley_min = np.min(edge_density[valley_start:valley_end + 1])
                    valley_drop = max(edge_density[prev_p], edge_density[p]) - valley_min
                    min_valley_drop_b = valley_drop/2
                    valley_drop_b = min(edge_density[prev_p], edge_density[p]) - valley_min
                    
                    if abs(p - prev_p) <= abs_min_distance:
                        pass

                    elif valley_drop >= min_valley_drop_a and valley_drop_b >= min_valley_drop_b:
                        filtered_peaks.append(p)  # Keep both close peaks
                    else:
                        pass
                        # Replace weaker with stronger
                        if edge_density[p] > edge_density[prev_p]:
                            # filtered_peaks[-1] = p
                            edge_density[prev_p] = edge_density[p]
                else:
                    filtered_peaks.append(p)

        # Step 2: Filter out peaks that don't have 3+ neighbors in ±25 pixel range
        filtered_peaks = np.array(filtered_peaks)
        final_peaks = []

        for p in filtered_peaks:
            b4 = np.sum((filtered_peaks < p) & (np.abs(filtered_peaks - p) <= 40))
            a4 = np.sum((filtered_peaks > p) & (np.abs(filtered_peaks - p) <= 40))
            if b4 >= 3 or a4 >= 3:
                final_peaks.append(p)

        return np.array(final_peaks)

    
    def count_sheets(self, img_path):
        """
        Main function to count sheets using density peaks method
        """
        tests = 10
        try:
            # Preprocess image
            original, original_gray = self.preprocess_image(img_path)
            
            if original is None:
                raise FileNotFoundError(f"Could not load image: {img_path}")

            height, width = original.shape[:2]
            crop_width = (width*2) // 3
            margin = width // 10
            peak_counts = []
            step =  int((width - margin - margin - crop_width)/(tests))
            start = margin
            starts = []
            for _ in range(tests):
                starts.append(start)
                start += step
                
            # print(starts)
            

            for index in range(tests):
                x_start = starts[index]
                gray = original_gray[:, x_start:x_start + crop_width]
                crop = original[:, x_start:x_start + crop_width]
                
                edges = self.detect_edges(gray)
                edge_density = self.analyze_edge_density(edges, window_size=10)
                density_peaks = self.find_peaks_in_density(edge_density)

                peak_counts.append(len(density_peaks))
                if(index<=tests*3/4 and index>=tests*1/4):
                    peak_counts.append(len(density_peaks))

                if self.debug_mode and visualize_results:
                    self.visualize_results(crop, edges, edge_density, density_peaks)

            peak_counts.sort
            peak_counts.pop
            peak_counts.reverse
            peak_counts.pop
            avg_peak_count = int(round(sum(peak_counts) / len(peak_counts), 0))
            return avg_peak_count, original
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return 0, None
    
    def visualize_results(self, original, edges, edge_density, density_peaks):
        """
        Visualize the density peaks analysis results
        """
        # Create result image with detected sheet lines
        result_img = original.copy()
        
        # Draw detected sheet lines in green
        for y_pos in density_peaks:
            cv2.line(result_img, (0, int(y_pos)), (result_img.shape[1], int(y_pos)), (0, 255, 0), 1)
        
        # Create visualization
        plt.figure(figsize=(20, 10))
        
        # Original image
        plt.subplot(2, 3, 1)
        plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')
        
        # Processed edges
        plt.subplot(2, 3, 2)
        plt.imshow(edges, cmap='gray')
        plt.title('Processed Edges')
        plt.axis('off')
        
        # Result with detected sheets
        plt.subplot(2, 3, 3)
        plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        plt.title(f'Detected Sheets: {len(density_peaks)}')
        plt.axis('off')
        
        # Edge density profile
        plt.subplot(2, 3, 4)
        plt.plot(edge_density, range(len(edge_density)))
        plt.gca().invert_yaxis()
        plt.xlabel('Edge Density')
        plt.ylabel('Y Position (pixels)')
        plt.title('Edge Density Profile')
        plt.grid(True, alpha=0.3)
        
        # Edge density with peaks marked
        plt.subplot(2, 3, 5)
        plt.plot(edge_density, range(len(edge_density)), 'b-', alpha=0.7, label='Edge Density')
        for peak in density_peaks:
            plt.axhline(y=peak, color='red', linestyle='--', alpha=0.7)
        plt.gca().invert_yaxis()
        plt.xlabel('Edge Density')
        plt.ylabel('Y Position (pixels)')
        plt.title('Detected Peaks (Red Lines)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Peak positions bar chart
        plt.subplot(2, 3, 6)
        plt.bar(['Detected Sheets'], [len(density_peaks)], color='green', alpha=0.7)
        plt.ylabel('Sheet Count')
        plt.title('Final Count')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add count label on bar
        plt.text(0, len(density_peaks) + 0.1, str(len(density_peaks)), 
                ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.show()

def main():
    counter = SheetStackCounter()
    
    # Folder containing images
    # folder_path = "data/Ramco_edits/final_aligned"
    folder_path = "data/png/final_aligned"
    
    # folder_path = "data/Ramco_edits/3_nowrap_images_bound_sharp"
    # folder_path = "data/raw/nowrap_images"
    # folder_path = "data/raw/nowrap_images/test"
    
    # folder_path = "data/nowrap_images_center/sharpened"
    # folder_path = "data/nowrap_images_center"
    
    # folder_path = "data/raw/wrap_images"
    
    
    

    # Supported image formats
    valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    image_files = [f for f in os.listdir(folder_path) if os.path.splitext(f)[1].lower() in valid_exts]
    sorted(image_files)

    for fname in sorted(image_files):
        image_path = os.path.join(folder_path, fname)
        # print(f"\nProcessing {fname}...")

        try:
            sheet_count, processed_image = counter.count_sheets(image_path)
            print(f"{fname}... → {sheet_count} sheets")

        except Exception as e:
            print(f"✖ Error processing {fname}: {e}")

    print("\n✅ Batch processing complete!")

if __name__ == "__main__":
    main()
    
    
# python scripts/6_DensityCVCounter/counter_4.py   

# up - 1,3,5,7,10