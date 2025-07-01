import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks

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
        
        original = img.copy()
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply multiple Gaussian blur iterations to reduce noise
        for _ in range(3):
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        # Apply morphological operations to enhance horizontal lines
        # Create horizontal kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        
        # Morphological opening to enhance horizontal lines
        gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        
        return original, gray
    
    def detect_edges(self, gray):
        """
        Detect edges and enhance horizontal features
        """
        # Canny edge detection with adjusted parameters
        edges = cv2.Canny(gray, 50, 150, apertureSize=3, L2gradient=True)
        
        # Create horizontal kernel for morphological operations
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        edges_processed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, horizontal_kernel)
        
        return edges_processed
    
    def analyze_edge_density(self, edges, window_size=10):
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
    
    def find_peaks_in_density(self, edge_density, min_prominence=0.01, min_distance=5):
        """
        Find peaks in edge density that correspond to sheet edges
        """
        # Find peaks in the edge density
        peaks, properties = find_peaks(
            edge_density, 
            prominence=min_prominence,
            distance=min_distance,
            height=np.mean(edge_density)  # Only consider peaks above average density
        )
        
        return peaks
    
    def count_sheets(self, img_path):
        """
        Main function to count sheets using density peaks method
        """
        try:
            # Preprocess image
            original, gray = self.preprocess_image(img_path)
            
            # Detect edges
            edges = self.detect_edges(gray)
            
            # Analyze edge density
            edge_density = self.analyze_edge_density(edges, window_size=10)
            
            # Find density peaks
            density_peaks = self.find_peaks_in_density(edge_density)
            
            if self.debug_mode:
                self.visualize_results(original, edges, edge_density, density_peaks)
            
            # Print results
            print(f"=== SHEET COUNTING RESULTS ===")
            print(f"Density peaks method: {len(density_peaks)} sheets detected")
            
            return len(density_peaks), original
            
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
            cv2.line(result_img, (0, int(y_pos)), (result_img.shape[1], int(y_pos)), (0, 255, 0), 3)
        
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
    
    # Replace with your image path
    image_path = "3 nowrap_images_bound_sharp/image7.png"  # Update this with your actual image path
    
    try:
        sheet_count, processed_image = counter.count_sheets(image_path)
        print(f"\nFinal Result: {sheet_count} sheets detected in the stack")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please make sure to update the image_path variable with your actual image file path")

if __name__ == "__main__":
    main()