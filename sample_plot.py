# import libraries
import numpy as np
from numpy.typing import NDArray
from matplotlib import pyplot as plt
import matplotlib
from scipy.spatial.transform import Rotation
import glob
import os



class Vizualisation:
    def __init__(self,):
        
        self.fig = None
        self.axes = None

    def draw_corners(self, frame, corners2d, centroid, bounding_box ):
        x,y,w,h = int(bounding_box[0]), int(bounding_box[1]), int(bounding_box[2]), int(bounding_box[3])
        cx , cy = int(centroid[0]), int(centroid[1])
        frame_ = frame.copy() 
        # Draw bounding box on frame
        cv2.rectangle(frame_, (x,y), (x+w,y+h), (0, 255, 0), 2)
        # Draw contours on frame
        cv2.drawContours(frame_, [corners2d], -1, (255, 0, 0 ), 3)
        # Draw the centroid on frame
        cv2.circle(frame_, (cx, cy), 7, (255, 255, 0), -1)  # Yellow circle for centroid
        cv2.putText(frame_, "C", (cx - 5, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        # Draw the corners on frame and scaled bounding box
        corner_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),(0,255, 255),(0, 255, 255 )]  # Different color for each corner
        corner_labels = ["1", "2", "3", "4", "5", "6"]

        for j, (corner, color, label) in enumerate(zip(corners2d, corner_colors, corner_labels)):
            cx, cy = corner
            cv2.circle(frame_, (cx, cy), 5, color, -1)
            cv2.putText(frame_, label, (cx-10, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return frame_
    
    def draw_bounding_box(self, frame , corners2d , centroid, bounding_box, projected_pts):
        # Draw scaled bounding box 
        x,y,w,h = int(bounding_box[0]), int(bounding_box[1]), int(bounding_box[2]), int(bounding_box[3])
        cx , cy = int(centroid[0]), int(centroid[1])

        padding = 20
        x_padded = max(0, x - padding)
        y_padded = max(0, y - padding)
        w_padded = min(frame.shape[1] - x_padded, w + 2*padding)
        h_padded = min(frame.shape[0] - y_padded, h + 2*padding)

        bounding_box = frame[y_padded:y_padded+h_padded, x_padded:x_padded+w_padded]
        scaled_bounding_box = cv2.resize(bounding_box, (w_padded*5, h_padded*5), interpolation=cv2.INTER_CUBIC)
        

        # Draw rectangle on the scaled bounding box 
        cv2.rectangle(scaled_bounding_box, 
                    (padding*5, padding*5), 
                    ((x_padded+w_padded-x_padded-padding)*5, (y_padded+h_padded-y_padded-padding)*5), 
                    (0, 255, 0), 2)
        #cv2.putText(scaled_bounding_box, "Bounding Box", (padding*5, padding*5-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


        # Draw centroid on scaled bounding box
        rel_centroid_x = (cx - x_padded) * 5
        rel_centroid_y = (cy - y_padded) * 5
        cv2.circle(scaled_bounding_box, (rel_centroid_x, rel_centroid_y), 5, (255, 255, 0), -1)   
        cv2.putText(scaled_bounding_box, "C", (rel_centroid_x- 15, rel_centroid_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)

        # Draw the corners on frame and scaled bounding box
        corner_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),(0,255, 255),(0, 255, 255 )]  # Different color for each corner
        corner_labels = ["1", "2", "3", "4", "5", "6"]

        for j, (corner, color, label) in enumerate(zip(corners2d, corner_colors, corner_labels)):
            cx, cy = corner
            rel_cx = (cx - x_padded) * 5
            rel_cy = (cy - y_padded) * 5
            cv2.circle(scaled_bounding_box, (rel_cx, rel_cy), 5, color, -1)
            cv2.putText(scaled_bounding_box, label, (rel_cx-15, rel_cy-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2) 



        # Draw reprojected points on frame and scaled bounding box
        for pts in range(projected_pts.shape[1]):
            px , py = int(projected_pts[0, pts]), int(projected_pts[1, pts])
            rel_px = (px - x_padded) * 5
            rel_py = (py - y_padded) * 5
            cv2.circle(scaled_bounding_box, (rel_px, rel_py), 3, (0,0,0), -1)
            cv2.putText(scaled_bounding_box, "rep", (rel_px+15, rel_py+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 0, 0) , 2) 

        return scaled_bounding_box
    
    def draw_closup(self, frame, bounding_box):
        # Draw scaled bounding box 
        x,y,w,h = int(bounding_box[0]), int(bounding_box[1]), int(bounding_box[2]), int(bounding_box[3])

        padding = 20
        x_padded = max(0, x - padding)
        y_padded = max(0, y - padding)
        w_padded = min(frame.shape[1] - x_padded, w + 2*padding)
        h_padded = min(frame.shape[0] - y_padded, h + 2*padding)

        bounding_box = frame[y_padded:y_padded+h_padded, x_padded:x_padded+w_padded]
        scaled_bounding_box = cv2.resize(bounding_box, (w_padded*5, h_padded*5), interpolation=cv2.INTER_CUBIC)
        return scaled_bounding_box
    
    def plotting(self, idx, image, corners2d, centroid2d, projected_pts, bounding_box, centroid3d_history,normal_history, flag):

        # Create or reuse figure
        if self.fig is None or self.axes is None:
            # Close any existing figures to prevent gray overlays
            plt.close('all')
            # Create a new figure with subplots
            self.fig = plt.figure(figsize=(16, 8))
            self.fig.suptitle("Screencast", fontsize=16)
        
        # Create or reuse subplots
        if self.axes is None:
            ax1 = self.fig.add_subplot(131, projection='3d')   # First 3D plot
            ax2 = self.fig.add_subplot(132)                    # 2D image plot
            ax3 = self.fig.add_subplot(133)                    # 2D boundingbox plot
            self.axes = (ax1, ax2, ax3)
        else:
            ax1, ax2, ax3 = self.axes
            # Clear existing plots
            ax1.clear()
            ax2.clear()
            ax3.clear()

        # Plot centroid trajectory
        ax1.scatter3D(centroid3d_history[0, :], centroid3d_history[1, :], centroid3d_history[2, :], c='r', s=1)

        # Plot quaternion vectors in 3D
        scale_factor = 100
        ax1.quiver(centroid3d_history[0, :], centroid3d_history[1, :],centroid3d_history[2, :],           
            scale_factor*normal_history[0, :], scale_factor*normal_history[1, :], scale_factor* normal_history[2, :],  
            color='black', 
            label='Quaternion Direction',
            length=100,  
            normalize=True)
        
        ax1.view_init(elev=-90, azim=-90)
        
        # Add a camera
        # Calculate the corners of the frustum at z=scale
        # based on the field of view angles
        scale = 76 
        fov_x = 23
        fov_y = 45
        color = 'k'
        half_width = scale * np.tan(np.radians(fov_x / 2))
        half_height = scale * np.tan(np.radians(fov_y / 2))
        
        # Define the camera center (origin)
        center = np.array([0, 0, 0])
        
        # Define the frustum corners at z=scale
        corners = np.array([
            [half_width, half_height, scale],     # top-right
            [half_width, -half_height, scale],    # bottom-right
            [-half_width, -half_height, scale],   # bottom-left
            [-half_width, half_height, scale]     # top-left
        ])
        
        # Draw the frustum lines from center to corners
        for corner in corners:
            ax1.plot3D([center[0], corner[0]], 
                    [center[1], corner[1]], 
                    [center[2], corner[2]], 
                    f'{color}-')

        # Draw the rectangle at z=scale
        ax1.plot3D([corners[0][0], corners[1][0], corners[2][0], corners[3][0], corners[0][0]],
                [corners[0][1], corners[1][1], corners[2][1], corners[3][1], corners[0][1]],
                [corners[0][2], corners[1][2], corners[2][2], corners[3][2], corners[0][2]],
                f'{color}-')
        
        # Draw coordinate axes to represent camera orientation
        axis_length = scale * 0.5
        # X-axis (red)
        ax1.plot3D([0, axis_length], [0, 0], [0, 0] , 'r-')
        # Y-axis (green)
        ax1.plot3D([0, 0], [0, axis_length], [0, 0], 'g-')
        # Z-axis (blue) - pointing in the viewing direction
        ax1.plot3D([0, 0], [0, 0], [0, axis_length], 'b-')
        
        # Optional: Add text labels for the axes
        ax1.text(axis_length*1.1, 0, 0, "X", color='red')
        ax1.text(0, axis_length*1.1, 0, "Y", color='green')
        ax1.text(0, 0, axis_length*1.1, "Z", color='blue')

        # Set labels and title
        ax1.set_xlabel('X (mm)')
        ax1.set_ylabel('Y (mm)')
        ax1.set_zlabel('Z (mm)')
        ax1.set_title('Stickynote Trajectory')

        # To ensure truly equal scale on all axes, use equal scaling for each:
        x_limits = ax1.get_xlim3d()
        y_limits = ax1.get_ylim3d()
        z_limits = ax1.get_zlim3d()

        # Get the range of each axis
        x_range = abs(x_limits[1] - x_limits[0])
        y_range = abs(y_limits[1] - y_limits[0])
        z_range = abs(z_limits[1] - z_limits[0])

        # Find the largest range to ensure all axes have equal scaling
        max_range = max(x_range, y_range, z_range)

        # Set the limits for each axis based on their centers and the max range
        x_center = np.mean(x_limits)
        y_center = np.mean(y_limits)
        z_center = np.mean(z_limits)

        ax1.set_xlim([x_center - max_range/2, x_center + max_range/2])
        ax1.set_ylim([y_center - max_range/2, y_center + max_range/2])
        ax1.set_zlim([z_center - max_range/2, z_center + max_range/2])

        if flag[1,idx]:
            frame = self.draw_corners(frame = image, corners2d = corners2d, centroid=centroid2d, bounding_box=bounding_box[:,idx])
        else: frame = image

        ax2.imshow(frame)
        ax2.set_title(f'Frame {idx}')
        ax2.axis('off')

        if flag[1,idx]:
            bounding_box_plot = self.draw_bounding_box(frame = image, corners2d = corners2d, centroid=centroid2d, bounding_box=bounding_box[:,idx], projected_pts = projected_pts)
        else: 
            indices = np.where(flag[1,:] == 1)[0]
            # Get the last such index (if any exist)
            if indices.size > 0:
                last_idx = indices[-1]
            else:
                last_idx = idx
            bounding_box_plot = self.draw_closup(frame=frame, bounding_box=bounding_box[:,last_idx])

        ax3.imshow(bounding_box_plot)
        ax3.set_title(f'Bounding Box {idx}')
        ax3.axis('off')

        plt.tight_layout()
        # Show the plot (non-blocking)
        plt.draw() 
        plt.pause(0.0001)
        return self.fig, self.axes

    def create_plots(self, idx, history, str):
        str = str.replace('Data/',"")
        str = str.replace('/',"_")
        str = str.replace('.mov',"")
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle( f'{str}', fontsize=16)
        ax1 = fig.add_subplot(231)    
        ax2 = fig.add_subplot(233)
        ax3 = fig.add_subplot(234)  
        ax4 = fig.add_subplot(232) 
        ax5 = fig.add_subplot(236)  
        ax6 = fig.add_subplot(235)  
               

        # Plot X, Y, Z coordinates of the centroid
        ax1.scatter(range(len(history.centroid3d[0, :idx])), history.centroid3d[0, :idx],s=1, color='red', label='X')
        ax1.scatter(range(len(history.centroid3d[1, :idx])), history.centroid3d[1, :idx],s=1,  color='green', label='Y')
        ax1.scatter(range(len(history.centroid3d[2, :idx])), history.centroid3d[2, :idx], s=1, color='blue', label='Z')

        # Add legend
        ax1.legend()
        # Optional: add labels and title for clarity
        ax1.set_xlabel('Frame')
        ax1.set_ylabel('Position [mm]')
        ax1.set_title('3D Centroid Position')  



        ax2.plot(history.rep_error[:idx], label='Reprojection Error')    
        # Add legend
        ax2.legend()
        # Optional: add labels and title for clarity
        ax2.set_xlabel('Frame')
        ax2.set_ylabel('Reprojection Error [pi^2]')
        ax2.set_title('Reprojection Error')  

        # Plot X, Y, Z coordinates of the centroid
        ax3.scatter(range(len(history.centroid3d_raw[0, :idx])), history.centroid3d_raw[0, :idx],s=1, color='red', label='X')
        ax3.scatter(range(len(history.centroid3d_raw[0, :idx])), history.centroid3d_raw[1, :idx],s=1,  color='green', label='Y')
        ax3.scatter(range(len(history.centroid3d_raw[0, :idx])), history.centroid3d_raw[2, :idx], s=1, color='blue', label='Z')

        # Add legend
        ax3.legend()
        # Optional: add labels and title  for clarity    
        ax3.set_xlabel('Frame')
        ax3.set_ylabel('Position [mm]')
        ax3.set_title('Raw 3D Centroid Position') 

        # pitch = np.arcsin(-history.n[0, :idx])  # Rotation around Y axis
        # yaw = np.arctan2(history.n[1, :idx], history.n[2, :idx])  # Rotation around Z axis
        # yaw = yaw % (2*np.pi) 
        # Plot X, Y, Z coordinates of the centroid
        ax4.scatter(range(len(history.n[0, :idx])), history.n[0, :idx],s=1, color='red', label='n_x')
        ax4.scatter(range(len(history.n[1, :idx])), history.n[1, :idx],s=1, color='green', label='n_y')
        ax4.scatter(range(len(history.n[2, :idx])), history.n[2, :idx],s=1, color='blue', label='n_z')

        # Add legend
        ax4.legend()
        # Optional: add labels and title for clarity
        ax4.set_xlabel('Frame')
        ax4.set_ylabel('Euler Angles [rad]')
        ax4.set_title('Normal Vector')

        mask_minus1 = history.flag[2,:idx] == -1
        ax5.scatter(np.where(mask_minus1)[0], history.flag[2,:idx][mask_minus1], 
                    s=1, color='blue', label='Depth approx.')

        # Flag = 0 (previous value) in red
        mask_0 = history.flag[2,:idx] == 0
        ax5.scatter(np.where(mask_0)[0], history.flag[2,:idx][mask_0], 
                    s=1, color='red', label='Prev. value')

        # Flag = 1 (PnP) in green
        mask_1 = history.flag[2,:idx] == 1
        ax5.scatter(np.where(mask_1)[0], history.flag[2,:idx][mask_1], 
                    s=1, color='green', label='PnP')

        # Add legend
        ax5.legend()
        # pitch_r = np.arcsin(-history.n_raw[0, :idx])  # Rotation around Y axis
        # yaw_r = np.arctan2(history.n_raw[1, :idx], history.n[2, :idx])  # Rotation around Z axis
        # yaw_r = yaw_r % (2*np.pi) 
        # Plot X, Y, Z coordinates of the centroid
        ax6.scatter(range(len(history.n_raw[1, :idx])), history.n_raw[1, :idx],s=1, color='green', label='n_y')
        ax6.scatter(range(len(history.n_raw[2, :idx])), history.n_raw[2, :idx],s=1, color='blue', label='n_z')
        ax6.scatter(range(len(history.n_raw[0, :idx])), history.n_raw[0, :idx],s=1, color='red', label='n_x')

        # Add legend
        ax6.legend()
        # Optional: add labels and title for clarity
        ax6.set_xlabel('Frame')
        ax6.set_ylabel('Euler Angles [rad]')
        ax6.set_title('Raw Normal Vector')

        fig.tight_layout()

        plt.savefig(f'{str}.pdf')
    
    def draw_trajectories(self, idx, video, history, colour):
        # Create figure only if it doesn't exist yet
        if self.fig is None:
            # Close any existing figures to prevent gray overlays
            print("Creating new figure")
            plt.close('all')
            # Create a new figure with 3 subplots
            self.fig = plt.figure(figsize=(18, 6))
            self.fig.suptitle("Trajectories from Multiple Viewpoints", fontsize=16)
            
            # Create three 3D axes for different viewpoints
            self.axes1 = self.fig.add_subplot(131, projection='3d')
            self.axes2 = self.fig.add_subplot(132, projection='3d')
            self.axes3 = self.fig.add_subplot(133, projection='3d')
            
            # Store axes in a list for easier iteration
            self.axes_list = [self.axes1, self.axes2, self.axes3]
            
            # Set different view angles for each subplot
            self.axes1.view_init(elev=-90, azim=-90)  # Top view (XY plane)
            self.axes2.view_init(elev=0, azim=-90)    # Front view (XZ plane)
            self.axes3.view_init(elev=0, azim=0, roll= -90)  # Side view (Z plane)
            
            # Titles for each viewpoint
            self.axes1.set_title("Front View (XY Plane)")
            self.axes2.set_title("Top View (XZ Plane)")
            self.axes3.set_title("Side View (YZ Plane)")
            
            # Initialize storage for legend items
            self.all_handles = []
            self.all_labels = []
            
            # Initialize separate min/max trackers for each view
            self.view1_min = {'x': float('inf'), 'y': float('inf')}
            self.view1_max = {'x': float('-inf'), 'y': float('-inf')}
            
            self.view2_min = {'x': float('inf'), 'z': float('inf')}
            self.view2_max = {'x': float('-inf'), 'z': float('-inf')}
            
            self.view3_min = {'z': float('inf'), 'y': float('inf')}
            self.view3_max = {'z': float('-inf'), 'y': float('-inf')}
        
        # Add this trajectory data to the plots
        scatter = None
        
        # View 1: XY plane (Front view)
        scatter = self.axes1.scatter3D(
            history.centroid3d[0, :idx],
            history.centroid3d[1, :idx],
            history.centroid3d[2, :idx],
            color=colour, s=1, label=f'Trajectory {video+1:2d}'
        )
        self.axes1.set_xlabel('X (mm)')
        self.axes1.set_ylabel('Y (mm)')
        self.axes1.set_zlabel('Z (mm)')
        self.axes1.set_zticklabels([])
        
        # Update min/max for view 1
        self.view1_min['x'] = min(self.view1_min['x'], min(history.centroid3d[0, :idx]))
        self.view1_max['x'] = max(self.view1_max['x'], max(history.centroid3d[0, :idx]))
        self.view1_min['y'] = min(self.view1_min['y'], min(history.centroid3d[1, :idx]))
        self.view1_max['y'] = max(self.view1_max['y'], max(history.centroid3d[1, :idx]))
        
        # View 2: XZ plane (Top view)
        self.axes2.scatter3D(
            history.centroid3d[0, :idx],
            history.centroid3d[1, :idx],
            history.centroid3d[2, :idx],
            color=colour, s=1, label=f'Trajectory {video+1:2d}'
        )
        self.axes2.set_xlabel('X (mm)')
        self.axes2.set_ylabel('Y (mm)')
        self.axes2.set_zlabel('Z (mm)')
        self.axes2.set_yticklabels([])
        
        # Update min/max for view 2
        self.view2_min['x'] = min(self.view2_min['x'], min(history.centroid3d[0, :idx]))
        self.view2_max['x'] = max(self.view2_max['x'], max(history.centroid3d[0, :idx]))
        self.view2_min['z'] = min(self.view2_min['z'], min(history.centroid3d[2, :idx]))
        self.view2_max['z'] = max(self.view2_max['z'], max(history.centroid3d[2, :idx]))
        
        # View 3: XY plane (Side view)
        scatter = self.axes3.scatter3D(
            history.centroid3d[0, :idx],
            history.centroid3d[1, :idx],
            history.centroid3d[2, :idx],
            color=colour, s=1, label=f'Trajectory {video+1:2d}'
        )
        self.axes3.set_xlabel('X (mm)')
        self.axes3.set_ylabel('Y (mm)')
        self.axes3.set_zlabel('Z (mm)')
        self.axes3.set_xticklabels([])
        
        # Update min/max for view 1
        self.view3_min['z'] = min(self.view3_min['z'], min(history.centroid3d[2, :idx]))
        self.view3_max['z'] = max(self.view3_max['z'], max(history.centroid3d[2, :idx]))
        self.view3_min['y'] = min(self.view3_min['y'], min(history.centroid3d[1, :idx]))
        self.view3_max['y'] = max(self.view3_max['y'], max(history.centroid3d[1, :idx]))
        
        # Store the handle and label for the legend (only once per trajectory)
        if video >= len(self.all_handles):
            self.all_handles.append(scatter)
            self.all_labels.append(f'Trajectory {video+1:2d}')
        
        # Apply margin factor for better visualization
        margin_factor = 0.15  # 15% margin
        
        # Set limits for view 1 (XY plane)
        x1_range = self.view1_max['x'] - self.view1_min['x']
        y1_range = self.view1_max['y'] - self.view1_min['y']
        max1_range = max(x1_range, y1_range) * (1 + margin_factor)
        x1_center = (self.view1_max['x'] + self.view1_min['x']) / 2
        y1_center = (self.view1_max['y'] + self.view1_min['y']) / 2

        # Round limits to nearest 100
        x1_min = np.floor((x1_center - max1_range/2) / 100) * 100
        x1_max = np.ceil((x1_center + max1_range/2) / 100) * 100
        y1_min = np.floor((y1_center - max1_range/2) / 100) * 100
        y1_max = np.ceil((y1_center + max1_range/2) / 100) * 100

        self.axes1.set_xlim([x1_min, x1_max])
        self.axes1.set_ylim([y1_min, y1_max])
        self.axes1.set_xticks(np.linspace(x1_min, x1_max, 5))
        self.axes1.set_yticks(np.linspace(y1_min, y1_max, 5))

        # Set limits for view 2 (XZ plane)
        x2_range = self.view2_max['x'] - self.view2_min['x']
        z2_range = self.view2_max['z'] - self.view2_min['z']
        max2_range = max(x2_range, z2_range) * (1 + margin_factor)
        x2_center = (self.view2_max['x'] + self.view2_min['x']) / 2
        z2_center = (self.view2_max['z'] + self.view2_min['z']) / 2

        # Round limits to nearest 100
        x2_min = np.floor((x2_center - max2_range/2) / 100) * 100
        x2_max = np.ceil((x2_center + max2_range/2) / 100) * 100
        z2_min = np.floor((z2_center - max2_range/2) / 100) * 100
        z2_max = np.ceil((z2_center + max2_range/2) / 100) * 100

        self.axes2.set_xlim([x2_min, x2_max])
        self.axes2.set_ylim([z2_min, z2_max])  # Assuming this should be set_zlim?
        self.axes2.set_xticks(np.linspace(x2_min, x2_max, 5))
        self.axes2.set_zticks(np.linspace(z2_min, z2_max, 5))  # Assuming this should be set_zticks?

        # Set limits for view 3 (YZ plane)
        z3_range = self.view3_max['z'] - self.view3_min['z']
        y3_range = self.view3_max['y'] - self.view3_min['y']
        max3_range = max(z3_range, y3_range) * (1 + margin_factor)
        z3_center = (self.view3_max['z'] + self.view3_min['z']) / 2
        y3_center = (self.view3_max['y'] + self.view3_min['y']) / 2

        # Round limits to nearest 100
        z3_min = np.floor((z3_center - max3_range/2) / 100) * 100
        z3_max = np.ceil((z3_center + max3_range/2) / 100) * 100
        y3_min = np.floor((y3_center - max3_range/2) / 100) * 100
        y3_max = np.ceil((y3_center + max3_range/2) / 100) * 100

        self.axes3.set_zlim([z3_min, z3_max])
        self.axes3.set_ylim([y3_min, y3_max])
        self.axes3.set_yticks(np.linspace(y3_min, y3_max, 5))
        self.axes3.set_zticks(np.linspace(z3_min, z3_max, 5))
        #Create a single unified legend on the right
        # if hasattr(self, 'legend') and self.legend:
        #     self.legend.remove()
        # self.legend = self.fig.legend(self.all_handles, self.all_labels, 
        #                             loc='center right', bbox_to_anchor=(1.0, 0.5))
        
        #plt.tight_layout(pad=1.0)
        
        return self.fig

    def draw_samples(self, title, idx, trajectory, colour):
        # Create figure only if it doesn't exist yet
        if self.fig is None:
            # Close any existing figures to prevent gray overlays
            print("Creating new figure")
            plt.close('all')

            plt.rcParams['figure.figsize'] = (10.0, 7.0) # set default size of plots
            font = {'family' : 'sans',
                    'weight' : 'normal',
                    'size'   : 16}
            matplotlib.rc('font', **font)
            # Create a new figure with 3 subplots
            self.fig = plt.figure(figsize=(18, 6))
            self.fig.suptitle(title, fontsize=16)
            
            # Create three 2D axes for different projections
            self.axes1 = self.fig.add_subplot(131)  # XY projection (Front view)
            self.axes2 = self.fig.add_subplot(132)  # XZ projection (Top view)
            self.axes3 = self.fig.add_subplot(133)  # YZ projection (Side view)
            
            # Store axes in a list for easier iteration
            self.axes_list = [self.axes1, self.axes2, self.axes3]
            
            # Add grid to all axes
            for ax in self.axes_list:
                ax.grid(True, linestyle='--', alpha=0.7)
            
            # Titles for each projection
            self.axes1.set_title("Front View (XY Plane)")
            self.axes2.set_title("Top View (XZ Plane)")
            self.axes3.set_title("Side View (YZ Plane)")
            
            # Initialize storage for legend items
            self.all_handles = []
            self.all_labels = []
            
            # Initialize separate min/max trackers for each view
            self.view1_min = {'x': float('inf'), 'y': float('inf')}
            self.view1_max = {'x': float('-inf'), 'y': float('-inf')}
            
            self.view2_min = {'x': float('inf'), 'z': float('inf')}
            self.view2_max = {'x': float('-inf'), 'z': float('-inf')}
            
            self.view3_min = {'z': float('inf'), 'y': float('inf')}
            self.view3_max = {'z': float('-inf'), 'y': float('-inf')}
        
        # Add this trajectory data to the plots
        line = None
        
        # View 1: XY plane (Front view)
        line1 = self.axes1.plot(
            trajectory[:, 0],
            -trajectory[:, 1],
            color=colour, linewidth=1, marker='.', markersize=1, 
            label=f'Sample Trajectory {idx+1:2d}'
        )
        self.axes1.set_xlabel('X (mm)')
        self.axes1.set_ylabel('Y (mm)')
        
        # Update min/max for view 1
        self.view1_min['x'] = min(self.view1_min['x'], min(trajectory[:, 0]))
        self.view1_max['x'] = max(self.view1_max['x'], max(trajectory[:, 0]))
        self.view1_min['y'] = min(self.view1_min['y'], min(trajectory[:, 1]))
        self.view1_max['y'] = max(self.view1_max['y'], max(trajectory[:, 1]))
        
        # View 2: XZ plane (Top view)
        line2 = self.axes2.plot(
            trajectory[:, 0],
            trajectory[:, 2],
            color=colour, linewidth=1, marker='.', markersize=1,
            label=f'Trajectory {idx+1:2d}'
        )
        self.axes2.set_xlabel('X (mm)')
        self.axes2.set_ylabel('Z (mm)')
        
        # Update min/max for view 2
        self.view2_min['x'] = min(self.view2_min['x'], min(trajectory[:, 0]))
        self.view2_max['x'] = max(self.view2_max['x'], max(trajectory[:, 0]))
        self.view2_min['z'] = min(self.view2_min['z'], min(trajectory[:, 2]))
        self.view2_max['z'] = max(self.view2_max['z'], max(trajectory[:, 2]))
        
        # View 3: YZ plane (Side view)
        line3 = self.axes3.plot(
            trajectory[:, 2],
            -trajectory[:, 1],
            color=colour, linewidth=1, marker='.', markersize=1,
            label=f'Trajectory {idx+1:2d}'
        )
        self.axes3.set_xlabel('Z (mm)')
        self.axes3.set_ylabel('Y (mm)')
        
        # Update min/max for view 3
        self.view3_min['z'] = min(self.view3_min['z'], min(trajectory[:, 2]))
        self.view3_max['z'] = max(self.view3_max['z'], max(trajectory[:, 2]))
        self.view3_min['y'] = min(self.view3_min['y'], min(trajectory[:, 1]))
        self.view3_max['y'] = max(self.view3_max['y'], max(trajectory[:, 1]))
        
        # Store the handle and label for the legend (only once per trajectory)
        if idx >= len(self.all_handles):
            self.all_handles.append(line1[0])
            self.all_labels.append(f'Sample Trajectory {idx+1:2d}')
        
        # Apply margin factor for better visualization
        margin_factor = 0.15  # 15% margin
        
        # Dynamic tick spacing for all axes
        from matplotlib.ticker import MaxNLocator
        for ax in self.axes_list:
            #ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
            #ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            
            # Update grid based on new ticks
            ax.grid(True, linestyle='--', alpha=0.7)
        
        # Set limits for view 1 (XY plane)
        x1_range = self.view1_max['x'] - self.view1_min['x']
        y1_range = self.view1_max['y'] - self.view1_min['y']
        max1_range = max(x1_range, y1_range) * (1 + margin_factor)
        x1_center = (self.view1_max['x'] + self.view1_min['x']) / 2
        y1_center = (self.view1_max['y'] + self.view1_min['y']) / 2

        # Set equal aspect ratio for better visualization
        self.axes1.set_aspect('equal', adjustable='datalim')
        
        # Set limits for view 2 (XZ plane)
        x2_range = self.view2_max['x'] - self.view2_min['x']
        z2_range = self.view2_max['z'] - self.view2_min['z']
        max2_range = max(x2_range, z2_range) * (1 + margin_factor)
        x2_center = (self.view2_max['x'] + self.view2_min['x']) / 2
        z2_center = (self.view2_max['z'] + self.view2_min['z']) / 2

        # Set equal aspect ratio for better visualization
        self.axes2.set_aspect('equal', adjustable='datalim')
        
        # Set limits for view 3 (YZ plane)
        z3_range = self.view3_max['z'] - self.view3_min['z']
        y3_range = self.view3_max['y'] - self.view3_min['y']
        max3_range = max(z3_range, y3_range) * (1 + margin_factor)
        z3_center = (self.view3_max['z'] + self.view3_min['z']) / 2
        y3_center = (self.view3_max['y'] + self.view3_min['y']) / 2

        # Set equal aspect ratio for better visualization
        self.axes3.set_aspect('equal', adjustable='datalim')
        
        # Add a unified legend if there are multiple trajectories
        if len(self.all_handles) > 1:
            if hasattr(self, 'legend') and self.legend:
                try:
                    self.legend.remove()
                except:
                    pass
            # Place the legend outside of the plots
            self.legend = self.fig.legend(
                self.all_handles, 
                self.all_labels, 
                loc='upper center', 
                bbox_to_anchor=(0.5, 0.01),
                ncol=min(5, len(self.all_handles))  # Limit columns for readability
            )
        
        # Apply tight layout for better spacing, with room for the legend
        self.fig.tight_layout(rect=[0, 0.05, 1, 0.95])
        
        return self.fig

def main():
        #str = "./Diffusion/ema_0.9999_250000/u3c_samples_16x300x3.npy"
        #str = "Diffusion/version_mirrored/ema_0.9999_250000/u3c_samples_32x280x3_frustum.npy"
        str = f'/home/rotogni/diffusion-lagr/pos_samples_256x60x3_cond_60_att_04.npy'
        viz = Vizualisation()
        trajectories = np.load(str)
        # 16 x 300 x 3
        str = str.replace('./Diffusion/', '')
        str = str.replace('version_4d/', '')
        str = str.replace('.npy', '')
        str = str.replace('/', '_')
                 
        trajectories_total = np.zeros((256,240,3))
        colours = plt.cm.jet(np.linspace(0, 1, 32))
        
        for i in range(96,128):
            trajectories = np.zeros_like(trajectories)
            for j in range(1,5):  
                
                str = f'/home/rotogni/diffusion-lagr/pos_samples_256x60x3_cond_60_att_{j:02d}.npy'
                trajectories = np.load(str) +  trajectories[i,-1,:] 
               
                trajectories_total[:,(j-1)*60:(j)*60,:] = np.load(str) +  trajectories_total[:,(j-1)*60-1,:].reshape((256,1,3)) 
                colour = colours[i-96]
                viz.draw_samples(idx=i, title=f'trajectories_{str}', trajectory= trajectories[i,:,:], colour = colour)

        pos_filename = '/home/rotogni/diffusion-lagr/pos_samples_256x240x3_cond_60_att'
        #np.save(pos_filename, trajectories_total)
        plt.savefig(f'{str}.pdf')
        plt.show()

    
    
if __name__ == "__main__":
    main()