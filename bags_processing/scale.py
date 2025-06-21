import cv2
import numpy as np

image_paths = [
    "0305_images/2025-03-06-20-10-14_left/1.png",
    "0305_images/2025-03-06-20-10-14_mid/1.png",
    "0305_images/2025-03-06-20-10-14_right/1.png"]

for image_path in image_paths:
    # Load the uploaded image
    image = cv2.imread(image_path)

    # Check if the image loaded correctly
    if image is None:
        raise ValueError("Error: Could not load the image.")

    # Get image dimensions
    height, width = image.shape[:2]

    # Camera intrinsic parameters (provided earlier)
    K = np.array([
        [613.4383544921875, 0.0, 318.04266357421875],
        [0.0, 613.40478515625, 243.88189697265625],
        [0.0, 0.0, 1.0]
    ])
    fx, fy = float(K[0, 0]), float(K[1, 1])
    u0, v0 = float(K[0, 2]), height - float(K[1, 2])

    # Camera parameters
    cx, cy, cz = 0, 3, 0 # Camera position (x, y, z in meters)
    fov_angles = np.arange(-60, 75, 15)  # -60° to +60° in 15° steps
    ranges = [50, 100, 150]  # Fan-like markers in meters
    range_colors = [(0, 0, 255), (0, 255, 255), (0, 255, 0)]  # Red, Yellow, Green

    def project_to_image(X, Y, Z):
        u = int(u0 + (fx * (X - cx)) / (Z - cz))
        v = int(v0 - (fy * (Y - cy)) / (Z - cz))
        
        return u, v

    # Draw FOV radial lines (-60° to +60°)
    for angle in fov_angles:
        rad = np.radians(angle)
        X = np.sin(rad) * 200  # X-position based on angle
        Y = 0
        Z = np.cos(rad) * 200  # Extend lines to 200m ahead
        u, v = project_to_image(X, Y, Z)
        origin_u = int(u0)
        origin_v = int(v0 + fy)
        cv2.line(image, (origin_u, origin_v), (u, v), (255, 255, 255), 2)

    # Draw perpendicular lines every 15 degrees for 50m, 100m, 150m distances
    for r, color in zip(ranges, range_colors):
        for angle in fov_angles:  # Every 15 degrees
            rad = np.radians(angle)
            X = np.sin(rad) * r  # Compute X position based on angle
            Y = 0
            Z = np.cos(rad) * r  # Distance from the camera

            # Get the perpendicular line points
            # (u1, v1), (u2, v2) = project_perpendicular_to_fov(X, Y, Z)

            u1, v1 = project_to_image(X, Y, Z)
            u2, v2 = u1 + 20, v1

            # Draw the perpendicular line
            cv2.line(image, (u1, v1), (u2, v2), color, 2)

            # add distance text and right of the line
            text = f"{r}m"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)[0]
            text_x = u2 + 5
            text_y = v2 + text_size[1] // 2
            cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

    fov_angles = np.arange(-60, 65, 5)  # -60° to +60° in 15° steps
    # Draw vertical lines at every 5 degrees for 200m distances
    for angle in fov_angles:
        rad = np.radians(angle)
        X = np.sin(rad) * 200  # X-position based on angle
        Y = 0
        Z = np.cos(rad) * 200  # Extend lines to 200m ahead
        u, v = project_to_image(X, Y, Z)
        u1 = u
        v1 = 50
        u2 = u
        v2 = 70
        cv2.line(image, (u1, v1), (u2, v2), (255, 255, 255), 2)
        # add degree text and put center of the line
        if image_path.find("mid") != -1:
            text = f"{angle}"
        elif image_path.find("left") != -1:
            text = f"{angle-35}"
        else:    
            text = f"{angle+35}"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        text_x = u - text_size[0] // 2
        text_y = v2 + text_size[1] + 5
        cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # add "Degree" text
    text = "Degree"
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    text_x = width // 2 - text_size[0] // 2
    text_y = 30
    cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        


    # Save and display the output image
    if image_path.find("left") != -1:
        output_image_path = "fov_ranges_left.png"
    elif image_path.find("mid") != -1:
        output_image_path = "fov_ranges_mid.png"
    else:
        output_image_path = "fov_ranges_right.png"
    cv2.imwrite(output_image_path, image)
    print(f"Output image saved to: {output_image_path}")

