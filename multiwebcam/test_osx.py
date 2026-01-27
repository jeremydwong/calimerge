import cv2
import platform

print(f"OpenCV version: {cv2.__version__}")
print(f"Platform: {platform.system()}")

port_to_test = 0 # Or whatever your camera port is
api_preference = cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else cv2.CAP_ANY

cap = cv2.VideoCapture(port_to_test, api_preference)

if not cap.isOpened():  
    print(f"Error: Cannot open camera at port {port_to_test} with API {api_preference}")
    exit()

print(f"Camera {port_to_test} opened successfully!")
print(f"Default Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
print(f"Default FPS: {cap.get(cv2.CAP_PROP_FPS)}")

# Try setting a common resolution
# success_res = cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# success_res = cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# if success_res:
# print(f"Set Resolution to: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
# else:
# print("Failed to set resolution.")


for i in range(30): # Try to read a few frames
    ret, frame = cap.read()
    if not ret:
        print(f"Error: Can't receive frame {i} (stream end?). Exiting ...")
        break
    cv2.imshow(f'Camera {port_to_test} - Press Q to quit', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Test finished.")