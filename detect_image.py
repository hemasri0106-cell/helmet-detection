import cv2

# Load the cascade classifier
helmet_cascade = cv2.CascadeClassifier('helmet_cascade.xml')

# Load the image
img = cv2.imread('test_image.jpg')

# Check if image is loaded
if img is None:
    print("Error: Could not read image.")
    exit()

# Convert to grayscale (Haar Cascades work on grayscale images)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect helmets
# You can tune scaleFactor and minNeighbors
# scaleFactor: How much the image size is reduced at each image scale
# minNeighbors: How many neighbors each candidate rectangle should have
helmets = helmet_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

print(f"Found {len(helmets)} helmets.")

# Draw a rectangle around the detected helmets
for (x, y, w, h) in helmets:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(img, 'Helmet', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Display the output
cv2.imshow('Helmet Detection', img)
cv2.waitKey(0) # Wait for a key press to close the window
cv2.destroyAllWindows()