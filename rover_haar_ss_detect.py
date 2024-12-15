import cv2 as cv
import os


image_dir = "stop_sign_dataset"

# sadece jpg uzantili dosyalari dondurme islemi
image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]


haar_cascade = cv.CascadeClassifier('haar_stop_sign.xml')


for image_file in image_files:
    
    img = cv.imread(os.path.join(image_dir, image_file))
    
    # bgr --> grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # stop sign tespiti
    sign_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=17)
    
    print(f'{image_file} - Stop Sign Sayisi : {len(sign_rect)}')

    
    for (x, y, w, h) in sign_rect:
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
        
        # merkez hesaplama
        center_x = x + w // 2
        center_y = y + h // 2
        
       
        print(f'Stop Sign Merkezi : ({center_x}, {center_y})')
        cv.circle(img, (center_x, center_y), radius=5, color=(255, 0, 0), thickness=-1)
    
    
    output_file = os.path.join('output_images', f'detected_{image_file}')
    cv.imwrite(output_file, img)
    
    
    cv.imshow(f'Tespit Edilen Stop Sign : - {image_file}', img)

cv.waitKey(0)
cv.destroyAllWindows()
