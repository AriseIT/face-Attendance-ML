''' for live recognition'''


import os
import cv2
import numpy as np
import face_recognition as fr

path = ".\\train\\"

known_names = []
known_name_encodings = []

images = os.listdir(path)

for _ in images:
    image = fr.load_image_file(path + _)
    image_path = path + _
    encoding = fr.face_encodings(image)[0]
    known_name_encodings.append(encoding)
    known_names.append(os.path.splitext(os.path.basename(image_path))[0].capitalize())

cap = cv2.VideoCapture(0)

while True:
    ret, image = cap.read()
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    face_locations = fr.face_locations(image)
    face_encodings = fr.face_encodings(image, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = fr.compare_faces(known_name_encodings, face_encoding)
        name = ""
        face_distances = fr.face_distance(known_name_encodings, face_encoding)
        best_match = np.argmin(face_distances)
        
        if matches[best_match]:
            name = known_names[best_match]
            print(name)
        
        font = cv2.FONT_HERSHEY_DUPLEX
        
        if name != '':        
            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(image, (left, bottom - 15), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(image, name, (left + 6, bottom), font, 0.7, (255, 255, 255), 1)
        else:
            cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(image, (left, bottom - 15), (right, bottom), (0, 0, 255), cv2.FILLED)
            cv2.putText(image, 'Unknown', (left + 6, bottom), font, 0.7, (255, 255, 255), 1)
            
    cv2.imshow("Result", image)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



'''for image recogiation'''
# import os
# import cv2
# import numpy as np
# import face_recognition as fr

# path = ".\\train\\"

# known_names = []
# known_name_encodings = []

# images = os.listdir(path)

# for _ in images:
#     image = fr.load_image_file(path + _)
#     image_path = path + _
#     encoding = fr.face_encodings(image)[0]
#     known_name_encodings.append(encoding)
#     known_names.append(os.path.splitext(os.path.basename(image_path))[0].capitalize())
    
# test_image = ".\\test\\test4.jpg"
# image = cv2.imread(test_image)
# up_width = 800
# up_height = 600
# up_points = (up_width, up_height)
# image = cv2.resize(image, up_points, interpolation= cv2.INTER_LINEAR)

# face_locations = fr.face_locations(image)
# face_encodings = fr.face_encodings(image, face_locations)

# for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
#    matches = fr.compare_faces(known_name_encodings, face_encoding)
#    name = ""
#    face_distances = fr.face_distance(known_name_encodings, face_encoding)
#    best_match = np.argmin(face_distances)
   
#    if matches[best_match]:
#        name = known_names[best_match]
#        print(name)
       
#    font = cv2.FONT_HERSHEY_DUPLEX
   
#    if name != '':        
#        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
#        cv2.rectangle(image, (left, bottom - 15), (right, bottom), (0, 255, 0), cv2.FILLED)
#        cv2.putText(image, name, (left + 6, bottom), font, 0.7, (255, 255, 255), 1)
#    else:
#        cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
#        cv2.rectangle(image, (left, bottom - 15), (right, bottom), (0, 0, 255), cv2.FILLED)
#        cv2.putText(image, 'Unknown', (left + 6, bottom), font, 0.7, (255, 255, 255), 1)
          
# cv2.imshow("Result", image)
# cv2.imwrite(".\\output.jpg", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
