from ultralytics import YOLO
import cv2
import numpy as np
import random
import time
#from models import net
import torch
from torch.autograd import Variable
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
import torch.nn.functional as F
from torchvision.transforms import Compose






def show_box( box , name , image ):
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        # np.int8(cls[i].detach().cpu().numpy())
        depth_object = (np.int8((x0+w/2).detach().cpu().numpy()) , np.int8((y0+h/2).detach().cpu().numpy()))
        image = cv2.rectangle(image , (int(x0)+5, int(y0)+5), (int(x0+w), int(y0+h)), [0,0,255] ,  2)  
        font = cv2.FONT_HERSHEY_SIMPLEX
        #image = cv2.putText(image ,name, (int(x0)+5,int(y0)+25) , font , 1 , [0,0,0] , 2 ,  cv2.LINE_AA)
        text = f'{name}'
        image= cv2.putText(image ,text, (int(x0)+5,int(y0)+25) , font , 1 , [0,0,0] , 2 ,  cv2.LINE_AA)
        return image


#config YOLO
model = YOLO("YOLO//yolov8n-seg.pt")  
video_capture = cv2.VideoCapture("IMG_3002.MOV")
yolo_classes = list(model.names.values())
classes_ids = [yolo_classes.index(clas) for clas in yolo_classes]
colors = [random.choices(range(256), k=3) for _ in classes_ids]
font = cv2.FONT_HERSHEY_SIMPLEX
#config Depth
#encoders = ['vits', 'vitb', 'vitl']
encoder = 'vits'
video_path = 1
margin_width = 50
caption_height = 60
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_thickness = 2
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{}14'.format(encoder)).to(DEVICE)
total_params = sum(param.numel() for param in depth_anything.parameters())
depth_anything.eval()
transform = Compose([
    Resize(
        width=518,
        height=518,
        resize_target=False,
        keep_aspect_ratio=True,
        ensure_multiple_of=14,
        resize_method='lower_bound',
        image_interpolation_method=cv2.INTER_CUBIC,
    ),
    NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    PrepareForNet(),
])


# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'x264' might also be available
out_video = cv2.VideoWriter("output_video.mp4", fourcc, 30.0, (1440,720))

t0 = time.time()


counter = 0 
while (video_capture.isOpened()):
            
    ret, frame = video_capture.read()
    counter += 1
    if not ret:
        break
    # frame = cv2.resize(frame, (1080, 720), 
    #     interpolation = cv2.INTER_LINEAR)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    Org = np.copy(frame)
    image = np.array(frame)
    image = cv2.resize(image, (640, 480)) / 255.0
    image = transform({'image': image})['image']
    image = torch.from_numpy(image).unsqueeze(0).to(DEVICE)
    h, w, ch = frame.shape
    results = model.predict(frame, conf=0.25)
    with torch.no_grad():
        depth = depth_anything(image)
    # depth = call(cv2.resize(frame, (640, 480)))
    depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    
    depth = depth.cpu().numpy().astype(np.uint8)
    depth_color = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
    colors = [random.choices(range(256), k=3) for _ in classes_ids]
    print(results)
    for result in results:
        boxes =  result.boxes.xyxy
        cls = result.boxes.cls
        #masks = np.int32(result.masks.masks)
        try:
            mask2 = result.masks.xy
            for i in range(len(boxes)):
                color_number = classes_ids.index(int(cls[i]))
                cv2.fillPoly(frame, np.int32([mask2[i]]) , colors[color_number])
                frame = show_box(boxes[i] , yolo_classes[np.int8(cls[i].detach().cpu().numpy())] , frame )


        except Exception as e:
            print( "The error is: ", e )


    # for result in results:
    #     counter = 0 
    #     for mask, box in zip(result.masks.xy, result.boxes):
    #         points = np.int32([mask])
    #         # cv2.polylines(img, points, True, (255, 0, 0), 1)
    #         color_number = classes_ids.index(int(box.cls[0]))
    #         cv2.fillPoly(frame, points, colors[color_number])
    #         counter+=1
    #         if counter > 1 : 
    #             print("hi")
    frame = cv2.resize(frame, (720, 720), 
        interpolation = cv2.INTER_LINEAR)
    
    depth_color = cv2.resize(depth_color, (720, 720), 
        interpolation = cv2.INTER_LINEAR)
    
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


    Fin = cv2.hconcat([frame , depth_color] )
    cv2.imshow("Image", Fin)
    cv2.waitKey(10)
    out_video.write(Fin)

video_capture.release()
out_video.release()
t1 = time.time()
print(f"the whole time was {t1-t0} and the frame per seconds is {counter / (t1-t0)}" ) 



