import time
import torch
from torchvision.transforms.functional import to_tensor 
from argparse import ArgumentParser
import matplotlib.pyplot as plt

from oft import KittiObjectDataset, OftNet, ObjectEncoder, visualize_objects
from oft.matrix import *

def parse_args():
    parser = ArgumentParser()

    parser.add_argument('--model-path', type=str,
                        help='path to checkpoint file containing trained model')
    parser.add_argument('-g', '--gpu', type=int, default=0,
                        help='gpu to use for inference (-1 for cpu)')
    
    # Data options
    parser.add_argument('--root', type=str, default='oft/data/kitti',
                        help='root directory of the KITTI dataset')
    parser.add_argument('--grid-size', type=float, nargs=2, default=(80., 80.),
                        help='width and depth of validation grid, in meters')
    parser.add_argument('--yoffset', type=float, default=1.74,
                        help='vertical offset of the grid from the camera axis')
    parser.add_argument('--nms-thresh', type=float, default=0.2,
                        help='minimum score for a positive detection')

    # Model options
    parser.add_argument('--grid-height', type=float, default=4.,
                        help='size of grid cells, in meters')
    parser.add_argument('-r', '--grid-res', type=float, default=0.5,
                        help='size of grid cells, in meters')
    parser.add_argument('--frontend', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34'],
                        help='name of frontend ResNet architecture')
    parser.add_argument('--topdown', type=int, default=8,
                        help='number of residual blocks in topdown network')
    
    return parser.parse_args()


def calculate_iou(d, o):
    """
    Calculate the 3D IoU between a detection and a ground truth object.
    """
    corners_3d_ground  = get_3d_box(o[2], o[3], o[1])
    corners_3d_predict = get_3d_box(d[2].detach().numpy(), d[3].detach().numpy(), d[1].detach().numpy())
    (IOU_3d, IOU_2d)=box3d_iou(corners_3d_predict,corners_3d_ground)
    return IOU_3d

def calculate_precision_recall(detections, objects):
    """
    Calculate precision and recall for each detection in a list of detections, sorted by score.
    """
    tp = 0
    fp = 0
    fn = [True] * len(objects)
    precision = []
    recall = []
    for d in detections:
        max_iou = 0
        thresh=0.2
        max_idx = -1
        for i, o in enumerate(objects):
            iou = calculate_iou(d, o)
            if iou > max_iou and iou > thresh:
                max_iou = iou
                max_idx = i
        if max_idx != -1:
            tp=+1
            fn[max_idx] = False
            #fp.append(False)
        else:
            #tp.append(False)
            fp+=1
        #print(sum(tp))
        #print(sum(fp))
    if (tp + fp)!=0:
        precision.append(tp / (tp + fp))
    else:
        precision.append(0)
    recall.append(tp / len(fn))
    #print("TP=",tp)
    #print("FP=",fp)
    #print("FN=",len(fn))
    return precision, recall

def compute_ap(recall, precision):
    """Compute average precision."""
    recall = np.concatenate(([0.], recall, [1.]))
    precision = np.concatenate(([0.], precision, [0.]))
    for i in range(precision.size - 2, -1, -1):
        precision[i] = np.maximum(precision[i], precision[i + 1])
    idx = np.where(recall[1:] != recall[:-1])[0]
    ap = np.sum((recall[idx + 1] - recall[idx]) * precision[idx + 1])
    return ap

def main():

    # Parse command line arguments
    args = parse_args()

    # Load validation dataset to visualise
    dataset = KittiObjectDataset(
        args.root, 'val', args.grid_size, args.grid_res, args.yoffset)
    
    # Build model
    model = OftNet(num_classes=1, frontend=args.frontend, 
                   topdown_layers=args.topdown, grid_res=args.grid_res, 
                   grid_height=args.grid_height)
    if args.gpu >= 0:
        torch.cuda.set_device(args.gpu)
        model.cuda()
    
    # Load checkpoint
    ckpt = torch.load(args.model_path)
    model.load_state_dict(ckpt['model'])
    #print(ckpt['model'])

    # Create encoder
    encoder = ObjectEncoder(nms_thresh=args.nms_thresh)

    # Set up plots
    _, (ax1, ax2) = plt.subplots(nrows=2)
    plt.ion()
    ap=[]
    img=1
    cycle=[]
    ped=[]
    tram=[]
    van=[]
    truck=[]
    per_sitting=[]
    car=[]
    misc=[]
    dont_care=[]
    
    # Iterate over validation images
    for _, image, calib, objects, grid in dataset:

        #print(car)
        #print(cycle)
        # Move tensors to gpu
        image = to_tensor(image)
        
        if args.gpu >= 0:
            image, calib, grid = image.cuda(), calib.cuda(), grid.cuda()

        # Run model forwards
        pred_encoded = model(image[None], calib[None], grid[None])
        #print("Pred!!!!!!")
        #print(pred_encoded)
        # Decode predictions
        pred_encoded = [t[0].cpu() for t in pred_encoded]
        detections = encoder.decode(*pred_encoded, grid.cpu())
        #print("OBJECTS!!!!!!")
        #print(objects)
        #print(len(detections))

        precision, recall= calculate_precision_recall(detections, objects)
        a=compute_ap(recall, precision)
        ap.append(a)
        print("Image {} done!".format(img))
        print(a)
        img+=1
        for obj in objects:
            #print(type(obj))
            if obj.classname == "Pedestrian":
                ped.append([obj[2]])
            elif obj.classname == "Cyclist":
                cycle.append([obj[2]])
            elif obj.classname == "Van":
                van.append([obj[2]])
            elif obj.classname == "Truck":
                truck.append([obj[2]])
            elif obj.classname == "Tram":
                tram.append([obj[2]])
            elif obj.classname == "Person_sitting":
                per_sitting.append([obj[2]])
            elif obj.classname == "Car":
                car.append([obj[2]])
                #s_car=np.std(car, axis=0, dtype=float)
                #m_car=np.mean(car, axis=0, dtype=float)
                #col=car[:,3]
                #print(m_car)
            elif obj.classname == "Misc":
                misc.append([obj[2]])
            elif obj.classname == "DontCare":
                dont_care.append([obj[2]])
        
        
        '''for i in range(min(len(objects),len(detections))):
            d = detections[i]
            print(d)
            o = objects[i]
            best=0
            check=[]
            for j in range(len(objects)):
                corners_3d_ground  = get_3d_box(objects[j][2], objects[j][3], objects[j][1]) 
                corners_3d_predict = get_3d_box(detections[i][2].detach().numpy(), detections[i][3].detach().numpy(), detections[i][1].detach().numpy())
                (IOU_3d, IOU_2d)=box3d_iou(corners_3d_predict,corners_3d_ground)
                check.append(IOU_3d)
                
            best=max(check)
            avg.append(best)
        acc=sum(avg)/len(avg)            
        print(acc)'''
        
        
        # Visualize predictions
        visualize_objects(image, calib, detections, ax=ax1)
        ax1.set_title('Detections{}'.format(a))
        visualize_objects(image, calib, objects, ax=ax2)
        ax2.set_title('Ground truth')

        plt.draw()
        plt.pause(0.01)
        time.sleep(0.5)
    
        plt.savefig("viz.png")
        
        #if img==4:
            #break
        
    if len(ap) != 0:
        mAp=sum(ap)/len(ap)
        print(mAp)
    '''m_car=np.mean(car, axis=0, dtype=float)
    m_van=np.mean(van, axis=0, dtype=float)
    m_tram=np.mean(tram, axis=0, dtype=float)
    m_truck=np.mean(truck, axis=0, dtype=float)
    m_ped=np.mean(ped, axis=0, dtype=float)
    m_per_sit=np.mean(per_sitting, axis=0, dtype=float)
    m_misc=np.mean(misc, axis=0, dtype=float)
    m_dc=np.mean(dont_care, axis=0, dtype=float)
    m_cycle=np.mean(cycle, axis=0, dtype=float)
    
    s_car=np.std(car, axis=0, dtype=float)
    s_van=np.std(van, axis=0, dtype=float)
    s_tram=np.std(tram, axis=0, dtype=float)
    s_truck=np.std(truck, axis=0, dtype=float)
    s_ped=np.std(ped, axis=0, dtype=float)
    s_per_sit=np.std(per_sitting, axis=0, dtype=float)
    s_misc=np.std(misc, axis=0, dtype=float)
    s_dc=np.std(dont_care, axis=0, dtype=float)
    s_cycle=np.std(cycle, axis=0, dtype=float)
    
    print("MEAN!!!!")
    print("CAR:",m_car)
    print("CYCLE:",m_cycle)
    print("TRAM:",m_tram)
    print("TRUCK:",m_truck)
    print("VAN:",m_van)
    print("PEDESTRAIN:",m_ped)
    print("PERSON SITTING:",m_per_sit)
    print("MISC:",m_misc)
    print("DONT CARE:",m_dc)
    
    print("STD!!!!!")
    print("CAR:",s_car)
    print("CYCLE:",s_cycle)
    print("TRAM:",s_tram)
    print("TRUCK:",s_truck)
    print("VAN:",s_van)
    print("PEDESTRAIN:",s_ped)
    print("PERSON SITTING:",s_per_sit)
    print("MISC:",s_misc)
    print("DONT CARE:",s_dc)'''
    
if __name__ == '__main__':
    main()