import torch, cv2, json, os, glob
import matplotlib.pyplot as plt
from cityscapesscripts.helpers.annotation import Annotation
from cityscapesscripts.helpers.labels     import labels, name2label
import PIL.Image     as Image
import PIL.ImageDraw as ImageDraw

# img_path = '/home/xianr/TurboRuns/cityscapes/leftImg8bit/train/aachen/aachen_000029_000019_leftImg8bit.png'
# json_path = '/home/xianr/TurboRuns/cityscapes/gtFine/train/aachen/aachen_000029_000019_gtFine_polygons.json'
# labled_img_path = '/home/xianr/TurboRuns/cityscapes/gtFine/train/aachen/aachen_000029_all_gtFine_instances.png'

cityscapesPath = '/home/xianr/TurboRuns/cityscapes/'
searchFine = os.path.join( cityscapesPath , "gtFine"   , "*" , "*" , "*_gt*_polygons.json" )
filesFine = glob.glob(searchFine )

for json_path in filesFine:
    # create the output filename
    labled_img_path = json_path.replace( "_polygons.json" , "_all_instances.png" )
    if labled_img_path.endswith('json'):
        import pdb; pdb.set_trace()

    annotation = Annotation()
    annotation.fromJsonFile(json_path)

    backgroundId = name2label['unlabeled'].id
    size = ( annotation.imgWidth , annotation.imgHeight )
    instanceImg = Image.new("I", size, backgroundId)
    drawer = ImageDraw.Draw( instanceImg )

    for obj in annotation.objects:
        # import pdb; pdb.set_trace()
        label   = obj.label
        polygon = obj.polygon

        if obj.deleted:
            import pdb; pdb.set_trace()
            continue
        if ( not label in name2label ):
            if label not in [
                'bicyclegroup', 'persongroup', 'cargroup', 'ridergroup', 'truckgroup',
                'motorcyclegroup'
            ]:
                import pdb; pdb.set_trace()
                print(label)
                img_path = json_path.replace('cityscapes/gtFine', 'cityscapes/leftImg8bit').replace( "_gtFine_polygons.json" , "_leftImg8bit.png" )
                img = plt.imread(img_path)
                plt.imshow(img)
                plt.show()
            label = label[:-len('group')]
        labelTuple = name2label[label]
        id = labelTuple.id
        if obj.id < 0:
            import pdb; pdb.set_trace()
        drawer.polygon( polygon, fill=obj.id)
        
    instanceImg.save( labled_img_path)



