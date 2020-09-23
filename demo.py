import argparse
import os
import cv2
import numpy as np
from PIL import Image
from PIL import ImageEnhance


def parse_args():
    parser = argparse.ArgumentParser(description='Demo a face recognition')

    parser.add_argument('--input',
                        help='the dir to img need to be proccessed')
    parser.add_argument('--output',
                        help='the dir to img output')
    parser.add_argument('--mode',
                        help='mode IAGCWD or LLIE',
                        default='IAGCWD', type=str)
    parser.add_argument("-l", '--lambda_', default=0.15, type=float,
                    help="the weight for balancing the two terms in the illumination refinement optimization objective.")
    parser.add_argument("-ul", "--lime", action='store_true',
                        help="Use the LIME method. By default, the DUAL method is used.",
                        default=False)
    parser.add_argument("-g", '--gamma', default=0.6, type=float,
                        help="the gamma correction parameter.")
    parser.add_argument('--sigma', default=3, type=int,
                        help="Spatial standard deviation for spatial affinity based Gaussian weights.")
    parser.add_argument("-bc", default=1, type=float,
                        help="parameter for controlling the influence of Mertens's contrast measure.")
    parser.add_argument("-bs", default=1, type=float,
                        help="parameter for controlling the influence of Mertens's saturation measure.")
    parser.add_argument("-be", default=1, type=float,
                        help="parameter for controlling the influence of Mertens's well exposedness measure.")
    parser.add_argument("-eps", default=1e-3, type=float,
                        help="constant to avoid computation instability.")
# LLIE only use with low-light Image
    args = parser.parse_args()
    return args


def increase_sharp(image):
    image_pil = Image.fromarray(image[..., ::-1])
    enh_sha = ImageEnhance.Sharpness(image_pil)
    sharpness = 3.0
    image_sharped = enh_sha.enhance(sharpness)
    return image_pil


def main():

    img = cv2.imread(args.input, 1)
    YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    Y = YCrCb[:,:,0]
    # Determine whether image is bright or dimmed
    threshold = 0.4
    exp_in = 112 # Expected global average intensity 
    M,N = img.shape[:2]
    mean_in = np.sum(Y/(M*N))
    t = (mean_in - exp_in)/ exp_in
    
    # Process image for gamma correction
    img_output = None
    if args.mode == 'IAGCWD':
        from IAGCWD.process import process_bright
        from IAGCWD.process import process_dimmed

        if t < -threshold:
            # Dimmed Image
            print ('Image is Dimmed {}'.format(t))
            result = process_dimmed(Y)
            YCrCb[:,:,0] = result
            img_output = cv2.cvtColor(YCrCb,cv2.COLOR_YCrCb2BGR)
        elif t > threshold:
            # Bright Image
            print ('Image is Bright {}'.format(t))
            result = process_bright(Y)
            YCrCb[:,:,0] = result
            img_output = cv2.cvtColor(YCrCb,cv2.COLOR_YCrCb2BGR)
        else:
            print('Image is normal {}'.format(t))
            img_output = img

    elif args.mode == 'LLIE':
        from LLIE.process import enhance_image_exposure
        if t < -threshold:
            enhanced_image = enhance_image_exposure(
                img, args.gamma, args.lambda_, not args.lime,
                sigma=args.sigma, bc=args.bc, bs=args.bs, be=args.be, eps=args.eps
            )
            img_output = enhanced_image
        else:
            img_output = img

    image_out = increase_sharp(img_output)
    image_out = image_out.save(args.output)

    return 0


if __name__ == '__main__':
    args = parse_args()
    main()
