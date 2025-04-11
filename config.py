import argparse

def get_arguments():
    parser = argparse.ArgumentParser(description="Internal Diverse Completion using SinGAN")
    parser.add_argument('--input_dir', default='Input', help='Directory containing the Images folder')
    parser.add_argument('--input_name', required=True, help='Input image file name (e.g., church.jpg)')
    parser.add_argument('--input_mask', default='', help='Input mask file name (e.g., church_mask.png)')
    parser.add_argument('--mode', default='train', help='Mode: train or random_samples')
    parser.add_argument('--num_scales', type=int, default=5, help='Number of pyramid scales')
    parser.add_argument('--nfc', type=int, default=32, help='Base number of filters (increase for higher capacity)')
    parser.add_argument('--ker_size', type=int, default=3, help='Kernel size for convolutions')
    parser.add_argument('--padd_size', type=int, default=1, help='Padding size for convolutions')
    parser.add_argument('--num_layer', type=int, default=5, help='Number of layers per scale')
    parser.add_argument('--stop_scale', type=int, default=5, help='Stop scale index')
    return parser

if __name__ == '__main__':
    parser = get_arguments()
    opt = parser.parse_args()
    print(opt)
