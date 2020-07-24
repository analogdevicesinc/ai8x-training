import argparse
import torch

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', dest='file_path', help='path for checkpoint file')
    parser.add_argument('-d', dest='dest_path', help='path for converted checkpoint file')
    parser.add_argument('--overwrite', action='store_true', help='overwrite checkpoint file with the converted one')
    return parser.parse_args()

def main():
    args = parse_arguments()
    checkpoint = torch.load(args.file_path)
    print(checkpoint['compression_sched']['masks_dict'].keys())
    
    state_dict = checkpoint['state_dict']
    print('BEFORE:')
    print(state_dict.keys())

    state_dict_keys = list(state_dict.keys())
    for key in state_dict_keys:
        if key.endswith('.conv2d.weight'):
            layer_name, _ = key.rsplit('.conv2d.weight', maxsplit=1)
            out_shift_key = '.'.join([layer_name, 'output_shift'])
            checkpoint['state_dict'][out_shift_key] = 0

    print('AFTER:')
    print(state_dict.keys())

if __name__ == "__main__":
    main()
