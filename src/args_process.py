import os
import argparse
import json

if __name__ == '__main__':
    # config_parser = argparse.ArgumentParser(prog=program_name, description=’Compile database script’,
    # add_help=False)
    # # JSON support
    # config_parser.add_argument('— json_file’, help=’Configuration JSON file’)config_parser.add_argument(‘ — json_keys’, nargs=’+’, help=’JSON keys’)
    # parser = argparse.ArgumentParser(parents=[config_parser])
    # parser.add_argument(‘ — force_remap’, help=’Re-Map all libraries (even if they are already mapped’, action=’store_true’)
    # parser_group_db_comp.add_argument('— opt_top', nargs='+', help= 'Specify which module to optimize (all : all modules defined in yaml file)',default=[])
    # json_dict = json.load(open('train_args/myargs0.json'))
    # print(json_dict)
    parser1 = argparse.ArgumentParser()
    parser1.add_argument('--arg_file', help = 'arg file')
    args1 = parser1.parse_args()
    arg_file_path = os.path.join('train_args', args1.arg_file + '.json')
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--train_batch_size', type = int, default = 4)
    # # parser.add_argument('--train_net', type = bool, default = False)
    # parser.add_argument('--num_train', type = int, default = 1000)
    # parser.add_argument('--num_val', type = int, default = 100)
    # parser.add_argument('--train_data_dir', type = str, default = '')
    # parser.add_argument('--val_data_dir', type = str, default = '')
    # parser.add_argument('--net_input_name', type = str, default = '')
    # parser.add_argument('--target_name', type = str, default = '')
    # parser.add_argument('--learning_rate', type = float, default = 0.1)
    # parser.add_argument('--num_train_epochs', type = int, default = 5)
    # parser.add_argument('--model_name', type = str, default = '')
    # parser.add_argument('--net_type', type = str, default= 'unet2d')
    # parser.add_argument('--in_channels', type = int, default= 2)
    # parser.add_argument('--out_channels', type = int, default = 1)
    # parser.add_argument('--mid_channels', type = int, default = 4)
    # parser.add_argument('--depth', type = int, default =6)
    # # parser.add_argument('--kernel_size', type = int, default=3)
    # # parser.add_argument('--dilation', type = int, default=2)
    # # parser.add_argument('--padding', type = int, default =2)
    # parser.add_argument('--print_every', type = int, default = 50)
    # parser.add_argument('--weight_decay', type = float, default = 0, help = 'l2 regularization on weights')
    # parser.add_argument('--scheduler_stepsize', type = int, default = 5)
    # parser.add_argument('--scheduler_gamma', type = float, default = 0.8)
    # args = parser.parse_args()

    # argparse_dict = vars(args)
    # argparse_dict.update(json_dict)

    t_args = argparse.Namespace()
    t_args.__dict__.update(json.load(open(arg_file_path)))
    args = parser1.parse_args(namespace=t_args)
    print(vars(args))


    