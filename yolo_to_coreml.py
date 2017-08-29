import coremltools
import argparse
import configparser
import io
import os
from collections import defaultdict
from coremltools.models.neural_network import NeuralNetworkBuilder
from coremltools.models import datatypes
from coremltools.models import utils
import numpy as np
import pprint

parser = argparse.ArgumentParser(
    description='tiny-yolo to coreml converter.')
parser.add_argument('config_path', help='Path to Darknet cfg file.')
parser.add_argument('weights_path', help='Path to Darknet weights file.')
parser.add_argument('output_path', help='Path to output coreml model file.')
parser.add_argument('--verbose', help='Show all debug info.', action='store_true')
parser.add_argument('--print_spec', help='Save .proto file along with mlmodel', action='store_true')
parser.add_argument('--output_name', type=str, help='Output layer name. Default is grid')

def configure_model(model, path, grid_w, grid_h):
    model.author = 'Original paper: Joseph Redmon, Ali Farhadi'
    model.license = 'Public Domain'
    model.short_description = "Based on Tiny YOLO network from the paper 'YOLO9000: Better, Faster, Stronger' (2016), arXiv:1612.08242"

    model.input_description['image'] = 'Input image'
    model.output_description['grid'] = 'The {}x{} grid with the bounding box data'.format(grid_w, grid_h)

    print(model)

    model.save(path)

def output_make(name):
    return '{}_output'.format(name)

def make_scheme(top_input, cfg_parser):
    output_shape_h = int(cfg_parser['net_0']['height'])
    output_shape_w = int(cfg_parser['net_0']['width'])
    output_shape_c = int(cfg_parser['net_0']['channels'])

    index  = 0
    scheme = []
    last_output = top_input
    last_padding = 1
    last_kernel  = 1
    for section in cfg_parser.sections():
        print('Parsing section {}'.format(section))
        index += 1
        if section.startswith('convolutional'):
            filters = int(cfg_parser[section]['filters'])
            size = int(cfg_parser[section]['size'])
            stride = int(cfg_parser[section]['stride'])
            pad = int(cfg_parser[section]['pad'])
            activation = cfg_parser[section]['activation']
            batch_normalize = 'batch_normalize' in cfg_parser[section]

            # padding='same' is equivalent to Darknet pad=1
            padding = 'same' if pad == 1 else 'valid'
            last_kernel = size

            convo_name = 'conv2D_{}'.format(index)

            scheme.append({
                'type': 'conv2D',
                'name': '{}:{}x{}x{}'.format(convo_name, output_shape_w, output_shape_h, filters),
                'input_name': last_output,
                'output_name': output_make(convo_name),
                'filters': filters,
                'size': size,
                'stride': stride,
                'pad': padding,
                'input_shape_c': output_shape_c,
                'input_shape_w': output_shape_w,
                'input_shape_h': output_shape_h,
                'normalized': batch_normalize
            })

            last_output = scheme[-1]['output_name']
            output_shape_c = filters

            if batch_normalize:
                batch_normalize_name = 'batch_normalization_{}'.format(index)
                scheme.append({
                    'type': 'batch_norm',
                    'name': '{}:{}x{}x{}'.format(batch_normalize_name, output_shape_w, output_shape_h, output_shape_c),
                    'input_name': last_output,
                    'output_name': output_make(batch_normalize_name),
                    'channels': filters
                })
                last_output = scheme[-1]['output_name']

            # Handle activation.
            act_fn = None
            if activation == 'leaky':
                act_fn = 'LEAKYRELU'
            elif activation == 'linear':
                act_fn = 'LINEAR'
            else:
                raise ValueError(
                    'Unknown activation function `{}` in section {}'.format(
                        activation, section))


            activation_name = '{}_{}'.format(activation, index)
            scheme.append({
                'type': 'activation',
                'name': activation_name,
                'input_name': last_output,
                'output_name': output_make(activation_name),
                'non_linearity': act_fn,
                'alpha': [0.1] # TODO: can it be hue from .cfg file ??
            })
            last_output = scheme[-1]['output_name']

        elif section.startswith('maxpool'):
            size = int(cfg_parser[section]['size'])
            stride = int(cfg_parser[section]['stride'])
            pool_padding = int(cfg_parser[section].get('padding') or 0)

            scheme.append({
                'type': 'maxpool',
                'name': section,
                'output_name': output_make(section),
                'input_name': last_output,
                'stride': stride,
                'size': size
            })

            last_output = scheme[-1]['output_name']

            output_shape_h = (output_shape_h + pool_padding*2)/stride
            output_shape_w = (output_shape_w + pool_padding*2)/stride

        elif section.startswith('avgpool'):
            size = int(cfg_parser[section]['size'])
            stride = int(cfg_parser[section]['stride'])
            scheme.append({
                'type': 'avgpool',
                'name': section,
                'output_name': output_make(name),
                'input_name': last_output,
                'stride': stride,
                'size': size
            })
            last_output = scheme[-1]['output_name']

        elif section.startswith('route'):
            assert False, "route layer is not implemented"

        elif section.startswith('reorg'):
            assert False, "reorg layer is not implemented"

        elif section.startswith('region'):
            print('anchors=%s'%(cfg_parser[section]['anchors']))

        elif (section.startswith('net') or section.startswith('cost') or
              section.startswith('softmax')):
            pass  # Configs not currently handled during model definition.

        else:
            raise ValueError(
                'Unsupported section header type: {}'.format(section))

    return (scheme, output_shape_h, output_shape_w, output_shape_c)



def unique_config_sections(config_file):
    """Convert all config sections to have unique names.

    Adds unique suffixes to config sections for compatibility with configparser.
    """
    section_counters = defaultdict(int)
    output_stream = io.BytesIO()
    with open(config_file) as fin:
        for line in fin:
            if line.startswith('['):
                section = line.strip().strip('[]')
                _section = section + '_' + str(section_counters[section])
                section_counters[section] += 1
                line = line.replace(section, _section)
            output_stream.write(line)
    output_stream.seek(0)
    return output_stream

def _main(args):
    config_path = os.path.expanduser(args.config_path)
    weights_path = os.path.expanduser(args.weights_path)
    assert config_path.endswith('.cfg'), '{} is not a .cfg file'.format(
        config_path)
    assert weights_path.endswith(
        '.weights'), '{} is not a .weights file'.format(weights_path)

    output_path = os.path.expanduser(args.output_path)
    assert output_path.endswith(
        '.mlmodel'), 'output path {} is not a .mlmodel file'.format(output_path)
    output_root = os.path.splitext(output_path)[0]

    # Load weights and config.
    print('Loading weights.')
    weights_file = open(weights_path, 'rb')

    major = np.ndarray(shape=(1, ), dtype='int32', buffer=weights_file.read(4))
    minor = np.ndarray(shape=(1, ), dtype='int32', buffer=weights_file.read(4))
    revision = np.ndarray(shape=(1, ), dtype='int32', buffer=weights_file.read(4))

    if args.verbose:
        print ("major = {}".format(major))
        print ("minor = {}".format(minor))

    if major[0]*10 + minor[0] >= 2:
        seen = np.ndarray(shape=(1, ), dtype='int64', buffer=weights_file.read(8))
    else:
        unknown = np.ndarray(shape=(1, ), dtype='int32', buffer=weights_file.read(4))

    assert major[0] < 1000 and minor[0] < 1000 , "Unsupported configuration. Sorry :("


    print('Parsing Darknet config {}'.format(config_path))
    unique_config_file = unique_config_sections(config_path)
    cfg_parser = configparser.ConfigParser()
    cfg_parser.read_file(unique_config_file)

    print('Creating CoreML model.')

    image_height = int(cfg_parser['net_0']['height'])
    image_width = int(cfg_parser['net_0']['width'])
    channels = int(cfg_parser['net_0']['channels'])

    top_input = 'image'
    config = make_scheme(top_input=top_input,cfg_parser=cfg_parser)
    scheme = config[0]
    scheme[-1]['output_name'] = 'grid' if args.output_name == None else args.output_name
    output_layer = scheme[-1]['output_name']

    if args.verbose:
        pp = pprint.PrettyPrinter(depth=6)
        pp.pprint(config[0])
        print('output_shape: {}'.format(config[1:]))
    else:
        print('Main config: image_width={}, image_height={}, channels={}'.format(image_width,image_height, channels))

    input_features=[(top_input, datatypes.Array(channels, image_height, image_width))]
    output_features=[(output_layer, datatypes.Array(config[3],config[1], config[2]))]

    builder = NeuralNetworkBuilder(input_features, output_features)
    count = 0
    conv_bias = None
    bn_weight_list = None

    for section in config[0]:
        print('Adding layer {}'.format(section['name']))
        if section['type'] == 'conv2D':
            filters = section['filters']
            size = section['size']
            stride = section['stride']
            padding = section['pad']
            normalized = section['normalized']
            input_shape_c = section['input_shape_c']
            output_shape_w = section['input_shape_w']
            output_shape_h = section['input_shape_h']

            # Setting weights.
            # Darknet serializes convolutional weights as:
            # [bias/beta, [gamma, mean, variance], conv_weights]

            conv_bias = np.ndarray(
                shape=(filters, ),
                dtype='float32',
                buffer=weights_file.read(filters * 4))
            count += filters

            if normalized:
                bn_weights = np.ndarray(
                    shape=(3, filters),
                    dtype='float32',
                    buffer=weights_file.read(filters * 12))
                count += 3 * filters

                bn_weight_list = [
                    bn_weights[0],  # scale gamma
                    conv_bias,  # shift beta
                    bn_weights[1],  # running mean
                    bn_weights[2]  # running var
                ]

            darknet_w_shape = (filters, input_shape_c, size, size)
            weights_size = np.product(darknet_w_shape)

            conv_weights = np.ndarray(
                shape=darknet_w_shape,
                dtype='float32',
                buffer=weights_file.read(weights_size * 4))
            count += weights_size

            # DarkNet conv_weights are serialized Caffe-style:
            # (out_dim, in_dim, height, width)
            # We would like to set these to Keras order:
            # (height, width, in_dim, out_dim)
            # because coremltools expect Keras format to convert back to Caffe-style

            conv_weights = np.transpose(conv_weights, [2, 3, 1, 0])

            builder.add_convolution(
                name=section['name'],
                kernel_channels=input_shape_c,
                output_channels=filters,
                height=size,
                width=size,
                stride_height=stride,
                stride_width=stride,
                border_mode=padding,
                groups=1,
                W=conv_weights,
                b=conv_bias,
                has_bias=not normalized,
                is_deconv=False,
                output_shape=(output_shape_w,output_shape_h),
                input_name=section['input_name'],
                output_name=section['output_name'])

        elif section['type'] == 'batch_norm':
            builder.add_batchnorm(
                name=section['name'],
                channels=section['channels'],
                gamma=bn_weight_list[0],
                beta=bn_weight_list[1],
                mean=bn_weight_list[2],
                variance=bn_weight_list[3],
                input_name=section['input_name'],
                output_name=section['output_name'])


        elif section['type'] == 'activation':
            builder.add_activation(
                name=section['name'],
                non_linearity=section['non_linearity'],
                input_name=section['input_name'],
                output_name=section['output_name'],
                params=section['alpha'])

        elif section['type'] == 'maxpool':
            builder.add_pooling(
                name=section['name'],
                height=section['size'],
                width=section['size'],
                stride_height=section['stride'],
                stride_width=section['stride'],
                layer_type="MAX",
                padding_type="SAME",
                input_name=section['input_name'],
                output_name=section['output_name'])

        elif section['type'] == 'avgpool':
            size = section['size']
            stride = section['stride']
            builder.add_pooling(
                name=section['name'],
                height=size, width=size,
                stride_height=stride,
                stride_width=stride,
                layer_type="AVERAGE",
                padding_type="SAME",
                input_name=section['input_name'],
                output_name=section['output_name'])
        else:
            raise ValueError(
                'Unsupported section header type: {}'.format(section))

    output_shape_w = config[2]
    output_shape_h = config[1]
    output_shape_c = config[3]

    builder.set_input([top_input], [(channels, image_height, image_width)])
    builder.set_output([output_layer], [(output_shape_c, output_shape_h, output_shape_w)])

    builder.set_pre_processing_parameters(
        image_input_names = [top_input],
        is_bgr=False,
        image_scale=1/255.)

    spec = builder.spec

    # Create and save model.
    path = '{}'.format(output_path)
    model = coremltools.models.MLModel(spec)
    configure_model(model=model, path=path, grid_w=output_shape_w, grid_h=output_shape_h)
    print('mlmodel saved to: {}'.format(path))

    # Check to see if all weights have been read.
    remaining_weights = len(weights_file.read()) / 4
    weights_file.close()

    if args.print_spec:
        outpath, ext = os.path.splitext(path)
        specname = '{}.proto'.format(outpath)
        with open(specname, 'wb') as f:
            f.write('{}'.format(spec))
            print("Poto file saved to: {}".format(specname))

    if remaining_weights > 0:
        print('Read {} of {} from Darknet weights.'.format(count, count + remaining_weights))
        print('Warning: {} unused weights'.format(remaining_weights))


if __name__ == '__main__':
    _main(parser.parse_args())
