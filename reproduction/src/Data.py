
def import_data():
    return


def parse_data():
    return

def load_data():
    return parse_data(import_data())



    #     # get symbol
    # pprint.pprint(config)
    # config.symbol = 'impression_network_dynamic_offset_sparse'
    # model = '/../local_run_output/impression_dynamic_offset-lr-10000-times-neighbor-4-dense-4'
    # first_sym_instance = eval(config.symbol + '.' + config.symbol)()
    # key_sym_instance = eval(config.symbol + '.' + config.symbol)()
    # cur_sym_instance = eval(config.symbol + '.' + config.symbol)()

    # first_sym = first_sym_instance.get_first_test_symbol_impression(config)
    # key_sym = key_sym_instance.get_key_test_symbol_impression(config)
    # cur_sym = cur_sym_instance.get_cur_test_symbol_impression(config)

    # # set up class names
    # num_classes = 31
    # classes = ['airplane', 'antelope', 'bear', 'bicycle',
    #            'bird', 'bus', 'car', 'cattle',
    #            'dog', 'domestic_cat', 'elephant', 'fox',
    #            'giant_panda', 'hamster', 'horse', 'lion',
    #            'lizard', 'monkey', 'motorcycle', 'rabbit',
    #            'red_panda', 'sheep', 'snake', 'squirrel',
    #            'tiger', 'train', 'turtle', 'watercraft',
    #            'whale', 'zebra']

    # # load demo data
    # image_names = glob.glob(cur_path + '/../demo/ILSVRC2015_val_00011005/*.JPEG')
    # output_dir = cur_path + '/../demo/motion-prior-output-00011005/'
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    # key_frame_interval = 10
    # image_names.sort()
    # data = []
    # for idx, im_name in enumerate(image_names):
    #     assert os.path.exists(im_name), ('%s does not exist'.format(im_name))
    #     im = cv2.imread(im_name, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    #     target_size = config.SCALES[0][0]
    #     max_size = config.SCALES[0][1]
    #     im, im_scale = resize(im, target_size, max_size, stride=config.network.IMAGE_STRIDE)
    #     im_tensor = transform(im, config.network.PIXEL_MEANS)
    #     im_info = np.array([[im_tensor.shape[2], im_tensor.shape[3], im_scale]], dtype=np.float32)
    #     if idx % key_frame_interval == 0:
    #         if idx == 0:
    #             data_oldkey = im_tensor.copy()
    #             data_newkey = im_tensor.copy()
    #             data_cur = im_tensor.copy()
    #         else:
    #             data_oldkey = data_newkey.copy()
    #             data_newkey = im_tensor
    #     else:
    #         data_cur = im_tensor
    #     shape = im_tensor.shape
    #     infer_height = int(np.ceil(shape[2] / 16.0))
    #     infer_width = int(np.ceil(shape[3] / 16.0))
    #     data.append({'data_oldkey': data_oldkey, 'data_newkey': data_newkey, 'data_cur': data_cur, 'im_info': im_info,
    #                  'impression': np.zeros((1, config.network.DFF_FEAT_DIM, infer_height, infer_width)),
    #                  'key_feat_task': np.zeros((1, config.network.DFF_FEAT_DIM, infer_height, infer_width))})