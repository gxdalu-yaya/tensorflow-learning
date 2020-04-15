## TensorRT4BERT

TensorRT6.0

```
cls_embed = network.add_slice(input_tensor, start=(0,0,0,0,0), shape=(1,1,hidden_size,1,1), stride=(1,1,1,1,1))

    ''' 
    shape = network.add_shape(input_tensor).get_output(0)

    mask = network.add_constant(shape=(5, ), weights=np.array([1, 0, 0, 1, 1], dtype=np.int32)).get_output(0)
    inv_mask = network.add_constant(shape=(5, ), weights=np.array([0, 1, 1, 0, 0], dtype=np.int32)).get_output(0)

    hidden_size_m = network.add_constant(shape=(5, ), weights=np.array([0, 1, hidden_size, 0, 0], dtype=np.float32)).get_output(0)

    #slice_size = network.add_select(mask, shape, hidden_size).get_output(0)
    slice_size = shape * mask + hidden_size_m * inv_mask 
    '''

    p_w = init_dict["bert_pooler_dense_kernel"]
    p_b = init_dict["bert_pooler_dense_bias"]

    print("cls_embed's shape: {}", cls_embed.shape)
    pool_output = network.add_fully_connected(cls_embed.get_output(0), hidden_size, p_w, p_b)
    pool_data = pool_output.get_output(0)
    tanh = network.add_activation(pool_data, trt.tensorrt.ActivationType.TANH)
    tanh_output = tanh.get_output(0)

    W_out = init_dict[SQD_W]
    B_out = init_dict[SQD_B]

    #W = network.add_constant((1, hidden_size, 18), W_out)
    logits = network.add_fully_connected(tanh_output, 18, W_out, B_out)
    softmax = network.add_softmax(logits.get_output(0))
    set_layer_name(softmax, prefix, "dense")
    return softmax 

```