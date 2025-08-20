def build_cmr_cnn_model(input_shape):
    inputs = define_input_layer(input_shape)

    conv_layers = add_multiscale_conv_layers(inputs)
    residual_blocks = add_residual_blocks(conv_layers)
    pooled = global_average_pooling(residual_blocks)
    dense_layers = add_dense_layers(pooled)

    outputs = define_output_layer(dense_layers)
    model = compile_model(inputs, outputs)
    return model

model = build_cmr_cnn_model(input_shape)
model.fit(X_train, y_train, validation_data=(X_val, y_val))
metrics = model.evaluate(X_test, y_test)
