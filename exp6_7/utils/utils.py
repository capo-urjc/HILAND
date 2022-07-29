# from torchinfo import summary

from model.transfer_learning import BackendClass


def log_model_structure(output_path, model, mode="train"):
    # if mode == "train":
    #     model_text = str(summary(model.model, input_size=tuple([1] + list(dataset.features_dataset.features[0].shape))))
    # elif mode == "backend":
    #     network_class = BackendClass.get_backend_name(config.get_param_value('backend_name'))
    #     if network_class == BackendClass.INCEPTION:
    #         input_size = [299, 299]
    #     else:
    #         input_size = [224, 224]
    #     model_text = str(summary(model.cnn, input_size=tuple([1, 3] + input_size)))
    # elif mode == "test":
    #     model_text = str(summary(model.model, input_size=tuple([1, model.num_features])))
    # else:
    #     raise ValueError()

    if mode == "train" or mode == "test":
        model_text = str(model.model)
    elif mode == "backend":
        model_text = str(model.cnn)
    else:
        raise ValueError()

    summary_file = open(str(output_path), "w")
    summary_file.write(model_text)
    summary_file.close()
