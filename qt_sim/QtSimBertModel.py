# encoding=utf8
import sys

from bert4keras.models import BERT
from bert4keras.models import Lambda,Dense, LayerNormalization, Embedding, Activation,BiasAdd
import json

def build_transformer_model_for_qtsim(
    config_path=None,
    checkpoint_path=None,
    model='bert',
    application='encoder',
    return_keras_model=True,
    **kwargs
):
    """根据配置文件构建模型，可选加载checkpoint权重
    """
    configs = {}
    if config_path is not None:
        configs.update(json.load(open(config_path)))
    configs.update(kwargs)
    if 'max_position' not in configs:
        configs['max_position'] = configs.get('max_position_embeddings')
    if 'dropout_rate' not in configs:
        configs['dropout_rate'] = configs.get('hidden_dropout_prob')

    model, application = model.lower(), application.lower()

    models = {
        'qtbert': QtSimBert,
    }
    MODEL = models[model]


    transformer = MODEL(**configs)
    transformer.build(**configs)

    if checkpoint_path is not None:
        transformer.load_weights_from_checkpoint(checkpoint_path)

    if return_keras_model:
        return transformer.model
    else:
        return transformer


class QtSimBert(BERT):
    """构建QtSimBert模型
    """

    def call(self, inputs):
        """定义模型的执行流程
        """
        # Embedding
        outputs = self.apply_embeddings(inputs)
        # Main
        for i in range(self.num_hidden_layers):
            outputs = self.apply_main_layers(outputs, i)
        # Final
        outputs = self.apply_final_layers(outputs)
        return outputs
