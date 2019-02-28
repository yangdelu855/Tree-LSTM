from module import LstmModel, BilstmModel, CnnModel, SumModel, TreeLstm


class ModelFactory:

    def get_model(self, data):
        if data.HP_encoder_type == 'lstm':
            return LstmModel(data)
        elif data.HP_encoder_type == 'bilstm':
            return BilstmModel(data)
        elif data.HP_encoder_type == 'cnn':
            return CnnModel(data)
        elif data.HP_encoder_type == 'sum':
            return SumModel(data)
        elif data.HP_encoder_type == 'treelstm':
            return TreeLstm(data)
