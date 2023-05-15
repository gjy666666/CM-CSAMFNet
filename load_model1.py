import numpy as np
import torch

from net1 import NestFuse_light2_nodense, Fusion_network
# from onestage_network import ModelOnestage


def load_model1(path):
	# nest_model=Model()
	nb_filter = [112, 160, 208, 256]
	# UNF_model = Model()
	nest_model = NestFuse_light2_nodense(nb_filter)
	nest_model.load_state_dict(torch.load(path))
	para = sum([np.prod(list(p.size())) for p in nest_model.parameters()])
	type_size = 4
	print('Model {} : params: {:4f}M'.format(nest_model._get_name(), para * type_size / 1000 / 1000))
	nest_model.eval()
	nest_model.cuda()
	return nest_model



def load_model2(path):
	nb_filter = [112, 160, 208, 256]
	nest_model=ModelOnestage(nb_filter)
	nest_model.load_state_dict(torch.load(path))
	para = sum([np.prod(list(p.size())) for p in nest_model.parameters()])
	type_size = 4
	print('Model {} : params: {:4f}M'.format(nest_model._get_name(), para * type_size / 1000 / 1000))
	nest_model.eval()
	nest_model.cuda()
	return nest_model


def load_model3(path):
	nb_filter = [112, 160, 208, 256]
	f_type = 'res'
	nest_model=Fusion_network(nb_filter, f_type)
	nest_model.load_state_dict(torch.load(path))
	para = sum([np.prod(list(p.size())) for p in nest_model.parameters()])
	type_size = 4
	print('Model {} : params: {:4f}M'.format(nest_model._get_name(), para * type_size / 1000 / 1000))
	nest_model.eval()
	nest_model.cuda()
	return nest_model

