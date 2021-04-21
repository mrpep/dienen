import networkx as nx
from .layer import Layer
import dienen.layers
from dienen.config_processors.utils import single_elem_list_to_elem
import tensorflow.keras.layers
import copy
import tensorflow as tf
import joblib

from IPython import embed

def recursive_replace(names,hierarchy,criteria='outputs'):
        return [recursive_replace(hierarchy[k][criteria], hierarchy) if k in hierarchy else k for k in names]

class Architecture:
	"""
	Processes a config with layers specification.
	"""
    
	def __init__(self,architecture_config,layer_modules=None, inputs=None, outputs=None, externals=None, processed_config=None):

		self.architecture_config = architecture_config
		self.processed_config = self.architecture_config.translate()

		if processed_config is not None:
			self.processed_config = processed_config

		self.hierarchy = self.architecture_config.hierarchy
		self.externals = externals

		self.inputs = single_elem_list_to_elem(recursive_replace(inputs,self.hierarchy,criteria='inputs'))
		self.outputs = single_elem_list_to_elem(recursive_replace(outputs,self.hierarchy,criteria='outputs'))
		
		self.inputs = self.inputs if isinstance(self.inputs,list) else [self.inputs]
		self.outputs = self.outputs if isinstance(self.outputs,list) else [self.outputs]

		if not layer_modules:
			layer_modules = [dienen.layers, tensorflow.keras.layers]

		self.layer_modules = layer_modules
		self.find_external_weights()
		self.create_layers()
		self.model = self.make_network()

	def find_external_weights(self):
		self.weights_to_assign = {}
		if self.externals and 'Models' in self.externals:
			external_weights = {}
			for model_name, model_path in self.externals['Models'].items():
				if isinstance(model_path,str):
					external_weights[model_name] = joblib.load(model_path)['weights']
				else:
					external_weights[model_name] = model_path.get_weights()

			for layer_name, layer in self.processed_config.items():
				if 'from_model' in layer:
					if layer['from_layer'] in external_weights[layer['from_model']]:
						self.weights_to_assign[layer_name] = external_weights[layer['from_model']][layer['from_layer']]
					else:
						raise Exception('Layer {} does not exist in external model {}'.format(layer['from_layer'],layer['from_model']))
					layer.pop('from_model')
					layer.pop('from_layer')

	def create_layers(self):
		self.layers = {}
		self.shared_layers = {}
		for layer_name, layer_config in self.processed_config.items():
			if layer_config['class'] != 'SharedLayer':
				layer_config['name'] = layer_name
				self.layers[layer_name] = Layer()
				self.layers[layer_name].set_layer_modules(self.layer_modules)
				self.layers[layer_name].set_config(layer_config) 
				self.layers[layer_name] = self.layers[layer_name].create()

		for layer_name, layer_config in self.processed_config.items():
			if layer_config['class'] == 'SharedLayer':
				target_layer = layer_config['layer']
				self.shared_layers[layer_name] = target_layer
				self.layers[layer_name] = self.layers[target_layer]

	def graph_from_layers(self,layers,inputs):
		graph = nx.DiGraph()
		for layer_name, layer_config in layers.items():
			layer_inputs = layer_config['input']
			if not isinstance(layer_inputs, list):
				layer_inputs = [layer_inputs]
			if 'mask' in layer_config:
				layer_inputs.append(layer_config['mask'])
			for layer_input in layer_inputs:
				if layer_input != 'unconnected' and layer_name not in inputs:
					try:
						graph.add_edge(layer_input,layer_name)
					except:
						raise Exception('Edge cant be added')
		return graph

	def find_dependency_order(self,config,inputs,outputs):
		whole_graph = self.graph_from_layers(config,inputs)
		dependency_order = list(nx.topological_sort(whole_graph))
		input_idx = min([dependency_order.index(input_i) for input_i in inputs])
		output_idx = max([dependency_order.index(input_i) for input_i in outputs])
		pruned_nodes = dependency_order[input_idx:output_idx+1]
		only_needed_layers = {k: config[k] for k in pruned_nodes}
		pruned_graph = self.graph_from_layers(only_needed_layers,inputs)
		connected_subgraphs = [pruned_graph.subgraph(c) for c in nx.connected_components(pruned_graph.to_undirected())]
		if len(connected_subgraphs)>1:
			reachable_subgraph = [g for g in connected_subgraphs if all(node in g.nodes for node in inputs + outputs)][0]
		else:
			reachable_subgraph = connected_subgraphs[0]
		dependency_order = list(nx.topological_sort(reachable_subgraph))
		return dependency_order

	def make_network(self):
		dependency_order = self.find_dependency_order(self.processed_config,self.inputs,self.outputs)
		self.tensors = {}
		shared_layers_counts = {}
		for layer in dependency_order:
			if layer not in self.inputs:
				layer_inputs = self.processed_config[layer]['input']
				layer_inputs = layer_inputs if isinstance(layer_inputs,list) else [layer_inputs]
				layer_inputs = [self.tensors[inp] for inp in layer_inputs]
				layer_mask = self.processed_config[layer].get('mask', None)
				if len(layer_inputs) == 1:
					layer_inputs = layer_inputs[0]
				if layer in self.shared_layers:
					target_layer = self.shared_layers[layer]
					if target_layer in shared_layers_counts:
						shared_layers_counts[target_layer] +=1
					else:
						shared_layers_counts[target_layer] = 0
					if layer_mask is not None:
						layer_mask = self.tensors[layer_mask]
						self.tensors[layer] = self.layers[layer](layer_inputs, mask = layer_mask)
					else:
						self.tensors[layer] = self.layers[layer](layer_inputs)
					self.tensors[layer] = self.layers[layer].get_output_at(shared_layers_counts[target_layer])
				else:
					try:
						if layer_mask is not None:
							layer_mask = self.tensors[layer_mask]
							layer_mask = tf.cast(layer_mask,tf.bool)
							self.tensors[layer] = self.layers[layer](layer_inputs, mask=layer_mask)
						else:
							self.tensors[layer] = self.layers[layer](layer_inputs)
					except Exception as e:
						embed()
						raise Exception('Could not connect layer {} with its inputs {}. {}'.format(layer,self.processed_config[layer]['input'],e))
			else:
				self.tensors[layer] = self.layers[layer]

		input_tensors = [self.tensors[in_tensor] for in_tensor in self.inputs]
		output_tensors = [self.tensors[out_tensor] for out_tensor in self.outputs]
		
		model = tf.keras.Model(inputs=input_tensors,outputs=output_tensors)
		for layer_name,weights in self.weights_to_assign.items():
			model.get_layer(layer_name).set_weights(weights)

		return model


		


				




		