from dienen.utils import *
from .unfolders import unfoldables,general_unfold, sharedlayer_unfolder
import networkx
import copy
from .utils import find_name, layerlist_to_layerdict,name_layer, replace_inputs_by_hierarchy, gather_blocks, uniformize_inputs
import dpath.util

class ArchitectureConfig():
    def __init__(self,config,externals,logger):
        self.config = config
        self.blocks = {}
        self.externals = externals
        self.logger=logger

    def unfold(self,config,hierarchy=None,metadata=None):
        if not hierarchy:
            hierarchy = {}
        unfolded_config = {}
        for layer_name, layer_config in config.items():
            layer_type = layer_config['class']
            if layer_type in unfoldables:
                unfolded_i, hierarchy_i = unfoldables[layer_type](layer_name,layer_config,metadata=metadata,logger=self.logger)
                unfolded_i, hierarchy_i = self.unfold(unfolded_i,hierarchy_i,metadata)
            else:
                unfolded_i, hierarchy_i = general_unfold(layer_name,layer_config,logger=self.logger)
            unfolded_config.update(unfolded_i)
            if hierarchy_i:
                hierarchy.update(hierarchy_i)
        return unfolded_config, hierarchy

    def check_globs(self,config):
        config_ = copy.deepcopy(config)
        for k,v in config_.items():
            layer_input = v['input']
            if not isinstance(layer_input,list):
                layer_input = [layer_input]
            new_layer_inputs = []
            for item in layer_input:
                if item.startswith('glob('):
                    glob_pattern = item[5:-1]
                    new_layer_inputs.extend(dpath.util.search(config_,glob_pattern))
                else:
                    new_layer_inputs.append(item)
            v['input'] = new_layer_inputs
        return config_

    def translate(self):
        processed_config = copy.deepcopy(self.config)
        processed_config, names = layerlist_to_layerdict(processed_config)
        self.blocks = gather_blocks(processed_config)
        metadata = {'blocks': self.blocks, 'externals': self.externals}
        unfolded, hierarchy = self.unfold(processed_config, metadata = metadata)

        unfolded = self.check_globs(unfolded)
        unfolded = sharedlayer_unfolder(unfolded,hierarchy)
        unfolded = replace_inputs_by_hierarchy(unfolded,hierarchy)
        unfolded = uniformize_inputs(unfolded)

        processed_config = unfolded
        self.hierarchy = hierarchy
        return processed_config