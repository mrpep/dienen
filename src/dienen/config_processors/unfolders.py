import copy
import joblib
from .utils import layerlist_to_layerdict, assign_names_and_inputs, find_name
from kahnfigh import Config, shallow_to_deep, deep_to_shallow, shallow_to_original_keys
import networkx as nx
from ruamel_yaml import YAML

def new_nodes_to_config(new_nodes, name):
    if len(new_nodes) == 1:
        layer_name = list(new_nodes[0].keys())[0]
        new_nodes[0][name] = new_nodes[0][layer_name]
        new_nodes[0].pop(layer_name)
        hierarchy = None
    elif len(new_nodes) > 1:
        child_names = [list(node.keys())[0] for node in new_nodes]
        child_inputs = []
        for node in new_nodes:
            inputs_i = list(node.values())[0]['input']
            if isinstance(inputs_i,list):
                child_inputs.extend(inputs_i)
            else:
                child_inputs.append(inputs_i)
        child_outputs = [child for child in child_names if child not in child_inputs]
        hierarchy_inputs = []
        for node in new_nodes:
            node_name = list(node.keys())[0]
            node_inputs = list(node.values())[0]['input']
            if not isinstance(node_inputs,list):
                node_inputs = [node_inputs]
            if len([k for k in node_inputs if k not in child_names])>0:
                hierarchy_inputs.append(node_name)

        hierarchy = {name: {
        'children': child_names,
        'outputs': child_outputs,
        'inputs': hierarchy_inputs}}

    new_config = {}
    for node in new_nodes:
        node_name = list(node.keys())[0]
        new_config[node_name] = node[node_name]

    return new_config, hierarchy

def general_unfold(name,config,metadata=None,logger=None):
    special_kwargs = ['dropout','batch_norm','activation','order','name','input','batch_norm_params','noise']
    
    unfoldable_nodes = {'dropout':{'layer':'Dropout','rate':'dropout'},
                        'batch_norm': {'layer':'BatchNormalization','kwargs':'batch_norm_params'},
                        'activation':{'layer':'Activation','activation':'activation'},
                        'noise': {'layer':'GaussianNoise','stddev': 'noise'}}

    order = config.get('order',['dropout','batch_norm','main','activation','noise'])
    if config['class'] == 'Activation':
        order = config.get('order',['dropout','batch_norm','main','noise'])
        special_kwargs.remove('activation')
    elif config['class'] == 'LSTM':
        order = config.get('order',['batch_norm','main','activation','noise'])
        special_kwargs.remove('dropout')
    
    new_nodes = []
    last_layer = config['input']
    for tag in order:
        if (tag in config) and (config[tag]):
            layer_type = unfoldable_nodes[tag]['layer']
            node_config = {'class': layer_type,
                           'input': last_layer}

            for param, name_in_config in unfoldable_nodes[tag].items():
                if (param not in ['layer','kwargs']) and (name_in_config in config):
                    node_config[param] = config[name_in_config]
                elif param == 'kwargs' and (name_in_config in config):
                    node_config.update(config[name_in_config])
            layer_name = '{}/{}'.format(name,layer_type)
            new_nodes.append({layer_name: node_config})
            last_layer = layer_name
        elif tag == 'main':
            node_config = copy.deepcopy(config)
            for kwarg in special_kwargs:
                if kwarg in node_config:
                    node_config.pop(kwarg)
            node_config['input'] = last_layer
            layer_name = '{}/{}'.format(name,config['class'])
            #node_config['name'] = layer_name
            new_nodes.append({layer_name: node_config})
            last_layer = layer_name

    new_config, hierarchy = new_nodes_to_config(new_nodes, name)

    return new_config, hierarchy

def stamp_unfold(name,config,metadata=None,logger=None):
    data_to_stamp = config['what']

    max_depth = 1
    for layer in data_to_stamp:
        layer_config = layer[list(layer.keys())[0]]
        for param_name,param in layer_config.items():
            if isinstance(param,list) and len(param)>max_depth:
                max_depth = len(param)

    stamp_times = config.get('times',max_depth)
    new_nodes = []
    
    for i in range(stamp_times):
        layers_mapping = {} 
        for layer in data_to_stamp: 
            layer_type = list(layer.keys())[0] 
            layer_i_config = layer[layer_type].copy()

            layer_original_name = layer_i_config.get('name',find_name(layer_type,list(layers_mapping.keys())))
            if isinstance(layer_original_name,str):
                new_name = '{}_{}/{}'.format(name,i,layer_original_name)
                layers_mapping[layer_original_name] = new_name
                layer_i_config['name'] = new_name
            else:
                layer_i_config['name'] = layer_original_name

            if 'input' in layer_i_config:
                layer_input = layer_i_config['input']
                if not isinstance(layer_input,list):
                    layer_input = [layer_input]
                new_layer_input = []
                for layer_in in layer_input:
                    if isinstance(layer_in, list):
                        new_layer_input.append([layers_mapping[l] if l in layers_mapping else '{}->{}'.format(layers_mapping[l.split('->')[0]],l.split('->')[1]) if ('->' in l and l.split('->')[0] in layers_mapping) else l for l in layer_in])
                    else:
                        if layer_in in layers_mapping:
                            new_layer_input.append(layers_mapping[layer_in])
                        elif '->' in layer_in and layer_in.split('->')[0] in layers_mapping:
                            parent, tag = layer_in.split('->')
                            new_layer_input.append('{}->{}'.format(layers_mapping[parent],tag))
                        else:
                            new_layer_input.append(layer_in)

                layer_i_config['input'] = new_layer_input

            for param_name, param in layer_i_config.items(): 
                if isinstance(param,list) and len(param) == stamp_times and len(param)>1: 
                    layer_i_config[param_name] = param[i]
                elif isinstance(param,list) and len(param) == 1:
                    layer_i_config[param_name] = param[0]
                    
            new_nodes.append({layer_type:layer_i_config})

    #
    if 'input' not in list(new_nodes[0].values())[0]:
        list(new_nodes[0].values())[0]['input'] = config['input']
    
    new_nodes, names = layerlist_to_layerdict(new_nodes,base_name = name,logger=logger)
    new_nodes = [dict([list(node)]) for node in new_nodes.items()]
    new_config, hierarchy = new_nodes_to_config(new_nodes, name)

    return new_config,hierarchy 

def conv_unfold(name,config,metadata=None,logger=None):
    special_kwargs = ['upsampling','pooling_type','pooling','depthwise','separable','transpose','input','name']
    convs = {1:'Conv1D',2:'Conv2D',3:'Conv3D'}
    separable_convs = {1:'SeparableConv1D', 2:'SeparableConv2D'}
    depthwise_convs = {2:'DepthwiseConv2D'}
    transpose_convs = {1:'Conv1DTranspose', 2:'Conv2DTranspose',3:'Conv3DTranspose'}
    upsamples = {1:'UpSampling1D', 2:'UpSampling2D', 3:'UpSampling3D'}
    maxpooling = {1:'MaxPooling1D', 2:'MaxPooling2D', 3:'MaxPooling3D'}
    avgpooling = {1:'AveragePooling1D', 2:'AveragePooling2D', 3:'AveragePooling3D'}

    order = config.get('order',['dropout','batch_norm','conv','activation','noise','upsampling','pooling'])
    config['order'] = copy.deepcopy(order)

    kernel_size = config['kernel_size']
    if not isinstance(kernel_size,list):
        kernel_size = [kernel_size]

    kernel_dims = len(kernel_size)

    pooling_type = config.get('pooling_type','max')
    upsample = config.get('upsampling', False)
    pooling = config.get('pooling', False)

    pool_dict = {'max':maxpooling,
                'average':avgpooling}

    conv_dict = {'depthwise': depthwise_convs,
                 'separable':separable_convs,
                 'transpose':transpose_convs}
    
    convtypes = []
    for k in config:
        if k in conv_dict:
            convtypes.append(k)
            conv_layer = conv_dict[k].get(kernel_dims,None)
            if conv_layer is None:
                raise Exception("{} convolution layer is not available for {} dimensions".format(k,kernel_dims))

    if len(convtypes) > 1:
        raise Exception("Setting {} at the same time is not allowed".format(str(convtypes)))
    elif len(convtypes) == 0:
        conv_layer = convs[kernel_dims]

    unfoldable_nodes = {'upsampling':{'layer':upsamples[kernel_dims],'size':upsample},
                        'conv':{'layer':conv_layer},
                        'pooling':{'layer':pool_dict[pooling_type][kernel_dims],'pool_size':pooling}}
    
    new_nodes = []

    last_layer = config['input']
    for tag in order:
        if tag == 'conv':
            main_layer_config = config.copy()
            for k in special_kwargs:
                if k in main_layer_config.keys():
                    main_layer_config.pop(k)
            order_ = [k for k in order if k not in ['upsampling','pooling']]
            main_layer_config['order'] = ['main' if k == 'conv' else k for k in order_]

            layer_name = '{}/{}'.format(name,conv_layer)
            main_layer_config['class'] = conv_layer
            main_layer_config['input'] = last_layer
            new_nodes.append({layer_name: main_layer_config})
            last_layer = layer_name

        elif tag in config and config[tag] and tag in unfoldable_nodes:
            layer_type = unfoldable_nodes[tag]['layer']
            node_config = {'class': layer_type,
                           'input': last_layer}

            for param, name_in_config in unfoldable_nodes[tag].items():
                if (param != 'layer') and (name_in_config in config):
                    node_config[param] = config[name_in_config]

            layer_name = '{}/{}'.format(name,layer_type)
            new_nodes.append({layer_name: node_config})
            last_layer = layer_name

    new_config, hierarchy = new_nodes_to_config(new_nodes, name)

    return new_config, hierarchy

def layer_add_basename(block_layers, base_name):
    rename_dict = {}
    for layer in block_layers:
        layer_type = list(layer.keys())[0]
        layer_config = list(layer.values())[0]
        layer_name = layer_config.get('name',None)
        if layer_name:
            new_layer_name = '{}/{}'.format(base_name,layer_name)
            rename_dict[layer_name] = new_layer_name
            layer_config['name'] = new_layer_name

    for layer in block_layers:
        layer_type = list(layer.keys())[0]
        layer_config = list(layer.values())[0]
        layer_input = layer_config.get('input',None)
        if layer_input:
            layer_input = layer_input if isinstance(layer_input,list) else [layer_input]
            layer_input = [rename_dict[layer_in] if layer_in in rename_dict else layer_in for layer_in in layer_input]
            layer_config['input'] = layer_input

    return block_layers

def block_unfold(name,config,metadata=None,logger=None):
    metadata = copy.deepcopy(metadata)

    available_blocks = metadata['blocks']
    block_config = available_blocks[config['block']]
    block_layers = block_config['block']
    block_inputs = ['{}/{}'.format(name,layer_name) for layer_name in block_config['block_in']]
    config_inputs = config['input'] if isinstance(config['input'],list) else [config['input']]
    
    block_layers = layer_add_basename(block_layers,name)

    #Assign corresponding inputs:
    i = 0
    for layer in block_layers:
        layer_type = list(layer.keys())[0]
        layer_config = list(layer.values())[0]
        if 'name' in layer_config:
            if layer_config['name'] in block_inputs and len(config_inputs) == 1:
                layer_config['input'] = config_inputs[0]
            elif layer_config['name'] in block_inputs and len(config_inputs) > 1 and len(config_inputs) == len(block_inputs):
                layer_config['input'] = config_inputs[i]
                i+=1
            elif layer_config['name'] not in block_inputs:
                pass
            else:
                raise Exception('Input sizes in block not matching')
        else:
            pass

    new_nodes, names = layerlist_to_layerdict(block_layers,base_name = name,logger=logger)
    new_nodes = [dict([list(node)]) for node in new_nodes.items()]
    new_config, hierarchy = new_nodes_to_config(new_nodes, name)

    return new_config, hierarchy

def sharedlayer_unfolder(layers,hierarchy,logger=None):
    new_nodes = {}
    for layer_name, layer_config in layers.items():
        if (layer_config['class'] == 'SharedLayer') and (layer_config['layer'] in hierarchy):
            target_layer = layer_config['layer']
            children = hierarchy[target_layer]['children']
            hierarchy_in = hierarchy[target_layer]['inputs']
            hierarchy_out = hierarchy[target_layer]['outputs']
            new_hierarchy_children = []
            new_hierarchy_inputs = []
            new_hierarchy_outputs = []
            for child in children:
                new_name = layer_name.split(target_layer)[0] + child
                new_hierarchy_children.append(new_name)
                if child in hierarchy_in:
                    input_name = layer_config['input']
                    new_hierarchy_inputs.append(new_name)
                elif child in hierarchy_out:
                    input_name = [layer_name + child_in.split(target_layer)[-1] for child_in in layers[child]['input']]
                    new_hierarchy_outputs.append(new_name)
                else:
                    input_name = [layer_name + child_in.split(target_layer)[-1] for child_in in layers[child]['input']]
                new_node = {
                'class': 'SharedLayer',
                'layer': child,
                'input': input_name
                }
                new_nodes[new_name] = new_node
            hierarchy[layer_name] = {'children': new_hierarchy_children,
            'inputs': new_hierarchy_inputs, 
            'outputs': new_hierarchy_outputs}
        else:
            new_nodes[layer_name] =  layer_config
    return new_nodes

def append_or_extend(list_to_grow,elem):
    if isinstance(elem,list):
        list_to_grow.extend(elem)
    else:
        list_to_grow.append(elem)

def pop_dictreturn(dictionary,key): 
    dictionary = copy.deepcopy(dictionary)
    dictionary.pop(key) 
    return dictionary

def external_unfold(name,config,metadata=None,logger=None):
    external_models = metadata['externals']['Models']
    external_model_name = config['model']
    external_layer_name = config.get('layer',None)
    external_last_layer = config.get('up_to',None)
    external_first_layer = config.get('from',None)
    external_exclude_inputs = config.get('exclude_input',True)
    external_reset_weights = config.get('reset_weights',False)
    external_mods = config.get('mods',None)
    external_time_distributed = config.get('time_distributed',False)

    trainable_from = config.get('trainable_from',None)
    trainable_layers = config.get('trainable_layers',None)
    trainable = config.get('trainable',True)

    import dienen

    if isinstance(external_models[external_model_name],str):
        external_model = joblib.load(external_models[external_model_name])
        external_model_architecture = external_model['unfolded_config']
        external_hierarchy = external_model['hierarchy']
    elif isinstance(external_models[external_model_name],dienen.core.model.Model):
        external_model = external_models[external_model_name]
        external_model_architecture = external_model.core_model.processed_config
        external_hierarchy = external_model.architecture_config.hierarchy

    if external_mods:
        import fnmatch
        yaml_loader = YAML()
        m_conf = Config(external_model_architecture)
        original_keys = list(m_conf.keys())
        deep_conf = Config(shallow_to_deep(m_conf))
        for mod in external_mods:
            mod_key = list(mod.keys())[0]
            mod_value = mod[mod_key]
            if mod_key == 'delete':
                deep_conf.pop(mod_value)
                if mod_value in original_keys:
                    original_keys.remove(mod_value)
            elif '*' in mod_key:
                mod_key = mod_key.lstrip('/')
                found_paths = [k for k in deep_conf.to_shallow().keys() if fnmatch.fnmatch(k,mod_key)]
                for k in found_paths:
                    k = k.replace('.','/')
                    if isinstance(mod_value,str):
                        deep_conf[k] = yaml_loader.load(mod_value)
                    else:
                        deep_conf[k] = mod_value
            else:
                mod_key = mod_key.replace('.','/')
                if mod_key.split('/')[0] not in deep_conf.keys(): #This means we are adding a new layer
                    layer_name = mod_key.split('/')[0]
                    original_keys.append(layer_name)
                    deep_conf['{}/name'.format(layer_name)]=layer_name
                if isinstance(mod_value,str):
                    deep_conf[mod_key] = yaml_loader.load(mod_value)
                else:
                    deep_conf[mod_key] = mod_value

        external_model_architecture = shallow_to_original_keys(deep_conf.to_shallow(),original_keys)
    unfolded_layers = []

    g = nx.DiGraph()
    for layer_name, layer_config in external_model_architecture.items():
        if layer_config['class'] != 'Input':
            if isinstance(layer_config['input'],list):
                for k in layer_config['input']:
                    g.add_edge(k,layer_name)
            else:
                g.add_edge(layer_config['input'],layer_name)

    if external_layer_name and external_layer_name not in g.nodes() and external_layer_name in external_hierarchy:
        external_layer_name = external_hierarchy[external_layer_name]['output'][0]
    if external_last_layer and external_last_layer not in g.nodes() and external_last_layer in external_hierarchy:
        external_last_layer = external_hierarchy[external_last_layer]['output'][0]
    if external_first_layer and external_first_layer not in g.nodes() and external_first_layer in external_hierarchy:
        external_first_layer = external_hierarchy[external_first_layer]['input'][0]

    if external_last_layer and not external_first_layer and not external_layer_name:
        layers_subset = list(nx.ancestors(g,external_last_layer)) + [external_last_layer]
    elif external_first_layer and not external_last_layer and not external_layer_name:
        layers_subset = list(nx.dfs_successors(g,external_first_layer))
    elif external_first_layer and external_last_layer and not external_layer_name:
        layers_subset = list(set(nx.dfs_successors(g,external_first_layer)).intersection(nx.ancestors(g,external_last_layer))) + [external_last_layer]
    elif external_layer_name:
        layers_subset = [external_layer_name]
    else:
        layers_subset = list(external_model_architecture.keys())
    if external_exclude_inputs:
        layers_subset = [layer for layer in layers_subset if external_model_architecture[layer]['class'] != 'Input']
    
    unfolded_layers = [external_model_architecture[l] for l in layers_subset]
    in_layers = []
    for l in unfolded_layers:
        ins = l['input']
        if not isinstance(ins,list):
            ins = [ins]
        for x in ins:
            if x not in layers_subset:
                in_layers.append(l['name'])

    #in_layers = [l['name'] for l in unfolded_layers if l['input'] not in layers_subset]
    new_nodes = [{layer['name']: pop_dictreturn(layer,'name')} for layer in unfolded_layers]
    new_config, hierarchy = new_nodes_to_config(new_nodes,name)

    if 'input' in config:
        for layer in in_layers:
            new_config[layer]['input'] = config['input']

    if trainable and not trainable_from and not trainable_layers:
        trainable_layers = [layer_name for layer_name, layer in new_config.items() if layer['class'] != 'Input']
    elif trainable_from:
        if trainable_from not in external_model_architecture and trainable_from in external_hierarchy:
            trainable_from = external_hierarchy[trainable_from]['inputs'][0]
        trainable_layers = nx.dfs_successors(g,trainable_from)
    elif not trainable:
        trainable_layers = []

    #Set trainable false in non-trainable layers    
    for layer_name, layer_config in new_config.items():
        if layer_name not in trainable_layers:
            layer_config['trainable'] = False

    #from IPython import embed
    #embed()

    if external_time_distributed:
        for layer_name, layer_config in new_config.items():
            layer_config['time_distributed'] = True

    #Make each layer search for the weights from external model
    if isinstance(external_reset_weights,bool) and not external_reset_weights:
        external_weight_layers = [layer_name for layer_name, layer in new_config.items() if layer['class'] != 'Input']
    elif isinstance(external_reset_weights,list):
        external_weight_layers = external_reset_weights
    for layer in external_weight_layers:
        new_config[layer]['from_model'] = external_model_name
        new_config[layer]['from_layer'] = layer

    return new_config, hierarchy

unfoldables = {'Conv': conv_unfold,
               'Stamp': stamp_unfold,
               'Block': block_unfold,
               'External': external_unfold}

